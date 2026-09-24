"""Reader for Nek5000 field files (``<case><fid>.f<nnnnn>``), used for the scientific
equivalence check of the application study.

Layout, from ``core/prepost.f`` of Nek5000 (``mfo_write_hdr``, ``mfo_outv``, ``mfo_outs``):

* 132-byte ASCII header written with the Fortran format
  ``'#std',1x,i1,1x,i2,1x,i2,1x,i2,1x,i10,1x,i10,1x,e20.13,1x,i9,1x,i6,1x,i6,1x,10a,1pe15.7,1x,l1,1x,a4``:
  word size, nx, ny, nz, elements in this file, elements in the run, time, step, file id,
  number of files, the variable code (X mesh, U velocity, P pressure, T temperature,
  Snn passive scalars), the reference pressure, a pressure-mesh flag and the h-refine code;
* a 4-byte float test pattern 6.54321 (byte order);
* the global element numbers of this file, one int32 each;
* the data, one variable after another in the order of the variable code; vector variables
  hold x, y, z blocks per element, scalar variables one block per element, each block
  nx*ny*nz words in the header's word size;
* for three-dimensional runs, per-element metadata after the data (``mfo_mdatav``,
  ``mfo_mdatas``): for every variable in the same order, the minimum and maximum of each
  component per element as ``real*4`` (six values per element for a vector, two for a
  scalar), written for this file's elements in the same element order.

With ``writeNFiles = 1`` the run writes one file per checkpoint through MPI-IO; with
``writeNFiles = N`` each of N I/O ranks writes the elements of its group to its own file.
``reassemble`` puts the files of one checkpoint back into global element order so that both
organizations can be compared element by element.
"""
import hashlib
import math
import struct
from pathlib import Path

import numpy as np

HEADER_BYTES = 132
TEST_PATTERN = 6.54321


class FldFormatError(ValueError):
    """The file does not follow the documented layout."""


def parse_header(raw):
    """Typed header fields from the 132 header bytes."""
    if len(raw) != HEADER_BYTES:
        raise FldFormatError(f"header must be {HEADER_BYTES} bytes, got {len(raw)}")
    text = raw.decode("ascii", errors="strict")
    if not text.startswith("#std"):
        raise FldFormatError(f"header does not start with #std: {text[:12]!r}")
    fields = text[4:83].split()
    if len(fields) != 10:
        raise FldFormatError(f"expected 10 numeric header fields, got {len(fields)}: {text[:83]!r}")
    varcode = text[83:93].strip()
    p0th = float(text[93:108])
    press_mesh = text[109] == "T"
    return {
        "wdsize": int(fields[0]),
        "nx": int(fields[1]), "ny": int(fields[2]), "nz": int(fields[3]),
        "nelo": int(fields[4]), "nelog": int(fields[5]),
        "time": float(fields[6]), "istep": int(fields[7]),
        "fid": int(fields[8]), "nfileo": int(fields[9]),
        "varcode": varcode, "p0th": p0th, "press_mesh": press_mesh,
        "hrefcuts": text[111:115].strip(),
    }


def variables(varcode, ndim=3):
    """(name, components) in file order for a variable code such as ``XUP`` or ``UPT``."""
    out = []
    i = 0
    while i < len(varcode):
        c = varcode[i]
        if c == "X":
            out.append(("X", ndim))
        elif c == "U":
            out.append(("U", ndim))
        elif c == "P":
            out.append(("P", 1))
        elif c == "T":
            out.append(("T", 1))
        elif c == "S":
            count = int(varcode[i + 1:i + 3])
            for k in range(count):
                out.append((f"S{k + 1:02d}", 1))
            i += 2
        else:
            raise FldFormatError(f"unknown variable code {c!r} in {varcode!r}")
        i += 1
    return out


def read_file(path):
    """Header, global element numbers and data of one field file.

    Returns a dict with ``header``, ``elements`` (int32 array of global numbers, 1-based)
    and ``data``: {variable: array of shape (nelo, components, nxyz)} in the file's word size.
    """
    path = Path(path)
    raw = path.read_bytes()
    header = parse_header(raw[:HEADER_BYTES])
    pattern = struct.unpack("<f", raw[HEADER_BYTES:HEADER_BYTES + 4])[0]
    if not math.isclose(pattern, TEST_PATTERN, rel_tol=1e-6):
        raise FldFormatError(f"{path.name}: test pattern {pattern} is not {TEST_PATTERN} (byte order)")
    nelo = header["nelo"]
    off = HEADER_BYTES + 4
    elements = np.frombuffer(raw, dtype="<i4", count=nelo, offset=off)
    off += 4 * nelo
    nxyz = header["nx"] * header["ny"] * header["nz"]
    dtype = {4: "<f4", 8: "<f8"}[header["wdsize"]]
    ndim = 3 if header["nz"] > 1 else 2
    data = {}
    for name, ncomp in variables(header["varcode"], ndim):
        count = nelo * ncomp * nxyz
        block = np.frombuffer(raw, dtype=dtype, count=count, offset=off)
        if block.size != count:
            raise FldFormatError(f"{path.name}: variable {name} truncated ({block.size} of {count} words)")
        data[name] = block.reshape(nelo, ncomp, nxyz)
        off += count * header["wdsize"]
    if ndim == 3:
        for name, ncomp in variables(header["varcode"], ndim):
            count = nelo * 2 * ncomp
            block = np.frombuffer(raw, dtype="<f4", count=count, offset=off)
            if block.size != count:
                raise FldFormatError(f"{path.name}: metadata of {name} truncated ({block.size} of {count} words)")
            data[f"{name}_minmax"] = block.reshape(nelo, 1, 2 * ncomp)
            off += 4 * count
    if off != len(raw):
        raise FldFormatError(f"{path.name}: {len(raw) - off} trailing bytes after the documented layout")
    return {"path": str(path), "header": header, "elements": elements, "data": data}


def reassemble(paths):
    """One checkpoint from its file(s): data in ascending global element order.

    Every file must carry the same time, step, mesh size and variable code, and the files
    together must cover elements 1..nelog exactly once.
    """
    parts = [read_file(p) for p in paths]
    if not parts:
        raise FldFormatError("no files given")
    first = parts[0]["header"]
    keys = ("wdsize", "nx", "ny", "nz", "nelog", "time", "istep", "varcode", "nfileo")
    for part in parts[1:]:
        for key in keys:
            if part["header"][key] != first[key]:
                raise FldFormatError(f"{part['path']}: header {key} {part['header'][key]!r} differs from "
                                     f"{parts[0]['path']} ({first[key]!r})")
    if len(parts) != first["nfileo"]:
        raise FldFormatError(f"checkpoint step {first['istep']}: {len(parts)} files, header says {first['nfileo']}")
    elements = np.concatenate([p["elements"] for p in parts])
    if elements.size != first["nelog"] or set(elements.tolist()) != set(range(1, first["nelog"] + 1)):
        raise FldFormatError(f"checkpoint step {first['istep']}: element numbers do not cover 1..{first['nelog']} "
                             f"exactly once ({elements.size} entries)")
    order = np.argsort(elements, kind="stable")
    data = {}
    for name in parts[0]["data"]:
        data[name] = np.concatenate([p["data"][name] for p in parts])[order]
    return {"header": first, "files": [p["path"] for p in parts], "per_file_nelo": [p["header"]["nelo"] for p in parts],
            "data": data}


def summarize(checkpoint):
    """Typed, comparable summary of one reassembled checkpoint: per variable component the
    element count, a canonical SHA-256 of the data in global element order, the sum
    (``math.fsum`` over the components, order independent up to rounding of the final sum),
    the minimum and the maximum."""
    header = checkpoint["header"]
    out = {"istep": header["istep"], "time": header["time"], "nelog": header["nelog"],
           "nfileo": header["nfileo"], "varcode": header["varcode"], "wdsize": header["wdsize"],
           "nxyz": header["nx"] * header["ny"] * header["nz"], "files": len(checkpoint["files"]),
           "per_file_elements": checkpoint["per_file_nelo"], "variables": {}}
    for name, array in checkpoint["data"].items():
        for comp in range(array.shape[1]):
            values = np.ascontiguousarray(array[:, comp, :])
            if not np.isfinite(values).all():
                raise FldFormatError(f"step {header['istep']} variable {name}[{comp}] holds non-finite values")
            flat = values.astype("<f8", copy=False) if values.dtype != np.dtype("<f8") else values
            out["variables"][f"{name}{comp}"] = {
                "sha256": hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest(),
                "sum": math.fsum(flat.ravel().tolist()),
                "min": float(flat.min()), "max": float(flat.max()),
                "count": int(flat.size),
            }
    return out


def write_file(path, header, elements, data):
    """Write one field file in the documented layout (used by the tests to build fixtures)."""
    ndim = 3 if header["nz"] > 1 else 2
    varcode = header["varcode"]
    text = "#std {:1d} {:2d} {:2d} {:2d} {:10d} {:10d} {:20.13E} {:9d} {:6d} {:6d} {:<10s}{:15.7E} {} {:<4s}".format(
        header["wdsize"], header["nx"], header["ny"], header["nz"], header["nelo"], header["nelog"],
        header["time"], header["istep"], header["fid"], header["nfileo"], varcode, header.get("p0th", 1.0),
        "T" if header.get("press_mesh") else "F", header.get("hrefcuts", ""))
    raw = bytearray(text.encode("ascii").ljust(HEADER_BYTES, b" "))
    raw += struct.pack("<f", TEST_PATTERN)
    raw += np.asarray(elements, dtype="<i4").tobytes()
    dtype = {4: "<f4", 8: "<f8"}[header["wdsize"]]
    for name, _ in variables(varcode, ndim):
        raw += np.ascontiguousarray(data[name], dtype=dtype).tobytes()
    if ndim == 3:
        for name, ncomp in variables(varcode, ndim):
            block = data.get(f"{name}_minmax")
            if block is None:
                values = np.asarray(data[name], dtype="<f8")
                block = np.stack([values.min(axis=2), values.max(axis=2)], axis=2).reshape(len(values), 1, 2 * ncomp)
            raw += np.ascontiguousarray(block, dtype="<f4").tobytes()
    Path(path).write_bytes(bytes(raw))
