#!/usr/bin/env python3
"""Validate or publish every Darshan member of one tar archive."""

import argparse
import hashlib
import json
import os
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_name(name):
    pure = PurePosixPath(name)
    if pure.is_absolute() or not pure.parts or any(part in ("", ".", "..") for part in pure.parts):
        raise ValueError(f"unsafe archive member path: {name!r}")
    return pure


def inspect_or_extract(archive, output_dir, staging_root, extract):
    archive = Path(archive).resolve()
    output_dir = Path(output_dir).resolve()
    if not archive.is_file() or archive.stat().st_size == 0:
        raise FileNotFoundError(f"archive missing or empty: {archive}")
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    seen = set()
    missing = []
    stage_dir = None
    with tarfile.open(archive, "r:gz") as tar:
        members = []
        for member in tar.getmembers():
            name = safe_name(member.name)
            if member.isfile() and name.suffix == ".darshan":
                if name.as_posix() in seen:
                    raise ValueError(f"duplicate Darshan member: {name}")
                seen.add(name.as_posix())
                members.append((member, name))
        if not members:
            raise ValueError(f"archive has no Darshan members: {archive}")
        for member, name in members:
            source = tar.extractfile(member)
            if source is None:
                raise ValueError(f"cannot read archive member: {name}")
            digest = hashlib.sha256()
            target = output_dir.joinpath(*name.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            existing_matches = False
            if target.exists():
                if not target.is_file() or target.stat().st_size != member.size:
                    raise ValueError(f"existing output differs in type or size: {target}")
                existing_matches = sha256_file(target) == sha256_stream(source, digest)
                if not existing_matches:
                    raise ValueError(f"existing output differs from archive: {target}")
                member_sha = digest.hexdigest()
            elif not extract:
                member_sha = sha256_stream(source, digest)
                missing.append(name.as_posix())
            else:
                if stage_dir is None:
                    Path(staging_root).mkdir(parents=True, exist_ok=True)
                    stage_dir = Path(tempfile.mkdtemp(prefix="unpack_", dir=staging_root))
                staged = stage_dir.joinpath(*name.parts)
                staged.parent.mkdir(parents=True, exist_ok=True)
                with open(staged, "xb") as output:
                    while True:
                        block = source.read(1024 * 1024)
                        if not block:
                            break
                        digest.update(block)
                        output.write(block)
                    output.flush()
                    os.fsync(output.fileno())
                if staged.stat().st_size != member.size:
                    raise ValueError(f"staged member has wrong size: {name}")
                member_sha = digest.hexdigest()
                os.rename(staged, target)
            records.append({"path": name.as_posix(), "bytes": member.size, "sha256": member_sha})
    inventory = {
        "schema_version": 1,
        "archive": str(archive),
        "archive_sha256": sha256_file(archive),
        "member_count": len(records),
        "members": records,
    }
    if missing:
        return "incomplete", inventory, missing
    if extract:
        inventory_path = output_dir / ".logs.tar.gz.inventory.json"
        encoded = json.dumps(inventory, indent=2, sort_keys=True) + "\n"
        if inventory_path.exists():
            if inventory_path.read_text() != encoded:
                raise ValueError(f"existing inventory differs: {inventory_path}")
        else:
            inventory_path.write_text(encoded)
    return "complete" if stage_dir is None else "extracted", inventory, []


def sha256_stream(stream, digest):
    while True:
        block = stream.read(1024 * 1024)
        if not block:
            break
        digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--staging-root", default=".codex-trash/unpack_staging")
    parser.add_argument("--extract", action="store_true")
    args = parser.parse_args()
    try:
        status, inventory, missing = inspect_or_extract(
            args.archive, args.output_dir, args.staging_root, args.extract)
    except (OSError, tarfile.TarError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": status, "members": inventory["member_count"],
                      "missing": len(missing)}, sort_keys=True))
    return 3 if status == "incomplete" else 0


if __name__ == "__main__":
    sys.exit(main())
