"""WRF evidence contract: namelist and rsl parsing, history naming, per-variable summaries of a
small NetCDF fixture, the correctness and I/O validation checks, and the registry entry."""
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from scripts.apps.wrf import evidence  # noqa: E402

NAMELIST = """ &time_control
 run_days                            = 0,
 run_hours                           = 1,
 run_minutes                         = 0,
 run_seconds                         = 0,
 start_year                          = 2019,
 start_month                         = 11,
 start_day                           = 26,
 start_hour                          = 23,
 start_minute                        = 00,
 start_second                        = 00,
 end_year                            = 2019,
 end_month                           = 11,
 end_day                             = 27,
 end_hour                            = 00,
 end_minute                          = 00,
 end_second                          = 00,
 history_interval_m                  = 60,
 frames_per_outfile                  = 1,
 restart                             = .true.,
 io_form_history                     = 2,
 io_form_restart                     = 2,
 io_form_boundary                    = 2,
 nocolons                            = .true.,
 use_netcdf_classic                  = .true.,
 /
 &domains
 time_step                           = 72,
 max_dom                             = 1,
 e_we                                = 425,
 e_sn                                = 300,
 e_vert                              = 50,
 /
 &physics
 physics_suite                       = 'conus'
 /
"""


def _rsl(success=True, steps=50, write_s=8.29863):
    lines = []
    import datetime
    t = datetime.datetime(2019, 11, 26, 23, 0, 0)
    for _ in range(steps):
        t += datetime.timedelta(seconds=72)
        lines.append(f"Timing for main: time {t:%Y-%m-%d_%H:%M:%S} on domain   1:    0.29277 elapsed seconds")
    lines.append(f"Timing for Writing wrfout_d01_2019-11-27_00_00_00 for domain        1:    {write_s} elapsed seconds")
    if success:
        lines.append("d01 2019-11-27_00:00:00 wrf: SUCCESS COMPLETE WRF")
    return "\n".join(lines) + "\n"


def _write_history(path, data_model, seed=0, values=None):
    import netCDF4

    rng = np.random.default_rng(seed)
    with netCDF4.Dataset(str(path), "w", format=data_model) as ds:
        ds.createDimension("Time", 1)
        ds.createDimension("DateStrLen", 19)
        ds.createDimension("south_north", 3)
        ds.createDimension("west_east", 4)
        ds.TITLE = "fixture"
        times = ds.createVariable("Times", "S1", ("Time", "DateStrLen"))
        times[0] = netCDF4.stringtochar(np.array(["2019-11-27_00:00:00"], dtype="S19"))
        t2 = ds.createVariable("T2", "f4", ("Time", "south_north", "west_east"))
        t2[0] = values if values is not None else rng.normal(size=(3, 4)).astype("f4")
        lu = ds.createVariable("LU_INDEX", "i4", ("Time", "south_north", "west_east"))
        lu[0] = np.arange(12, dtype="i4").reshape(3, 4)


def test_namelist_and_rsl_parsing():
    values = evidence.parse_namelist(NAMELIST)
    assert evidence.namelist_int(values, "io_form_history") == 2 and evidence.namelist_str(values, "physics_suite") == "conus"
    names, frames, end, seconds = evidence.history_file_names(values)
    assert names == ["wrfout_d01_2019-11-27_00_00_00"] and frames == 1 and end == "2019-11-27_00:00:00" and seconds == 3600
    rsl = evidence.parse_rsl(_rsl())
    assert rsl["success"] and rsl["completed_steps"] == 50 and rsl["final_time"] == "2019-11-27_00:00:00"
    assert rsl["history_write_s"] == 8.29863 and rsl["history_writes"][0]["file"] == "wrfout_d01_2019-11-27_00_00_00"
    assert not evidence.parse_rsl(_rsl(success=False))["success"]


def test_history_summary_is_format_independent_and_value_sensitive():
    values = np.arange(12, dtype="f4").reshape(3, 4)
    with tempfile.TemporaryDirectory() as tmp:
        a = Path(tmp, "a.nc"); b = Path(tmp, "b.nc"); c = Path(tmp, "c.nc")
        _write_history(a, "NETCDF3_CLASSIC", values=values)
        _write_history(b, "NETCDF3_64BIT_OFFSET", values=values)
        _write_history(c, "NETCDF3_CLASSIC", values=values + 1e-3)
        sa, sb, sc = (evidence.summarize_history(p) for p in (a, b, c))
    assert sa["data_model"] == "NETCDF3_CLASSIC" and sb["data_model"] == "NETCDF3_64BIT_OFFSET"
    assert sa["variables"] == sb["variables"], "same values through two data models must summarize identically"
    assert sa["dimensions"] == sb["dimensions"] and sa["variables"]["T2"]["count"] == 12
    assert sa["variables"]["T2"]["sha256"] != sc["variables"]["T2"]["sha256"] and sa["variables"]["LU_INDEX"] == sc["variables"]["LU_INDEX"]
    assert "sum" not in sa["variables"]["Times"] and sa["variables"]["Times"]["dtype"] == "|S1"


def _darshan(path, module, size, shared, extra_modules=()):
    rec = {"bytes_written": size, "bytes_read": 0, "writes": 100, "reads": 0,
           "coll_writes": 100 if module == "MPI-IO" else 0, "coll_reads": 0, "ranks": [] if shared else [0], "shared": shared}
    files = {str(path): {module: rec}}
    if module == "MPI-IO":
        files[str(path)]["POSIX"] = dict(rec, coll_writes=0)
    for m in extra_modules:
        files[str(path)][m] = dict(rec, coll_writes=0)
    return {"files": files, "modules": sorted({"POSIX", "STDIO", module}), "partial": [], "nprocs": 128}


def test_correctness_and_io_validation_for_both_forms():
    values = evidence.parse_namelist(NAMELIST)
    with tempfile.TemporaryDirectory() as tmp:
        for io_form, model, module, shared in ((2, "NETCDF3_64BIT_OFFSET", "POSIX", False), (11, "NETCDF3_64BIT_OFFSET", "MPI-IO", True)):
            scratch = Path(tmp, f"f{io_form}"); scratch.mkdir()
            name = "wrfout_d01_2019-11-27_00_00_00"
            _write_history(scratch / name, model)
            correctness = evidence.build_correctness(evidence.parse_rsl(_rsl()), values, scratch)
            assert correctness["pass"], correctness["problems"]
            models = {n: h["data_model"] for n, h in correctness["result"]["history"].items()}
            size = (scratch / name).stat().st_size
            io = evidence.build_io_validation([name], io_form, scratch, _darshan(scratch / name, module, size, shared), 128, models)
            assert io["pass"], io["problems"]
            assert io["expected"]["write_module"] == module and io["observed"]["history_files"][0]["size"] == size
            short = _darshan(scratch / name, module, size - 1, shared)
            assert not evidence.build_io_validation([name], io_form, scratch, short, 128, models)["pass"]
            wrong_form = 11 if io_form == 2 else 2
            assert not evidence.build_io_validation([name], wrong_form, scratch, _darshan(scratch / name, module, size, shared), 128, models)["pass"]
            assert not evidence.build_io_validation([name], io_form, scratch, _darshan(scratch / name, module, size, shared), 128, {name: "NETCDF4"})["pass"]
            control = evidence.build_io_validation([name], io_form, scratch, None, 128, models)
            assert control["pass"] and control["check"] == "control_run_without_darshan"
        scratch = Path(tmp, "f2")
        assert not evidence.build_correctness(evidence.parse_rsl(_rsl(success=False)), values, scratch)["pass"]
        assert not evidence.build_correctness(evidence.parse_rsl(_rsl(steps=49)), values, scratch)["pass"]
        stdio = _darshan(scratch / name, "POSIX", size, False, extra_modules=("STDIO",))
        assert not evidence.build_io_validation([name], 2, scratch, stdio, 128, models)["pass"]


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_"):
            function()
    print("wrf evidence tests pass")
