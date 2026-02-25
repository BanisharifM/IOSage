"""
Darshan Log Parser
==================
Parses .darshan files into structured dictionaries of raw counters + job metadata.

Supports two backends:
  1. PyDarshan (preferred, faster): ``import darshan``
  2. darshan-parser CLI (fallback): ``darshan-parser --total <file>``

The parser extracts:
  - Job metadata (jobid, uid, nprocs, runtime, timestamps, modules)
  - POSIX module counters (always present)
  - MPI-IO module counters (present for MPI-parallel jobs)
  - STDIO module counters (present for C stdio usage)
  - Performance summary (aggregate bandwidth)
  - Mount point information (Eagle/Grand filesystem detection)

Usage::

    from src.data.parse_darshan import parse_darshan_log

    result = parse_darshan_log("/path/to/file.darshan")
    # result = {
    #     'job': { 'jobid': ..., 'nprocs': ..., 'runtime': ..., ... },
    #     'counters': { 'POSIX_READS': ..., 'POSIX_WRITES': ..., ... },
    #     'modules': ['POSIX', 'STDIO', ...],
    # }
"""

import logging
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------
_HAS_PYDARSHAN = False
try:
    import darshan
    _HAS_PYDARSHAN = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_darshan_log(darshan_path, backend=None):
    """Parse a single .darshan file and return structured data.

    Parameters
    ----------
    darshan_path : str or Path
        Path to a ``.darshan`` file.
    backend : str, optional
        ``'pydarshan'`` or ``'cli'``.  Auto-detected if not specified.

    Returns
    -------
    dict or None
        Dictionary with keys ``'job'``, ``'counters'``, ``'modules'``.
        Returns ``None`` if the file cannot be parsed.
    """
    darshan_path = str(darshan_path)

    if backend is None:
        backend = 'pydarshan' if _HAS_PYDARSHAN else 'cli'

    try:
        if backend == 'pydarshan':
            return _parse_with_pydarshan(darshan_path)
        return _parse_with_cli(darshan_path)
    except Exception:
        logger.debug("Failed to parse %s", darshan_path, exc_info=True)
        return None


def list_available_modules(darshan_path):
    """Return list of module names present in a Darshan log."""
    result = parse_darshan_log(darshan_path)
    if result is None:
        return []
    return result['modules']


# ---------------------------------------------------------------------------
# PyDarshan backend
# ---------------------------------------------------------------------------

def _parse_with_pydarshan(path):
    """Parse using the PyDarshan library."""
    report = darshan.DarshanReport(path, read_all=True)

    # --- Job metadata ---
    job_meta = report.metadata.get('job', {})
    job = {
        'jobid': job_meta.get('jobid', 0),
        'uid': job_meta.get('uid', 0),
        'nprocs': job_meta.get('nprocs', 1),
        'start_time': job_meta.get('start_time', 0),
        'end_time': job_meta.get('end_time', 0),
        'runtime': job_meta.get('run_time', 0.0),
        'log_version': report.metadata.get('lib_ver', ''),
    }

    modules = list(report.modules.keys())
    job['modules'] = modules

    # Detect filesystem from mount points
    job['uses_lustre'] = False
    job['lustre_mount'] = ''
    for mount_info in report.metadata.get('mounts', []):
        mount_point = mount_info[0] if isinstance(mount_info, (list, tuple)) else str(mount_info)
        fs_type = mount_info[1] if isinstance(mount_info, (list, tuple)) and len(mount_info) > 1 else ''
        if 'lustre' in str(fs_type).lower() or '/lus/' in str(mount_point):
            job['uses_lustre'] = True
            job['lustre_mount'] = str(mount_point)
            break

    # --- Extract counters using --total aggregation ---
    counters = {}

    # POSIX
    if 'POSIX' in report.records:
        _extract_pydarshan_module(report, 'POSIX', counters)

    # MPI-IO
    if 'MPI-IO' in report.records:
        _extract_pydarshan_module(report, 'MPI-IO', counters, prefix='MPIIO')

    # STDIO
    if 'STDIO' in report.records:
        _extract_pydarshan_module(report, 'STDIO', counters)

    # Count unique files
    counters['num_files'] = len(report.name_records) if hasattr(report, 'name_records') else 0

    return {'job': job, 'counters': counters, 'modules': modules}


def _extract_pydarshan_module(report, module_name, counters, prefix=None):
    """Extract counters from a PyDarshan module, aggregated across files."""
    try:
        rec = report.records[module_name]
        dfs = rec.to_df()
    except Exception:
        logger.debug("Cannot read module %s", module_name, exc_info=True)
        return

    pfx = prefix or module_name

    # Integer counters
    if 'counters' in dfs:
        df_int = dfs['counters']
        for col in df_int.columns:
            if col in ('id', 'rank'):
                continue
            key = col if col.startswith(pfx) else f"{pfx}_{col}"
            # Aggregate: sum across all file records
            # Special handling for rank-based counters (use max for variance)
            if 'VARIANCE' in col or 'FASTEST' in col or 'SLOWEST' in col:
                counters[key] = float(df_int[col].max())
            else:
                counters[key] = float(df_int[col].sum())

    # Float counters
    if 'fcounters' in dfs:
        df_float = dfs['fcounters']
        for col in df_float.columns:
            if col in ('id', 'rank'):
                continue
            key = col if col.startswith(pfx) else f"{pfx}_{col}"
            # Timing: sum for cumulative, max for timestamps/variance
            if 'TIMESTAMP' in col or 'VARIANCE' in col:
                counters[key] = float(df_float[col].max())
            elif 'FASTEST' in col:
                counters[key] = float(df_float[col].min())
            elif 'SLOWEST' in col:
                counters[key] = float(df_float[col].max())
            else:
                counters[key] = float(df_float[col].sum())


# ---------------------------------------------------------------------------
# CLI backend (darshan-parser --total)
# ---------------------------------------------------------------------------

def _parse_with_cli(path):
    """Parse using darshan-parser CLI tool."""
    # Run darshan-parser --total --perf
    try:
        result = subprocess.run(
            ['darshan-parser', '--total', '--perf', str(path)],
            capture_output=True, text=True, timeout=60,
        )
    except FileNotFoundError:
        logger.error("darshan-parser not found in PATH")
        return None
    except subprocess.TimeoutExpired:
        logger.warning("darshan-parser timed out on %s", path)
        return None

    if result.returncode != 0:
        logger.debug("darshan-parser failed on %s: %s", path, result.stderr[:200])
        return None

    output = result.stdout
    job = _parse_cli_header(output)
    counters = _parse_cli_counters(output)
    modules = _detect_cli_modules(output)

    job['modules'] = modules

    # Detect filesystem
    job['uses_lustre'] = '/lus/' in output
    for line in output.splitlines():
        if 'mount entry' in line and 'lustre' in line.lower():
            parts = line.split('\t')
            if len(parts) >= 2:
                job['lustre_mount'] = parts[1].strip()
                break

    # Count files from --total output is not directly available,
    # but we can get it from the record table note or set to 0
    counters.setdefault('num_files', 0)

    # Try to get file count from a separate --base run
    try:
        base_result = subprocess.run(
            ['darshan-parser', '--base', str(path)],
            capture_output=True, text=True, timeout=60,
        )
        if base_result.returncode == 0:
            # Count unique file IDs
            file_ids = set()
            for line in base_result.stdout.splitlines():
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.split('\t')
                if len(parts) >= 3:
                    file_ids.add(parts[2].strip())
            counters['num_files'] = len(file_ids)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return {'job': job, 'counters': counters, 'modules': modules}


def _parse_cli_header(output):
    """Extract job metadata from darshan-parser header comments."""
    job = {
        'jobid': 0,
        'uid': 0,
        'nprocs': 1,
        'start_time': 0,
        'end_time': 0,
        'runtime': 0.0,
        'log_version': '',
        'uses_lustre': False,
        'lustre_mount': '',
    }

    for line in output.splitlines():
        line = line.strip()
        if not line.startswith('#'):
            continue

        if '# jobid:' in line:
            job['jobid'] = _safe_int(line.split(':')[-1])
        elif '# uid:' in line:
            job['uid'] = _safe_int(line.split(':')[-1])
        elif '# nprocs:' in line:
            job['nprocs'] = _safe_int(line.split(':')[-1])
        elif '# start_time:' in line and 'asci' not in line:
            job['start_time'] = _safe_int(line.split(':')[-1])
        elif '# end_time:' in line and 'asci' not in line:
            job['end_time'] = _safe_int(line.split(':')[-1])
        elif '# run time:' in line:
            job['runtime'] = _safe_float(line.split(':')[-1])
        elif '# darshan log version:' in line:
            job['log_version'] = line.split(':')[-1].strip()
        elif '# start_time_asci:' in line:
            job['start_time_str'] = line.split(':', 1)[-1].strip()

    return job


def _parse_cli_counters(output):
    """Extract total_* counter lines from darshan-parser --total output."""
    counters = {}
    perf_section = False

    for line in output.splitlines():
        line = line.strip()

        # Parse total_MODULE_COUNTER: value lines
        if line.startswith('total_'):
            match = re.match(r'^total_(\w+):\s+(-?\d+\.?\d*(?:e[+-]?\d+)?)', line)
            if match:
                key = match.group(1)
                val = _safe_float(match.group(2))
                counters[key] = val

        # Parse performance section
        if 'agg_perf_by_slowest:' in line:
            match = re.search(r'agg_perf_by_slowest:\s+(-?\d+\.?\d*)', line)
            if match:
                counters['agg_perf_mib_s'] = _safe_float(match.group(1))

    return counters


def _detect_cli_modules(output):
    """Detect which modules are present from darshan-parser output."""
    modules = []
    module_patterns = {
        'POSIX': r'POSIX module.*ver=',
        'MPI-IO': r'MPI-IO module.*ver=',
        'STDIO': r'STDIO module.*ver=',
        'H5F': r'H5F module.*ver=',
        'H5D': r'H5D module.*ver=',
        'LUSTRE': r'LUSTRE module.*ver=',
        'PNETCDF': r'PNETCDF module.*ver=',
        'APMPI': r'APMPI module.*ver=',
        'HEATMAP': r'HEATMAP module.*ver=',
    }

    for mod_name, pattern in module_patterns.items():
        if re.search(pattern, output):
            modules.append(mod_name)

    # Also check for module data sections
    for mod_name in ['POSIX', 'STDIO']:
        if f'{mod_name} module data' in output and mod_name not in modules:
            modules.append(mod_name)
    if 'MPI-IO module data' in output and 'MPI-IO' not in modules:
        modules.append('MPI-IO')

    return modules


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _safe_int(s):
    """Safely parse an integer from a string."""
    try:
        return int(str(s).strip())
    except (ValueError, TypeError):
        return 0


def _safe_float(s):
    """Safely parse a float from a string."""
    try:
        return float(str(s).strip())
    except (ValueError, TypeError):
        return 0.0
