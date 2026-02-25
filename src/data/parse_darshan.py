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
    """Parse using the PyDarshan library.

    Uses read_all=False to avoid segfault on APMPI/HEATMAP generic records
    (PyDarshan does not support these modules' record format), then reads
    only the modules we need (POSIX, MPI-IO, STDIO).
    """
    report = darshan.DarshanReport(path, read_all=False)

    # Read only the modules we need for feature extraction
    _FEATURE_MODULES = ['POSIX', 'MPI-IO', 'STDIO']
    for mod in _FEATURE_MODULES:
        if mod in report.modules:
            try:
                report.mod_read_all_records(mod)
            except Exception:
                logger.debug("Could not read module %s from %s", mod, path)

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

    # --- Extract counters from read modules ---
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


def _top4_merge(agg, new):
    """Merge two sets of (value, count) entries using Darshan's TOP-4 algorithm.

    Steps (from darshan-posix-logutils.c):
      1. Collapse duplicates: if a value exists in both sets, add the counts.
      2. Insert remaining new entries.
      3. Sort by count descending; tie-break by value descending.
      4. Truncate to keep only the top 4.
    """
    merged = {}
    for v, c in agg:
        merged[v] = merged.get(v, 0) + c
    for v, c in new:
        merged[v] = merged.get(v, 0) + c

    entries = [(v, c) for v, c in merged.items() if c > 0]
    entries.sort(key=lambda x: (x[1], x[0]), reverse=True)
    return entries[:4]


def _extract_pydarshan_module(report, module_name, counters, prefix=None):
    """Extract counters from a PyDarshan module, aggregated across files.

    Replicates darshan-parser --total aggregation semantics exactly using
    the 7 rules derived from Darshan 3.5.0 C source code analysis.
    See docs/darshan_counter_aggregation.md for full documentation.

    Rules:
      SUM:          operation counts, byte totals, histograms, cumulative times
      MAX:          MAX_BYTE_*, *_END_TIMESTAMP, F_MAX_*_TIME
      MIN_NONZERO:  *_START_TIMESTAMP (ignores 0 = "not set")
      LAST_VALUE:   MODE, MEM_ALIGNMENT, FILE_ALIGNMENT, RENAMED_FROM
      TOP-4 MERGE:  ACCESS1-4, STRIDE1-4 (sorted merge keeping 4 most frequent)
      CONDITIONAL:  FASTEST/SLOWEST rank (-1/0.0 if not shared), MAX_*_TIME_SIZE
      ZEROED:       F_VARIANCE_RANK_TIME, F_VARIANCE_RANK_BYTES (always 0)
    """
    try:
        rec = report.records[module_name]
        dfs = rec.to_df()
    except Exception:
        logger.debug("Cannot read module %s", module_name, exc_info=True)
        return

    pfx = prefix or module_name

    # Determine shared_file_flag: True only if ALL records reference the same file
    shared_file_flag = False
    if 'counters' in dfs:
        df_int = dfs['counters']
        if 'id' in df_int.columns and len(df_int) > 0:
            shared_file_flag = (df_int['id'].nunique() == 1)

    # --- Integer counter rules (keyed by suffix after prefix stripping) ---
    _MAX_INT = {'MAX_BYTE_READ', 'MAX_BYTE_WRITTEN'}
    _LAST_VALUE_INT = {'MODE', 'MEM_ALIGNMENT', 'FILE_ALIGNMENT', 'RENAMED_FROM'}
    _SENTINEL_INT = {
        'FASTEST_RANK', 'FASTEST_RANK_BYTES',
        'SLOWEST_RANK', 'SLOWEST_RANK_BYTES',
    }
    _CONDITIONAL_INT = {'MAX_READ_TIME_SIZE', 'MAX_WRITE_TIME_SIZE'}

    # --- Float counter rules ---
    _MAX_FLOAT = {'F_MAX_READ_TIME', 'F_MAX_WRITE_TIME'}
    _ZEROED_FLOAT = {'F_VARIANCE_RANK_TIME', 'F_VARIANCE_RANK_BYTES'}
    _SENTINEL_FLOAT = {'F_FASTEST_RANK_TIME', 'F_SLOWEST_RANK_TIME'}

    # --- TOP-4 MERGE groups: (group_prefix, value_suffix, count_suffix) ---
    _TOP4_GROUPS = []
    if module_name in ('POSIX', 'MPI-IO'):
        _TOP4_GROUPS.append(('ACCESS', 'ACCESS', 'COUNT'))
    if module_name == 'POSIX':
        _TOP4_GROUPS.append(('STRIDE', 'STRIDE', 'COUNT'))

    # Build set of TOP-4 column names to handle separately
    top4_cols = set()
    for grp, val_sfx, cnt_sfx in _TOP4_GROUPS:
        for i in range(1, 5):
            top4_cols.add(f'{pfx}_{grp}{i}_{val_sfx}')
            top4_cols.add(f'{pfx}_{grp}{i}_{cnt_sfx}')

    # --- Integer counters ---
    if 'counters' in dfs:
        df_int = dfs['counters']

        for col in df_int.columns:
            if col in ('id', 'rank') or col in top4_cols:
                continue
            key = col if col.startswith(pfx) else f"{pfx}_{col}"
            cname = col.replace(f'{pfx}_', '') if col.startswith(pfx) else col

            if cname in _MAX_INT:
                counters[key] = float(df_int[col].max())
            elif cname in _LAST_VALUE_INT:
                counters[key] = float(df_int[col].iloc[-1])
            elif cname in _SENTINEL_INT:
                if shared_file_flag:
                    counters[key] = float(df_int[col].iloc[-1])
                else:
                    counters[key] = -1.0
            elif cname in _CONDITIONAL_INT:
                # Deferred: set after float pass using F_MAX_*_TIME winner index
                counters[key] = 0.0
            else:
                # Default: SUM with -1 sentinel propagation.
                # Darshan C code uses overflow-clamp: if(sum < prev) sum = -1.
                # When any record has -1 (e.g., MMAPS = "not tracked"),
                # the sum decreases below prev, triggering the clamp to -1.
                vals = df_int[col]
                if (vals == -1).any():
                    counters[key] = -1.0
                else:
                    counters[key] = float(vals.sum())

        # TOP-4 MERGE for ACCESS and STRIDE counters
        for grp, val_sfx, cnt_sfx in _TOP4_GROUPS:
            agg_entries = []
            for row_idx in range(len(df_int)):
                row_entries = []
                for i in range(1, 5):
                    vc = f'{pfx}_{grp}{i}_{val_sfx}'
                    cc = f'{pfx}_{grp}{i}_{cnt_sfx}'
                    if vc in df_int.columns and cc in df_int.columns:
                        v = int(df_int[vc].iloc[row_idx])
                        c = int(df_int[cc].iloc[row_idx])
                        if c > 0:
                            row_entries.append((v, c))
                agg_entries = _top4_merge(agg_entries, row_entries)

            for i in range(1, 5):
                vc = f'{pfx}_{grp}{i}_{val_sfx}'
                cc = f'{pfx}_{grp}{i}_{cnt_sfx}'
                if i - 1 < len(agg_entries):
                    counters[vc] = float(agg_entries[i - 1][0])
                    counters[cc] = float(agg_entries[i - 1][1])
                else:
                    counters[vc] = 0.0
                    counters[cc] = 0.0

    # --- Float counters ---
    max_time_winner = {}  # 'READ' -> row_idx, 'WRITE' -> row_idx
    if 'fcounters' in dfs:
        df_float = dfs['fcounters']

        # Pre-compute which record wins F_MAX_READ_TIME and F_MAX_WRITE_TIME
        for direction in ('READ', 'WRITE'):
            col_name = f'{pfx}_F_MAX_{direction}_TIME'
            if col_name in df_float.columns and len(df_float) > 0:
                max_time_winner[direction] = int(df_float[col_name].idxmax())

        for col in df_float.columns:
            if col in ('id', 'rank'):
                continue
            key = col if col.startswith(pfx) else f"{pfx}_{col}"
            cname = col.replace(f'{pfx}_', '') if col.startswith(pfx) else col

            if cname in _ZEROED_FLOAT:
                counters[key] = 0.0
            elif cname in _MAX_FLOAT:
                counters[key] = float(df_float[col].max())
            elif cname in _SENTINEL_FLOAT:
                if shared_file_flag:
                    if 'FASTEST' in cname:
                        counters[key] = float(df_float[col].min())
                    else:
                        counters[key] = float(df_float[col].max())
                else:
                    counters[key] = 0.0
            elif 'START_TIMESTAMP' in cname:
                vals = df_float[col][df_float[col] > 0]
                counters[key] = float(vals.min()) if len(vals) > 0 else 0.0
            elif 'END_TIMESTAMP' in cname:
                counters[key] = float(df_float[col].max())
            else:
                # Default: SUM (cumulative times)
                counters[key] = float(df_float[col].sum())

    # Set CONDITIONAL integer counters using F_MAX_*_TIME winner indices
    if 'counters' in dfs:
        df_int = dfs['counters']
        for direction, idx in max_time_winner.items():
            size_key = f'{pfx}_MAX_{direction}_TIME_SIZE'
            if size_key in df_int.columns:
                counters[size_key] = float(df_int[size_key].iloc[idx])


# ---------------------------------------------------------------------------
# CLI backend (darshan-parser --total)
# ---------------------------------------------------------------------------

def _parse_with_cli(path):
    """Parse using darshan-parser CLI tool."""
    # Find darshan-parser: prefer sc2026 conda env build (has bzip2), then PATH
    import shutil
    parser_paths = [
        '/projects/bdau/envs/sc2026/bin/darshan-parser',
        shutil.which('darshan-parser'),
    ]
    parser_bin = next((p for p in parser_paths if p and Path(p).exists()), None)
    if parser_bin is None:
        logger.error("darshan-parser not found")
        return None

    # Run darshan-parser --total --perf
    try:
        result = subprocess.run(
            [parser_bin, '--total', '--perf', str(path)],
            capture_output=True, text=True, timeout=60,
        )
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
