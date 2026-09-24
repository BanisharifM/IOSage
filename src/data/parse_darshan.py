"""
Darshan Log Parser
==================
Parses .darshan files into structured dictionaries of raw counters + job metadata
with PyDarshan (``import darshan``). PyDarshan must load a libdarshan-util that
can read the log's compression (the Polaris logs are bzip2): see
``scripts/build_env_iosage.slurm``.

The parser extracts:
  - Job metadata (jobid, uid, nprocs, runtime, timestamps, modules)
  - POSIX, MPI-IO and STDIO module counters, aggregated per job
  - Per-rank statistics from the file records (``_rank_statistics``)

Usage::

    from src.data.parse_darshan import parse_darshan_log

    result = parse_darshan_log("/path/to/file.darshan")
    # result = {
    #     'job': { 'jobid': ..., 'nprocs': ..., 'runtime': ..., ... },
    #     'counters': { 'POSIX_READS': ..., 'POSIX_WRITES': ..., ... },
    #     'modules': ['POSIX', 'STDIO', ...],
    #     'shared_file_flags': {'POSIX': False, ...},
    # }
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path

import darshan
import darshan.backend.cffi_backend as darshan_backend
import numpy as np

logger = logging.getLogger(__name__)

# Modules whose records become features. A log that declares one of them but
# whose records cannot be read is a failed sample, not a job with zero I/O.
FEATURE_MODULES = (('POSIX', None), ('MPI-IO', 'MPIIO'), ('STDIO', None))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_darshan_log(
    darshan_path: str | Path, strict: bool = False,
) -> dict[str, object] | None:
    """Parse a single .darshan file and return structured data.

    Parameters
    ----------
    darshan_path : str or Path
        Path to a ``.darshan`` file.
    strict : bool
        If True, a parse failure raises; otherwise it is logged and None is
        returned (callers that must record the cause use ``strict=True``).

    Returns
    -------
    dict or None
        Dictionary with keys ``'job'``, ``'counters'``, ``'modules'``,
        ``'shared_file_flags'`` and ``'partial_modules'``.
        None if the file cannot be parsed.
    """
    darshan_path = str(darshan_path)
    try:
        return _parse_with_pydarshan(darshan_path)
    except Exception:
        if strict:
            raise
        logger.warning("Failed to parse %s", darshan_path, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# PyDarshan backend
# ---------------------------------------------------------------------------

def _job_metadata(report, nprocs=None):
    """Job dictionary from a report's metadata; ``nprocs`` overrides the log's."""
    job_meta = report.metadata['job']
    required = {
        'jobid', 'uid', 'nprocs', 'start_time_sec', 'end_time_sec',
        'run_time', 'log_ver',
    }
    missing = required - set(job_meta)
    if missing:
        raise ValueError(f"Darshan job metadata lacks {sorted(missing)}")
    process_count = job_meta['nprocs'] if nprocs is None else nprocs
    if int(process_count) < 1:
        raise ValueError(f"invalid Darshan process count: {process_count}")
    return {
        'jobid': job_meta['jobid'],
        'uid': job_meta['uid'],
        'nprocs': int(process_count),
        'start_time': job_meta['start_time_sec'],
        'end_time': job_meta['end_time_sec'],
        'runtime': job_meta['run_time'],
        'log_version': job_meta['log_ver'],
    }


def _read_module_frames(report, module_name, path):
    """``to_df()`` of one declared module; a read failure is an error."""
    try:
        report.mod_read_all_records(module_name)
        return report.records[module_name].to_df()
    except Exception as exc:
        raise ValueError(f"cannot read module {module_name} of {path}: {exc}") from exc


def _partial_feature_modules(report):
    """Feature modules whose record sets are incomplete."""
    return {
        mod for mod, _ in FEATURE_MODULES
        if mod in report.modules and report.modules[mod]['partial_flag']
    }


def _open_report(path):
    """Open a report, tolerating only an invalid unused mount type string.

    Darshan 3.4.x could write stray bytes at the end of the mount table. The
    feature pipeline does not use mount metadata. All other metadata and
    module reads still use libdarshan-util and keep their normal errors.
    """
    try:
        return darshan.DarshanReport(path, read_all=False)
    except UnicodeDecodeError as original_error:
        log = darshan_backend.log_open(path)
        if not bool(log['handle']):
            raise RuntimeError(f"failed to open {path}") from original_error
        try:
            darshan_backend.log_get_mounts(log)
        except UnicodeDecodeError:
            pass
        else:
            darshan_backend.log_close(log)
            raise original_error

        report = darshan.DarshanReport()
        report.filename = path
        report.log = log
        report.metadata['job'] = darshan_backend.log_get_job(log)
        report.metadata['exe'] = darshan_backend.log_get_exe(log)
        job = report.metadata['job']
        report.start_time = datetime.datetime.fromtimestamp(job['start_time_sec'])
        report.end_time = datetime.datetime.fromtimestamp(job['end_time_sec'])
        report.data['mounts'] = []
        report.mounts = []
        report.data['modules'] = darshan_backend.log_get_modules(log)
        report._modules = report.data['modules']
        logger.warning("Ignoring invalid unused mount metadata in %s", path)
        return report


def _parse_with_pydarshan(path):
    """Parse using the PyDarshan library.

    Opens with read_all=False (PyDarshan cannot decode APMPI/HEATMAP records)
    and reads only the feature modules.
    """
    report = _open_report(path)
    job = _job_metadata(report)
    modules = list(report.modules.keys())
    job['modules'] = modules
    partial_modules = _partial_feature_modules(report)
    if partial_modules:
        logger.warning("Incomplete module records in %s: %s", path,
                       ", ".join(sorted(partial_modules)))

    counters = {}
    shared_file_flags = {}
    module_dfs = {}
    for mod, prefix in FEATURE_MODULES:
        if mod not in report.modules:
            continue
        dfs = _read_module_frames(report, mod, path)
        module_dfs[mod] = dfs
        shared_file_flags[mod] = _extract_pydarshan_module(
            dfs, mod, counters, job['nprocs'], prefix=prefix)

    name_records = getattr(report, 'name_records', {}) or {}
    counters.update(_rank_statistics(module_dfs, job['nprocs'], name_records))
    file_facts = _file_facts(module_dfs, name_records)
    counters['num_files'] = file_facts['num_files']
    counters['num_data_files'] = file_facts['data_files']

    return {
        'job': job,
        'counters': counters,
        'modules': modules,
        'shared_file_flags': shared_file_flags,
        'partial_modules': sorted(partial_modules),
    }


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


def _shared_file_flag(df_int, nprocs):
    """True when any module record belongs to a file used by several ranks."""
    if df_int is None or 'id' not in df_int.columns or df_int.empty:
        return False
    if (df_int['rank'] == -1).any():
        return True
    if nprocs <= 1:
        return False
    own = df_int[df_int['rank'] >= 0]
    return bool((own.groupby('id')['rank'].nunique() > 1).any())


def _single_shared_file_flag(df_int, nprocs):
    """True when all records describe one file used by every rank."""
    if df_int is None or 'id' not in df_int.columns or df_int.empty:
        return False
    if df_int['id'].nunique() != 1:
        return False
    if (df_int['rank'] == -1).any():
        return True
    return nprocs > 1 and df_int['rank'].nunique() == nprocs


def _shared_reduction(df_int, df_float, pfx):
    """Darshan's shared-record reduction over per-rank records of one file.

    Mirrors what the runtime computes at MPI_Finalize for a file opened by all
    ranks (darshan-posix.c, ``posix_shared_record_variance`` and the fastest
    and slowest rank fields): the rank with the least and the most cumulative
    I/O time, the bytes those two ranks moved, and the population variance of
    time and bytes over the ranks.
    """
    time_cols = [f'{pfx}_F_READ_TIME', f'{pfx}_F_WRITE_TIME', f'{pfx}_F_META_TIME']
    byte_cols = [f'{pfx}_BYTES_READ', f'{pfx}_BYTES_WRITTEN']
    times = df_float[[c for c in time_cols if c in df_float.columns]].sum(axis=1).to_numpy()
    byts = df_int[[c for c in byte_cols if c in df_int.columns]].sum(axis=1).to_numpy(dtype=float)
    ranks = df_int['rank'].to_numpy()
    fastest = int(np.argmin(times))
    slowest = int(np.argmax(times))
    return {
        f'{pfx}_FASTEST_RANK': float(ranks[fastest]),
        f'{pfx}_FASTEST_RANK_BYTES': float(byts[fastest]),
        f'{pfx}_SLOWEST_RANK': float(ranks[slowest]),
        f'{pfx}_SLOWEST_RANK_BYTES': float(byts[slowest]),
        f'{pfx}_F_FASTEST_RANK_TIME': float(times[fastest]),
        f'{pfx}_F_SLOWEST_RANK_TIME': float(times[slowest]),
        f'{pfx}_F_VARIANCE_RANK_TIME': float(times.var()),
        f'{pfx}_F_VARIANCE_RANK_BYTES': float(byts.var()),
    }


def _extract_pydarshan_module(dfs, module_name, counters, nprocs, prefix=None):
    """Extract counters from a PyDarshan module, aggregated across files.

    Replicates darshan-parser --total aggregation semantics using the rules
    derived from the Darshan 3.5.0 C source. See
    docs/3_guides/darshan_counter_aggregation.md for full documentation.

    Rules:
      SUM:          operation counts, byte totals, histograms, cumulative times
      MAX:          MAX_BYTE_*, *_END_TIMESTAMP, F_MAX_*_TIME
      MIN_NONZERO:  *_START_TIMESTAMP (ignores 0 = "not set")
      LAST_VALUE:   MODE, MEM_ALIGNMENT, FILE_ALIGNMENT, RENAMED_FROM
      TOP-4 MERGE:  ACCESS1-4, STRIDE1-4 (sorted merge keeping 4 most frequent)
      SHARED:       FASTEST/SLOWEST rank and bytes, F_FASTEST/SLOWEST_RANK_TIME,
                    F_VARIANCE_RANK_*: taken from the reduced record when the
                    whole log is one file used by all ranks (``_shared_file_flag``),
                    computed by ``_shared_reduction`` when that file appears as
                    per-rank records (merged per-process logs), else -1 / 0.0
      CONDITIONAL:  MAX_*_TIME_SIZE from the record that holds F_MAX_*_TIME

    Darshan fills the SHARED counters only for shared records (rank -1), and
    ``darshan-parser --total`` zeroes the variances because they cannot be
    summed across records. Rank-level imbalance for every other layout comes
    from ``_rank_statistics``.

    Parameters
    ----------
    dfs : dict
        ``to_df()`` output of the module (``'counters'`` and ``'fcounters'``).
    module_name : str
    counters : dict
        Filled in place.
    nprocs : int
        Processes of the job (for the shared-file test).
    prefix : str, optional
        Counter prefix when it differs from the module name (``MPIIO``).

    Returns
    -------
    bool
        The shared-file flag of this module.
    """
    pfx = prefix or module_name
    df_int = dfs.get('counters')
    df_float = dfs.get('fcounters')

    shared_file_flag = _shared_file_flag(df_int, nprocs)
    single_shared_file = _single_shared_file_flag(df_int, nprocs)
    shared_values = {}
    if single_shared_file:
        reduced = df_int['rank'] == -1
        if reduced.any():
            pos = int(np.flatnonzero(reduced.to_numpy())[-1])
            for frame in (df_int, df_float):
                if frame is not None:
                    row = frame.iloc[pos]
                    for col in frame.columns:
                        if 'RANK' in col:
                            shared_values[col] = float(row[col])
        else:
            shared_values = _shared_reduction(df_int, df_float, pfx)

    # --- Integer counter rules (keyed by suffix after prefix stripping) ---
    _MAX_INT = {'MAX_BYTE_READ', 'MAX_BYTE_WRITTEN'}
    _LAST_VALUE_INT = {'MODE', 'MEM_ALIGNMENT', 'FILE_ALIGNMENT', 'RENAMED_FROM'}
    _SHARED_INT = {
        'FASTEST_RANK', 'FASTEST_RANK_BYTES',
        'SLOWEST_RANK', 'SLOWEST_RANK_BYTES',
    }
    _CONDITIONAL_INT = {'MAX_READ_TIME_SIZE', 'MAX_WRITE_TIME_SIZE'}

    # --- Float counter rules ---
    _MAX_FLOAT = {'F_MAX_READ_TIME', 'F_MAX_WRITE_TIME'}
    _SHARED_FLOAT = {'F_VARIANCE_RANK_TIME', 'F_VARIANCE_RANK_BYTES',
                     'F_FASTEST_RANK_TIME', 'F_SLOWEST_RANK_TIME'}

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
    if df_int is not None:
        for col in df_int.columns:
            if col in ('id', 'rank') or col in top4_cols:
                continue
            key = col if col.startswith(pfx) else f"{pfx}_{col}"
            cname = col.replace(f'{pfx}_', '') if col.startswith(pfx) else col

            if cname in _MAX_INT:
                counters[key] = float(df_int[col].max())
            elif cname in _LAST_VALUE_INT:
                counters[key] = float(df_int[col].iloc[-1])
            elif cname in _SHARED_INT:
                counters[key] = shared_values.get(key, -1.0)
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
    max_time_winner = {}  # 'READ' -> row position, 'WRITE' -> row position
    if df_float is not None:
        # Which record (by position, not index label) wins F_MAX_*_TIME
        for direction in ('READ', 'WRITE'):
            col_name = f'{pfx}_F_MAX_{direction}_TIME'
            if col_name in df_float.columns and len(df_float) > 0:
                max_time_winner[direction] = int(np.argmax(df_float[col_name].to_numpy()))

        for col in df_float.columns:
            if col in ('id', 'rank'):
                continue
            key = col if col.startswith(pfx) else f"{pfx}_{col}"
            cname = col.replace(f'{pfx}_', '') if col.startswith(pfx) else col

            if cname in _SHARED_FLOAT:
                counters[key] = shared_values.get(key, 0.0)
            elif cname in _MAX_FLOAT:
                counters[key] = float(df_float[col].max())
            elif 'START_TIMESTAMP' in cname:
                vals = df_float[col][df_float[col] > 0]
                counters[key] = float(vals.min()) if len(vals) > 0 else 0.0
            elif 'END_TIMESTAMP' in cname:
                counters[key] = float(df_float[col].max())
            else:
                # Default: SUM (cumulative times)
                counters[key] = float(df_float[col].sum())

    # Set CONDITIONAL integer counters using F_MAX_*_TIME winner positions
    if df_int is not None:
        for direction, pos in max_time_winner.items():
            size_key = f'{pfx}_MAX_{direction}_TIME_SIZE'
            if size_key in df_int.columns:
                counters[size_key] = float(df_int[size_key].iloc[pos])

    return shared_file_flag


# ---------------------------------------------------------------------------
# Per-rank statistics (computed from the file records before aggregation)
# ---------------------------------------------------------------------------

# Layers whose records are summed per rank. MPI-IO is left out because its
# bytes reach POSIX as well and would be counted twice.
_RANK_LAYERS = ('POSIX', 'STDIO')

# Darshan's names for the standard streams. They are not data files: their
# bytes are log messages, so they are left out of every rank statistic.
_STANDARD_STREAMS = {'<STDIN>', '<STDOUT>', '<STDERR>'}

RANK_STAT_KEYS = (
    'RANK_IO_COUNT', 'RANK_BYTES_MAX', 'RANK_BYTES_MIN', 'RANK_BYTES_VAR',
    'RANK_BYTES_GINI', 'RANK_TIME_MAX', 'RANK_TIME_MIN', 'RANK_TIME_VAR',
    'RANK_BYTES_TOTAL', 'RANK_TIME_TOTAL', 'RANK_SHARED_BYTES',
    'SHARED_BYTE_IMBALANCE', 'SHARED_TIME_IMBALANCE',
    'FILE_WRITE_IMBALANCE', 'FILE_READ_IMBALANCE',
    'SHARED_POSIX_READS', 'SHARED_POSIX_WRITES',
    'SHARED_POSIX_SMALL_READS', 'SHARED_POSIX_SMALL_WRITES',
)


def _gini(values):
    """Gini coefficient of a non-negative array (0 = equal, 1 = one holder)."""
    total = float(values.sum())
    if total <= 0 or len(values) < 2:
        return 0.0
    sorted_vals = sorted(float(v) for v in values)
    n = len(sorted_vals)
    weighted = sum((i + 1) * v for i, v in enumerate(sorted_vals))
    return (2.0 * weighted) / (n * total) - (n + 1.0) / n


def _stream_ids(name_records):
    return {i for i, name in name_records.items() if name in _STANDARD_STREAMS}


def _file_facts(module_dfs, name_records):
    """Count feature-module files and files that moved data."""
    streams = _stream_ids(name_records)
    feature_ids = set()
    data_ids = set()
    for module in _RANK_LAYERS + ('MPI-IO',):
        dfs = module_dfs.get(module)
        if not dfs or dfs.get('counters') is None:
            continue
        frame = dfs['counters']
        ids = {int(value) for value in frame['id'].unique()} - streams
        feature_ids.update(ids)
        if module not in _RANK_LAYERS:
            continue
        prefix = 'MPIIO' if module == 'MPI-IO' else module
        byte_columns = [
            name for name in (f'{prefix}_BYTES_READ', f'{prefix}_BYTES_WRITTEN')
            if name in frame.columns
        ]
        if byte_columns:
            totals = frame.groupby('id')[byte_columns].sum().sum(axis=1)
            data_ids.update(int(record_id) for record_id in totals[totals > 0].index
                            if int(record_id) not in streams)
    return {
        'num_files': len(feature_ids),
        'data_files': len(data_ids),
    }


def _per_file_imbalance(df_int, column):
    """Drishti's individual-file rule: (max - min) / max of ``column`` over the
    ranks that accessed the same file id, for records with rank != -1; returns
    the largest value over all such files (0.0 if none is used by 2+ ranks)."""
    if column not in df_int.columns:
        return 0.0
    own = df_int[df_int['rank'] != -1]
    if own.empty:
        return 0.0
    grouped = own.groupby('id')[column].agg(['max', 'min', 'count'])
    grouped = grouped[(grouped['count'] > 1) & (grouped['max'] > 0)]
    if grouped.empty:
        return 0.0
    return float(((grouped['max'] - grouped['min']) / grouped['max']).max())


def _shared_record_imbalance(df_int, df_float, pfx):
    """Drishti's shared-file straggler measures, largest over the reduced
    (rank -1) records: |SLOWEST_RANK_BYTES - FASTEST_RANK_BYTES| / record bytes
    (P18) and |F_SLOWEST_RANK_TIME - F_FASTEST_RANK_TIME| / record time (P19).

    Returns (byte_imbalance, time_imbalance); 0.0 when there is no reduced
    record. Rank -1 records are the only place Darshan keeps per-rank evidence
    for a file opened by all ranks, so this survives the per-file aggregation.
    """
    byte_imb = 0.0
    time_imb = 0.0
    shared = df_int['rank'] == -1
    if not shared.any():
        return byte_imb, time_imb
    fast_b, slow_b = f'{pfx}_FASTEST_RANK_BYTES', f'{pfx}_SLOWEST_RANK_BYTES'
    if fast_b in df_int.columns and slow_b in df_int.columns:
        rec = df_int[shared]
        total = rec[[f'{pfx}_BYTES_READ', f'{pfx}_BYTES_WRITTEN']].clip(lower=0).sum(axis=1)
        ok = total > 0
        if ok.any():
            byte_imb = float(((rec[slow_b] - rec[fast_b]).abs()[ok] / total[ok]).max())
    fast_t, slow_t = f'{pfx}_F_FASTEST_RANK_TIME', f'{pfx}_F_SLOWEST_RANK_TIME'
    if df_float is not None and fast_t in df_float.columns and slow_t in df_float.columns:
        rec = df_float[df_float['rank'] == -1]
        time_cols = [c for c in (f'{pfx}_F_READ_TIME', f'{pfx}_F_WRITE_TIME',
                                 f'{pfx}_F_META_TIME') if c in rec.columns]
        total = rec[time_cols].clip(lower=0).sum(axis=1)
        ok = total > 0
        if ok.any():
            time_imb = float(((rec[slow_t] - rec[fast_t]).abs()[ok] / total[ok]).max())
    return byte_imb, time_imb


def _rank_statistics(module_dfs, nprocs, name_records):
    """Distribution of I/O bytes and time over the ranks of a job.

    Darshan writes one record per (file, rank); files opened by all ranks are
    reduced to one record with rank -1. Summing the records per rank gives the
    per-process totals that ``darshan-parser --perf`` uses for its "unique
    files: slowest_rank" figures and that AIIO's "time of the slowest process"
    refers to. This is the only place where imbalance of a file-per-process
    job is visible, because Darshan's FASTEST/SLOWEST/VARIANCE counters exist
    for shared records alone.

    Shared (rank -1) records are spread evenly over ``nprocs`` for the rank
    totals; the imbalance inside them is kept separately as
    ``SHARED_BYTE_IMBALANCE`` and ``SHARED_TIME_IMBALANCE`` (Drishti's P18 and
    P19 measures, largest over the reduced records). Those two come from the
    MPI-IO records when the job has any, because collective buffering makes
    a few aggregator ranks do the POSIX writes for everyone and the POSIX
    view of a balanced collective write is then maximally uneven; otherwise
    from POSIX. Ranks without records count as zero, so a single writer in a
    32-rank job gets ``RANK_BYTES_GINI`` near 1 and ``RANK_IO_COUNT`` = 1.
    The standard streams are excluded.

    Parameters
    ----------
    module_dfs : dict
        ``{module: {'counters': df, 'fcounters': df}}`` as returned by
        ``to_df()``; only POSIX and STDIO are used.
    nprocs : int
        Number of processes of the job.
    name_records : dict
        Darshan record id to file name, used to recognize the standard streams.

    Returns
    -------
    dict
        Keys listed in ``RANK_STAT_KEYS``.
    """
    n = max(int(nprocs), 1)
    rank_bytes = np.zeros(n, dtype=np.float64)
    rank_time = np.zeros(n, dtype=np.float64)
    shared_bytes = 0.0
    file_write_imb = 0.0
    file_read_imb = 0.0
    shared_posix_reads = 0.0
    shared_posix_writes = 0.0
    shared_posix_small_reads = 0.0
    shared_posix_small_writes = 0.0
    streams = _stream_ids(name_records)

    # Straggler evidence from the layer the application used
    mpiio = module_dfs.get('MPI-IO')
    mpiio_has_shared = (
        mpiio
        and mpiio.get('counters') is not None
        and not mpiio['counters'].empty
        and (mpiio['counters']['rank'] == -1).any()
    )
    if mpiio_has_shared:
        shared_byte_imb, shared_time_imb = _shared_record_imbalance(
            mpiio['counters'], mpiio.get('fcounters'), 'MPIIO')
    else:
        shared_byte_imb, shared_time_imb = 0.0, 0.0
        posix = module_dfs.get('POSIX')
        if posix and posix.get('counters') is not None and not posix['counters'].empty:
            keep = ~posix['counters']['id'].isin(streams).to_numpy()
            fl = posix.get('fcounters')
            shared_byte_imb, shared_time_imb = _shared_record_imbalance(
                posix['counters'][keep], fl[keep] if fl is not None else None, 'POSIX')

    def _accumulate(df, columns, target):
        """Add the row sums of ``columns`` to ``target`` by rank; shared rows
        (rank -1) are spread evenly. Returns the shared part."""
        per_rec = df[columns].clip(lower=0).sum(axis=1)
        shared_mask = df['rank'] == -1
        shared_part = float(per_rec[shared_mask].sum())
        target += shared_part / n
        own = per_rec[~shared_mask].groupby(df.loc[~shared_mask, 'rank']).sum()
        if len(own) and int(own.index.max()) >= n:
            raise ValueError(
                f"record for rank {int(own.index.max())} in a job with nprocs={n}")
        for rank, value in own.items():
            target[int(rank)] += float(value)
        return shared_part

    for mod in _RANK_LAYERS:
        dfs = module_dfs.get(mod)
        if not dfs or dfs.get('counters') is None:
            continue
        keep = ~dfs['counters']['id'].isin(streams).to_numpy()
        df_int = dfs['counters'][keep]
        df_float = dfs.get('fcounters')
        if df_float is not None:
            df_float = df_float[keep]

        byte_cols = [c for c in (f'{mod}_BYTES_READ', f'{mod}_BYTES_WRITTEN')
                     if c in df_int.columns]
        if byte_cols and not df_int.empty:
            shared_bytes += _accumulate(df_int, byte_cols, rank_bytes)
        if mod == 'POSIX' and not df_int.empty:
            file_write_imb = _per_file_imbalance(df_int, 'POSIX_BYTES_WRITTEN')
            file_read_imb = _per_file_imbalance(df_int, 'POSIX_BYTES_READ')
            shared_ids = set(df_int.loc[df_int['rank'] == -1, 'id'])
            own = df_int[df_int['rank'] >= 0]
            shared_ids.update(
                own.groupby('id')['rank'].nunique().loc[lambda count: count > 1].index
            )
            shared_records = df_int[df_int['id'].isin(shared_ids)]
            shared_posix_reads = float(shared_records['POSIX_READS'].sum()) \
                if 'POSIX_READS' in shared_records else 0.0
            shared_posix_writes = float(shared_records['POSIX_WRITES'].sum()) \
                if 'POSIX_WRITES' in shared_records else 0.0
            small_read_columns = [name for name in [
                'POSIX_SIZE_READ_0_100', 'POSIX_SIZE_READ_100_1K',
                'POSIX_SIZE_READ_1K_10K', 'POSIX_SIZE_READ_10K_100K',
                'POSIX_SIZE_READ_100K_1M',
            ] if name in shared_records]
            small_write_columns = [name for name in [
                'POSIX_SIZE_WRITE_0_100', 'POSIX_SIZE_WRITE_100_1K',
                'POSIX_SIZE_WRITE_1K_10K', 'POSIX_SIZE_WRITE_10K_100K',
                'POSIX_SIZE_WRITE_100K_1M',
            ] if name in shared_records]
            shared_posix_small_reads = float(shared_records[small_read_columns].sum().sum())
            shared_posix_small_writes = float(shared_records[small_write_columns].sum().sum())

        if df_float is not None and not df_float.empty:
            time_cols = [c for c in (f'{mod}_F_READ_TIME', f'{mod}_F_WRITE_TIME',
                                     f'{mod}_F_META_TIME') if c in df_float.columns]
            if time_cols:
                _accumulate(df_float, time_cols, rank_time)

    active = (rank_bytes > 0) | (rank_time > 0)
    return {
        'RANK_IO_COUNT': float(active.sum()),
        'RANK_BYTES_MAX': float(rank_bytes.max()),
        'RANK_BYTES_MIN': float(rank_bytes.min()),
        'RANK_BYTES_VAR': float(rank_bytes.var()),
        'RANK_BYTES_GINI': _gini(rank_bytes),
        'RANK_TIME_MAX': float(rank_time.max()),
        'RANK_TIME_MIN': float(rank_time.min()),
        'RANK_TIME_VAR': float(rank_time.var()),
        'RANK_BYTES_TOTAL': float(rank_bytes.sum()),
        'RANK_TIME_TOTAL': float(rank_time.sum()),
        'RANK_SHARED_BYTES': shared_bytes,
        'SHARED_BYTE_IMBALANCE': shared_byte_imb,
        'SHARED_TIME_IMBALANCE': shared_time_imb,
        'FILE_WRITE_IMBALANCE': file_write_imb,
        'FILE_READ_IMBALANCE': file_read_imb,
        'SHARED_POSIX_READS': shared_posix_reads,
        'SHARED_POSIX_WRITES': shared_posix_writes,
        'SHARED_POSIX_SMALL_READS': shared_posix_small_reads,
        'SHARED_POSIX_SMALL_WRITES': shared_posix_small_writes,
    }


# ---------------------------------------------------------------------------
# Per-rank log aggregation for non-MPI benchmarks (DLIO, custom Python)
# ---------------------------------------------------------------------------

def parse_benchmark_job(rank_files: list[str | Path]) -> dict[str, object]:
    """Aggregate per-rank Darshan logs into a single job-level result.

    When Python/mpi4py programs run with LD_PRELOAD + DARSHAN_ENABLE_NONMPI=1,
    each MPI rank creates its own .darshan file (nprocs=1). This function
    merges the records of one launch into one job record and applies the same
    aggregation rules as ``parse_darshan_log()`` on a native MPI-mode log
    (the operation ``darshan-merge --shared-redux`` performs). Before the
    aggregation the per-process records feed ``_rank_statistics``, so the
    distribution of I/O over the processes is kept.

    Each file is one process. Its position in the sorted file list is used as
    the rank id (the file name carries the pid, not the MPI rank), which is
    enough for distribution statistics but does not identify rank 0.

    Parameters
    ----------
    rank_files : list of str or Path
        Paths to per-rank .darshan files belonging to the same job.
        Must contain at least one file. Files should be filtered to exclude
        startup probes (lscpu, uname) before calling.

    Returns
    -------
    dict
        Same structure as ``parse_darshan_log()``:
        ``{'job': {...}, 'counters': {...}, 'modules': [...], 'shared_file_flags': {...}}``

    Raises
    ------
    ValueError
        If ``rank_files`` is empty, or a file cannot be opened or one of its
        modules cannot be read. Incomplete module record sets are retained
        with an explicit indicator.
    """
    if not rank_files:
        raise ValueError("parse_benchmark_job needs at least one per-rank log")

    import pandas as pd

    nprocs = len(rank_files)
    frames = {mod: {'counters': [], 'fcounters': []} for mod, _ in FEATURE_MODULES}
    job_meta = None
    all_modules = set()
    all_name_records = {}
    partial_modules = set()
    start_times = []
    end_times = []

    for rank_idx, fpath in enumerate(sorted(rank_files)):
        try:
            report = _open_report(str(fpath))
        except Exception as exc:
            raise ValueError(f"cannot open per-rank log {fpath}: {exc}") from exc

        file_partial = _partial_feature_modules(report)
        if file_partial:
            partial_modules.update(file_partial)
            logger.warning("Incomplete module records in %s: %s", fpath,
                           ", ".join(sorted(file_partial)))

        if job_meta is None:
            job_meta = _job_metadata(report, nprocs=nprocs)
        jm = report.metadata['job']
        required_times = {'start_time_sec', 'start_time_nsec', 'end_time_sec', 'end_time_nsec'}
        missing_times = required_times - set(jm)
        if missing_times:
            raise ValueError(f"Darshan job timestamps lack {sorted(missing_times)} in {fpath}")
        start_times.append(jm['start_time_sec'] + jm['start_time_nsec'] / 1e9)
        end_times.append(jm['end_time_sec'] + jm['end_time_nsec'] / 1e9)

        # Every record of this file belongs to this process
        for mod, _ in FEATURE_MODULES:
            if mod not in report.modules:
                continue
            all_modules.add(mod)
            dfs = _read_module_frames(report, mod, fpath)
            for kind in ('counters', 'fcounters'):
                if kind in dfs:
                    df = dfs[kind].copy()
                    df['rank'] = rank_idx
                    frames[mod][kind].append(df)

        all_name_records.update(getattr(report, 'name_records', {}) or {})

    if job_meta is None:
        raise ValueError("no job metadata in any per-rank log")

    # Runtime of the launch: first start to last end over the processes
    if start_times and end_times:
        job_meta['start_time'] = min(start_times)
        job_meta['end_time'] = max(end_times)
        job_meta['runtime'] = max(end_times) - min(start_times)
    job_meta['modules'] = sorted(all_modules)

    counters = {}
    shared_file_flags = {}
    module_dfs = {}
    for mod, prefix in FEATURE_MODULES:
        if not frames[mod]['counters']:
            continue
        dfs = {'counters': pd.concat(frames[mod]['counters'], ignore_index=True)}
        if frames[mod]['fcounters']:
            dfs['fcounters'] = pd.concat(frames[mod]['fcounters'], ignore_index=True)
        module_dfs[mod] = dfs
        shared_file_flags[mod] = _extract_pydarshan_module(
            dfs, mod, counters, nprocs, prefix=prefix)

    counters.update(_rank_statistics(module_dfs, nprocs, all_name_records))
    file_facts = _file_facts(module_dfs, all_name_records)
    counters['num_files'] = file_facts['num_files']
    counters['num_data_files'] = file_facts['data_files']

    return {
        'job': job_meta,
        'counters': counters,
        'modules': sorted(all_modules),
        'shared_file_flags': shared_file_flags,
        'partial_modules': sorted(partial_modules),
    }
