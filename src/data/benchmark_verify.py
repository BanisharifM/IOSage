"""Verify that a benchmark sample shows the I/O pattern its label claims.

One rule per label dimension, each an observable condition on the engineered
features (or, for Lustre striping, on the log itself). A sample labeled with a
dimension must satisfy that dimension's rule; a sample labeled healthy must
satisfy none of the bottleneck rules and move data at a minimum rate. A label
set without any rule to check is an error, not a pass.

Thresholds: Drishti's boundaries (small request under 1 MB, straggler share
0.15, size imbalance 0.3) and the verification table of
``docs/1_strategy/paper_materials.md`` (metadata share 10 percent,
sequential share 80 percent, fsync count, files per rank).
"""

import logging
import shutil
import subprocess
import sys
from pathlib import Path

from src.data.preprocessing import load_preprocessing_config

logger = logging.getLogger(__name__)

# Dimension names (must match drishti_labeling.py)
DIMENSION_NAMES = [
    'access_granularity', 'metadata_intensity', 'parallelism_efficiency',
    'access_pattern', 'interface_choice', 'file_strategy',
    'throughput_utilization', 'healthy',
]
BOTTLENECK_DIMENSIONS = DIMENSION_NAMES[:7]

# Cleaning rule of the production data (configs/preprocessing.yaml). A
# benchmark sample below it is reported, not failed: the constructed corpus
# holds sub-second runs and zero-byte metadata runs on purpose, and whether
# they belong in the training set is a dataset decision, not a verification
# outcome.
_CLEANING = load_preprocessing_config()['cleaning']
MIN_RUNTIME_SECONDS = float(_CLEANING['min_duration_seconds'])
MIN_TOTAL_BYTES = float(_CLEANING['min_total_bytes'])
MIN_IO_OPS = float(_CLEANING['min_io_ops'])

# Drishti thresholds (drishti/includes/config.py): a request under 1 MB is
# small (P05/P06 fire when small requests exceed 10 percent of the requests
# and 1000 in number), M02/M03 fire when a job with over 1000 MPI-IO
# operations makes fewer than half of them collective
SMALL_REQUEST_SHARE = 0.10
SMALL_REQUEST_COUNT = 1000
COLLECTIVE_MIN_OPS = 1000
COLLECTIVE_SHARE = 0.5
METADATA_TIME_SHARE = 0.10          # paper_materials: meta time over 10 percent
SEQUENTIAL_SHARE = 0.80             # paper_materials: random when under 80 percent
FSYNC_PER_WRITE = 0.5               # construction: a sync after (nearly) every write
HEALTHY_MIN_BYTES_PER_S = 1024.0    # paper_materials: healthy moves over 1 KB/s
LARGE_FILE_BYTES = 16 * 1048576     # Delta PFL: files above 16 MiB span several OSTs

# Rank imbalance: Drishti's size-imbalance threshold (imbalance_size, 0.3:
# bytes of the busiest rank minus bytes of the idlest rank, over the busiest
# rank) applied to the per-rank totals of the whole job, or Drishti's
# straggler threshold (imbalance_stragglers, 0.15) on a shared record. Bytes
# rather than time, because the time of a balanced run varies by tens of
# percent between ranks on a shared file system while its bytes do not.
RANK_IMBALANCE_RANGE_RATIO = 0.3
SHARED_STRAGGLER_SHARE = 0.15


def rank_imbalance_present(features):
    """True when the I/O bytes of a job are spread unevenly over its ranks.

    ``rank_byte_range_ratio`` covers file-per-process layouts,
    ``SHARED_BYTE_IMBALANCE`` a file opened by all ranks. A single process
    job is never imbalanced.
    """
    if features['nprocs'] <= 1:
        return False
    return (features['rank_byte_range_ratio'] > RANK_IMBALANCE_RANGE_RATIO
            or features['SHARED_BYTE_IMBALANCE'] > SHARED_STRAGGLER_SHARE)


# ---------------------------------------------------------------------------
# Lustre striping (verification only: production logs carry no LUSTRE module)
# ---------------------------------------------------------------------------

def _darshan_parser():
    exe = shutil.which('darshan-parser') or str(Path(sys.prefix) / 'bin' / 'darshan-parser')
    if not Path(exe).exists():
        raise FileNotFoundError("darshan-parser not found (needed to read LUSTRE records)")
    return exe


def lustre_stripe_counts(log_paths):
    """``{record id: stripe count}`` from the LUSTRE records of the logs.

    PyDarshan 3.5.0 cannot decode LUSTRE records, so this reads the
    ``LUSTRE_COMP*_STRIPE_COUNT`` lines of ``darshan-parser``; the smallest
    component count of a file is kept.
    """
    counts = {}
    exe = _darshan_parser()
    for path in log_paths:
        out = subprocess.run([exe, str(path)], capture_output=True, text=True, timeout=300)
        if out.returncode != 0:
            raise RuntimeError(f"darshan-parser failed on {path}: {out.stderr[:200]}")
        for line in out.stdout.splitlines():
            if not line.startswith('LUSTRE\t'):
                continue
            parts = line.split('\t')
            if len(parts) > 4 and parts[3].endswith('_STRIPE_COUNT'):
                rec = int(parts[2])
                counts[rec] = min(counts.get(rec, 1 << 30), int(parts[4]))
    return counts


def single_stripe_large_file(log_paths, offsets_by_id):
    """True when a file accessed past ``LARGE_FILE_BYTES`` sits on one OST.

    ``offsets_by_id`` maps record id to the highest offset read or written.
    Delta's PFL layout puts files above 16 MiB on several OSTs, so a large
    file with stripe count 1 was striped on purpose.
    """
    large = {rid for rid, offset in offsets_by_id.items() if offset >= LARGE_FILE_BYTES}
    if not large:
        return False
    counts = lustre_stripe_counts(log_paths)
    return any(counts.get(rid, 0) == 1 for rid in large)


# ---------------------------------------------------------------------------
# Dimension rules, evaluated at the layer the application used: when a job
# issues MPI-IO requests, its request sizes are the MPI-IO aggregate
# histograms and its interface use is judged by Drishti's collective rule;
# otherwise the POSIX counters describe the application directly.
# ---------------------------------------------------------------------------

_POSIX_SMALL = {
    'READ': ['POSIX_SIZE_READ_0_100', 'POSIX_SIZE_READ_100_1K', 'POSIX_SIZE_READ_1K_10K',
             'POSIX_SIZE_READ_10K_100K', 'POSIX_SIZE_READ_100K_1M'],
    'WRITE': ['POSIX_SIZE_WRITE_0_100', 'POSIX_SIZE_WRITE_100_1K', 'POSIX_SIZE_WRITE_1K_10K',
              'POSIX_SIZE_WRITE_10K_100K', 'POSIX_SIZE_WRITE_100K_1M'],
}
_MPIIO_SMALL = {d: [c.replace('POSIX_SIZE_', 'MPIIO_SIZE_').replace(f'{d}_', f'{d}_AGG_', 1)
                    for c in cols] for d, cols in _POSIX_SMALL.items()}


def _mpiio_ops(f):
    return (f['MPIIO_INDEP_READS'] + f['MPIIO_INDEP_WRITES'] + f['MPIIO_COLL_READS']
            + f['MPIIO_COLL_WRITES'] + f['MPIIO_NB_READS'] + f['MPIIO_NB_WRITES'])


def small_requests(f):
    """Drishti P05/P06 at the application layer: (present, detail)."""
    if _mpiio_ops(f) > 0:
        counts = {'READ': f['MPIIO_INDEP_READS'] + f['MPIIO_COLL_READS'] + f['MPIIO_NB_READS'],
                  'WRITE': f['MPIIO_INDEP_WRITES'] + f['MPIIO_COLL_WRITES'] + f['MPIIO_NB_WRITES']}
        small = {d: sum(f[c] for c in cols) for d, cols in _MPIIO_SMALL.items()}
        layer = 'mpiio'
    else:
        counts = {'READ': f['POSIX_READS'], 'WRITE': f['POSIX_WRITES']}
        small = {d: sum(f[c] for c in cols) for d, cols in _POSIX_SMALL.items()}
        layer = 'posix'
    present = any(small[d] > SMALL_REQUEST_COUNT and small[d] / max(counts[d], 1) > SMALL_REQUEST_SHARE
                  for d in ('READ', 'WRITE'))
    detail = ' '.join(f"{d.lower()}_small={small[d]:.0f}/{counts[d]:.0f}" for d in ('READ', 'WRITE'))
    return present, f"{layer} {detail}"


def _sequential_share(f):
    ops = f['POSIX_READS'] + f['POSIX_WRITES']
    return (f['POSIX_SEQ_READS'] + f['POSIX_SEQ_WRITES']) / max(ops, 1)


def interface_problem(f):
    """Drishti M02/M03 (independent MPI-IO at scale), or POSIX alone on a
    file that all ranks share: (present, detail)."""
    ops = _mpiio_ops(f)
    coll = f['MPIIO_COLL_READS'] + f['MPIIO_COLL_WRITES']
    if ops > 0:
        present = ops > COLLECTIVE_MIN_OPS and coll / ops < COLLECTIVE_SHARE
        return present, f"mpiio_ops={ops:.0f} collective={coll:.0f}"
    present = bool(f['is_shared_file']) and f['nprocs'] > 1
    return present, f"posix_only shared={int(f['is_shared_file'])} nprocs={f['nprocs']}"


def _bytes_per_second(f):
    total = f['POSIX_BYTES_READ'] + f['POSIX_BYTES_WRITTEN']
    return total / max(f['runtime_seconds'], 1e-9)


def bottleneck_rules(features, context):
    """``{dimension: (present, detail)}`` for the seven bottleneck dimensions.

    ``context`` carries what the features cannot: ``log_paths`` and
    ``offsets`` for the striping check (throughput) and ``data_files`` (files
    with bytes moved, standard streams excluded) for the file-per-process rule.
    """
    f = features
    seq = _sequential_share(f)
    fsync_per_write = f['POSIX_FSYNCS'] / max(f['POSIX_WRITES'], 1)
    syncing = fsync_per_write >= FSYNC_PER_WRITE
    striped = (not syncing
               and single_stripe_large_file(context['log_paths'], context['offsets']))
    small_present, small_detail = small_requests(f)
    iface_present, iface_detail = interface_problem(f)
    return {
        'access_granularity': (small_present, small_detail),
        'metadata_intensity': (
            f['metadata_time_ratio'] > METADATA_TIME_SHARE
            or f['POSIX_BYTES_WRITTEN'] + f['POSIX_BYTES_READ'] == 0,
            f"metadata_time_ratio={f['metadata_time_ratio']:.3f} "
            f"bytes={f['POSIX_BYTES_WRITTEN'] + f['POSIX_BYTES_READ']:.0f}"),
        'parallelism_efficiency': (
            rank_imbalance_present(f),
            f"range_ratio={f['rank_byte_range_ratio']:.3f} shared_imb={f['SHARED_BYTE_IMBALANCE']:.3f} nprocs={f['nprocs']}"),
        'access_pattern': (seq < SEQUENTIAL_SHARE, f"sequential_share={seq:.3f}"),
        'interface_choice': (iface_present, iface_detail),
        'file_strategy': (
            f['nprocs'] > 1 and context['data_files'] >= f['nprocs'],
            f"data_files={context['data_files']} nprocs={f['nprocs']}"),
        'throughput_utilization': (
            syncing or striped,
            f"fsync_per_write={fsync_per_write:.3f} single_stripe_large_file={striped}"),
    }


def cleaning_rule(features):
    """(passes, reason) of the production cleaning rule for this sample."""
    runtime = features['runtime_seconds']
    total_bytes = features['POSIX_BYTES_READ'] + features['POSIX_BYTES_WRITTEN']
    total_ops = features['POSIX_READS'] + features['POSIX_WRITES']
    for ok, reason in ((runtime >= MIN_RUNTIME_SECONDS, f'runtime={runtime:.1f}s < {MIN_RUNTIME_SECONDS}s'),
                       (total_bytes >= MIN_TOTAL_BYTES, f'total_bytes={total_bytes:.0f} < {MIN_TOTAL_BYTES}'),
                       (total_ops >= MIN_IO_OPS, f'total_ops={total_ops:.0f} < {MIN_IO_OPS}')):
        if not ok:
            return False, reason
    return True, ''


def verify_benchmark_log(features, intended_labels, context):
    """Check that a sample's features match its intended labels.

    Parameters
    ----------
    features : dict
        Engineered features of the sample (``preprocessing.engineer_one``).
    intended_labels : dict
        ``{dimension: 0 or 1}`` for the eight dimensions; at least one must
        be 1 and healthy excludes the bottleneck dimensions.
    context : dict
        ``log_paths`` (the sample's Darshan files), ``offsets`` (record id to
        highest offset) and ``data_files``, from ``benchmark_logs.posix_file_facts``.

    Returns
    -------
    (passed, report)
        ``report['checks']`` holds every rule evaluated with its value;
        ``report['cleaning_rule']`` says whether the production cleaning rule
        would keep the sample (informational).
    """
    report = {'checks': {}, 'passed_checks': 0, 'total_checks': 0}

    positives = [d for d in DIMENSION_NAMES if intended_labels.get(d, 0) == 1]
    if not positives:
        raise ValueError("intended labels name no dimension")
    if 'healthy' in positives and len(positives) > 1:
        raise ValueError(f"healthy cannot be combined with {positives}")

    report['cleaning_rule'], report['cleaning_reason'] = cleaning_rule(features)

    rules = bottleneck_rules(features, context)

    def record(name, passed, detail):
        report['total_checks'] += 1
        report['checks'][name] = {'status': 'pass' if passed else 'fail', 'value': detail}
        if passed:
            report['passed_checks'] += 1

    if 'healthy' in positives:
        rate = _bytes_per_second(features)
        record('healthy/data_rate', rate >= HEALTHY_MIN_BYTES_PER_S, f"bytes_per_s={rate:.0f}")
        for dim, (present, detail) in rules.items():
            record(f'healthy/no_{dim}', not present, detail)
    else:
        for dim in positives:
            present, detail = rules[dim]
            record(f'{dim}/rule', present, detail)

    if report['total_checks'] == 0:
        raise AssertionError("no rule evaluated")
    passed = report['passed_checks'] == report['total_checks']
    if not passed:
        fails = {k: v['value'] for k, v in report['checks'].items() if v['status'] != 'pass'}
        logger.warning("Benchmark FAILED verification: %s", fails)
    return passed, report
