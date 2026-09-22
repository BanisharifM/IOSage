"""Small synthetic frames with the full raw schema, for the pipeline tests."""
import numpy as np
import pandas as pd

from src.data.feature_extraction import FEATURE_SCHEMA_VERSION, get_raw_feature_names


def raw_frame(n_rows, start=0):
    """A raw-feature frame of ``n_rows`` jobs with the current schema: every
    raw column at 0 except a runtime, one POSIX write of 4 KiB, one process
    and the info columns."""
    df = pd.DataFrame(0.0, index=range(start, start + n_rows), columns=get_raw_feature_names())
    df['nprocs'] = 1
    df['runtime_seconds'] = 20.0
    df['has_posix'] = 1
    df['POSIX_WRITES'] = 1.0
    df['POSIX_BYTES_WRITTEN'] = 4096.0
    df['POSIX_F_WRITE_TIME'] = 0.5
    df['_schema_version'] = FEATURE_SCHEMA_VERSION
    df['_jobid'] = np.arange(start, start + n_rows)
    df['_uid'] = 1
    df['_start_time'] = 1_700_000_000 + np.arange(n_rows) * 60
    df['_end_time'] = df['_start_time'] + 20
    df['_modules'] = 'POSIX'
    df['_log_version'] = '3.41'
    return df
