"""Data pipeline for Darshan log feature extraction."""

from src.data.parse_darshan import parse_darshan_log
from src.data.feature_extraction import (
    extract_features,
    extract_raw_features,
    get_feature_names,
    get_raw_feature_names,
    get_info_columns,
    FEATURE_GROUPS,
)
from src.data.preprocessing import (
    stage2_clean,
    stage3_engineer,
    compute_statistics,
    stage5_normalize,
    create_splits,
    load_preprocessing_config,
)
from src.data.batch_extract import batch_extract
from src.data.drishti_labeling import (
    generate_heuristic_labels,
    generate_silver_labels,  # backward compat alias
    compute_drishti_codes,
    codes_to_labels,
    DIMENSION_NAMES,
    DRISHTI_THRESHOLDS,
)

__all__ = [
    'parse_darshan_log',
    'extract_features',
    'extract_raw_features',
    'get_feature_names',
    'get_raw_feature_names',
    'get_info_columns',
    'FEATURE_GROUPS',
    'stage2_clean',
    'stage3_engineer',
    'compute_statistics',
    'stage5_normalize',
    'create_splits',
    'load_preprocessing_config',
    'batch_extract',
    'generate_heuristic_labels',
    'generate_silver_labels',  # backward compat alias
    'compute_drishti_codes',
    'codes_to_labels',
    'DIMENSION_NAMES',
    'DRISHTI_THRESHOLDS',
]
