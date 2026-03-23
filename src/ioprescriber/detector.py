"""
IOPrescriber Step 1: ML Bottleneck Detection.

Wraps the Phase 2 biquality XGBoost model to detect 8 I/O bottleneck
dimensions from Darshan features.

Input: Darshan log path OR pre-extracted feature dict
Output: {dimension: confidence} for all 8 dimensions + detected list
"""

import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

DIMENSIONS = [
    "access_granularity", "metadata_intensity", "parallelism_efficiency",
    "access_pattern", "interface_choice", "file_strategy",
    "throughput_utilization", "healthy",
]


class Detector:
    """ML-based multi-label I/O bottleneck detector."""

    def __init__(self, model_path=None, config_path=None, threshold=0.3):
        config_path = config_path or PROJECT_DIR / "configs" / "training.yaml"
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        model_path = model_path or PROJECT_DIR / "models" / "phase2" / "xgboost_biquality_w100.pkl"
        with open(model_path, "rb") as f:
            self.models = pickle.load(f)

        self.threshold = threshold
        self.feature_cols = self._get_feature_cols()
        logger.info("Detector loaded: %d models, %d features, threshold=%.2f",
                    len(self.models), len(self.feature_cols), threshold)

    def _get_feature_cols(self):
        prod_feat = pd.read_parquet(
            PROJECT_DIR / self.config["paths"]["production_features"],
        )
        exclude = set(self.config.get("exclude_features", []))
        for col in prod_feat.columns:
            if col.startswith("_") or col.startswith("drishti_"):
                exclude.add(col)
        return [c for c in prod_feat.columns if c not in exclude]

    def detect_from_features(self, features_dict):
        """Detect bottlenecks from a feature dictionary.

        Args:
            features_dict: dict of {feature_name: value}

        Returns:
            predictions: dict of {dimension: confidence}
            detected: list of dimension names with confidence > threshold
        """
        X = np.array([[features_dict.get(col, 0) for col in self.feature_cols]],
                      dtype=np.float32)

        predictions = {}
        for dim in DIMENSIONS:
            if dim in self.models:
                predictions[dim] = round(float(self.models[dim].predict_proba(X)[0][1]), 4)

        detected = [d for d in DIMENSIONS
                     if predictions.get(d, 0) > self.threshold and d != "healthy"]
        if not detected:
            detected = ["healthy"]

        return predictions, detected

    def detect_from_darshan(self, darshan_path):
        """Detect bottlenecks directly from a Darshan log file.

        Args:
            darshan_path: path to .darshan file

        Returns:
            predictions, detected, features_dict
        """
        from src.data.parse_darshan import parse_darshan_log
        from src.data.feature_extraction import extract_raw_features
        from src.data.preprocessing import stage3_engineer

        parsed = parse_darshan_log(str(darshan_path))
        if parsed is None:
            raise ValueError(f"Failed to parse Darshan log: {darshan_path}")

        raw_features = extract_raw_features(parsed)
        df = pd.DataFrame([raw_features])
        df = stage3_engineer(df)
        features_dict = df.iloc[0].to_dict()

        predictions, detected = self.detect_from_features(features_dict)
        return predictions, detected, features_dict
