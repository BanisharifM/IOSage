"""
IOPrescriber Step 1: ML Bottleneck Detection.

Loads a model bundle written by ``src.models.biquality`` (one classifier per
problem label with its ordered feature contract and decision threshold) and
predicts every registered dimension of a Darshan log. Healthy is derived from
the problem decisions.

Input: Darshan log path OR pre-extracted feature dict
Output: {dimension: confidence} for all dimensions + detected list
"""

import logging
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.biquality import BOTTLENECK_DIMENSIONS, DIMENSION_NAMES, predict, validate_bundle  # noqa: E402

logger = logging.getLogger(__name__)

DIMENSIONS = list(DIMENSION_NAMES)


class Detector:
    """ML-based multi-label I/O bottleneck detector."""

    def __init__(self, model_path, threshold=None):
        """``model_path``: a bundle pickle from ``scripts/train_biquality.py``.
        ``threshold`` overrides the bundle's decision threshold (for studies
        that vary it; the bundle's value is the trained protocol's)."""
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)
        validate_bundle(bundle)
        if not bundle['final_evaluation']:
            raise ValueError("detector requires a bundle from an explicit final evaluation run")
        if threshold is not None:
            if not 0 < float(threshold) < 1:
                raise ValueError("threshold override must be in (0, 1)")
            bundle = dict(bundle, decision_threshold=float(threshold))
        self.bundle = bundle
        self.models = bundle["models"]
        self.feature_cols = list(bundle["feature_names"])
        self.threshold = bundle["decision_threshold"]
        logger.info("Detector loaded: %s seed %s, %d features, threshold=%.2f",
                    bundle["model_type"], bundle["seed"], len(self.feature_cols), self.threshold)

    def feature_vector(self, features_dict):
        """The bundle's feature order as one row; a missing feature is an error."""
        missing = [c for c in self.feature_cols if c not in features_dict]
        if missing:
            raise KeyError(f"{len(missing)} features missing from the input, first: {missing[:5]}")
        return np.array([[features_dict[col] for col in self.feature_cols]], dtype=np.float32)

    def detect_from_features(self, features_dict):
        """Detect bottlenecks from a feature dictionary.

        Returns:
            predictions: dict of {dimension: confidence}; healthy's value is
                1 minus the highest bottleneck probability
            detected: bottleneck dimensions at or above the threshold, or
                ["healthy"]
        """
        proba, decisions = predict(self.bundle, self.feature_vector(features_dict))
        predictions = {dim: round(float(proba[0, i]), 4) for i, dim in enumerate(BOTTLENECK_DIMENSIONS)}
        predictions["healthy"] = round(float(1.0 - proba[0].max()), 4)
        detected = [dim for i, dim in enumerate(BOTTLENECK_DIMENSIONS) if decisions[0, i]]
        return predictions, detected or ["healthy"]

    def detect_from_darshan(self, darshan_path):
        """Detect bottlenecks directly from a Darshan log file.

        Args:
            darshan_path: path to .darshan file

        Returns:
            predictions, detected, features_dict
        """
        from src.data.parse_darshan import parse_darshan_log
        from src.data.preprocessing import engineer_one

        features_dict = engineer_one(parse_darshan_log(str(darshan_path), strict=True))

        predictions, detected = self.detect_from_features(features_dict)
        return predictions, detected, features_dict
