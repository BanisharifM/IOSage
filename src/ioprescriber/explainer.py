"""
IOPrescriber Step 2: SHAP Feature Attribution.

Computes per-label SHAP values to explain WHY each bottleneck was detected.
Provides top-K features with direction (positive/negative impact).

Input: ML models + feature vector
Output: {dimension: [{feature, value, shap_value, direction}]}
"""

import logging
from pathlib import Path

import numpy as np
import shap

logger = logging.getLogger(__name__)

DIMENSIONS = [
    "access_granularity", "metadata_intensity", "parallelism_efficiency",
    "access_pattern", "interface_choice", "file_strategy",
    "throughput_utilization", "healthy",
]


class Explainer:
    """SHAP-based per-label feature attribution."""

    def __init__(self, models, feature_cols, top_k=10):
        self.models = models
        self.feature_cols = feature_cols
        self.top_k = top_k
        self.explainers = {}

        # Pre-build TreeExplainers
        for dim in DIMENSIONS:
            if dim in models:
                self.explainers[dim] = shap.TreeExplainer(models[dim])

        logger.info("Explainer initialized: %d dimensions, top_k=%d",
                    len(self.explainers), top_k)

    def explain(self, X_single, detected_dims=None):
        """Compute SHAP values for a single sample.

        Args:
            X_single: numpy array shape (1, n_features) or (n_features,)
            detected_dims: list of dims to explain (None = all detected)

        Returns:
            dict of {dimension: [{feature, value, shap_value, direction}]}
        """
        if X_single.ndim == 1:
            X_single = X_single.reshape(1, -1)

        dims_to_explain = detected_dims or list(self.explainers.keys())

        attributions = {}
        for dim in dims_to_explain:
            if dim not in self.explainers:
                continue

            sv = self.explainers[dim].shap_values(X_single)
            if isinstance(sv, list):
                sv = sv[1]  # positive class
            sv = sv[0]  # single sample

            # Top-K by absolute SHAP value
            top_idx = np.argsort(np.abs(sv))[-self.top_k:][::-1]

            attributions[dim] = []
            for idx in top_idx:
                attributions[dim].append({
                    "feature": self.feature_cols[idx],
                    "value": round(float(X_single[0, idx]), 6),
                    "shap_value": round(float(sv[idx]), 6),
                    "direction": "increases_risk" if sv[idx] > 0 else "decreases_risk",
                    "abs_importance": round(float(abs(sv[idx])), 6),
                })

        return attributions
