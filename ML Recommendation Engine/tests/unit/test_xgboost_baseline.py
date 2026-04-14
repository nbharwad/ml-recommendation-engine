"""
Unit Tests — XGBoost Baseline Model
=====================================
Covers: fallback behaviour when xgboost is unavailable or model file is missing,
and the neutral-score fix (P0-3: replace random fallback with 0.05 constant).
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_features(batch_size: int = 10) -> np.ndarray:
    """Return a (batch_size, 42) float32 feature matrix."""
    rng = np.random.default_rng(seed=0)
    return rng.random((batch_size, 42)).astype(np.float32)


def _inference_no_model() -> object:
    """Return an XGBoostInference instance whose _model is None."""
    from ml.models.xgboost_baseline.model import XGBoostInference
    inf = XGBoostInference.__new__(XGBoostInference)
    inf.model_path = "/nonexistent/model.json"
    inf._model = None
    return inf


# ---------------------------------------------------------------------------
# Fallback path: xgboost unavailable at import time
# ---------------------------------------------------------------------------

class TestNeutralFallbackWhenXGBoostMissing:
    """predict() must return a deterministic 0.05 array when _model is None."""

    def test_fallback_values_are_all_0_05(self):
        inf = _inference_no_model()
        features = _make_features(batch_size=8)
        preds = inf.predict(features)
        assert np.all(preds == 0.05), f"Expected all 0.05, got {preds}"

    def test_fallback_shape_matches_batch(self):
        inf = _inference_no_model()
        for batch_size in (1, 5, 100, 1000):
            features = _make_features(batch_size=batch_size)
            preds = inf.predict(features)
            assert preds.shape == (batch_size,), (
                f"Shape mismatch for batch_size={batch_size}: {preds.shape}"
            )

    def test_fallback_dtype_is_float32(self):
        inf = _inference_no_model()
        preds = inf.predict(_make_features())
        assert preds.dtype == np.float32, f"Expected float32, got {preds.dtype}"

    def test_fallback_is_deterministic(self):
        """Two calls with identical inputs must produce identical outputs."""
        inf = _inference_no_model()
        features = _make_features(batch_size=20)
        assert np.array_equal(inf.predict(features), inf.predict(features))

    def test_fallback_is_not_random(self):
        """Scores must not vary across items — the old random bug would fail this."""
        inf = _inference_no_model()
        preds = inf.predict(_make_features(batch_size=50))
        assert preds.min() == preds.max(), (
            "Fallback scores vary — random behaviour has regressed"
        )


# ---------------------------------------------------------------------------
# load() when xgboost package is absent
# ---------------------------------------------------------------------------

class TestLoadWhenXGBoostUnavailable:
    """load() must leave _model=None and log a warning when xgboost is absent."""

    def test_model_stays_none_when_xgboost_missing(self, tmp_path):
        with patch(
            "ml.models.xgboost_baseline.model.XGBOOST_AVAILABLE", False
        ):
            from ml.models.xgboost_baseline.model import XGBoostInference
            inf = XGBoostInference(model_path=str(tmp_path / "model.json"))
            inf.load()
            assert inf._model is None

    def test_load_logs_neutral_fallback_message(self, tmp_path, caplog):
        import logging
        with patch(
            "ml.models.xgboost_baseline.model.XGBOOST_AVAILABLE", False
        ):
            from ml.models.xgboost_baseline.model import XGBoostInference
            inf = XGBoostInference(model_path=str(tmp_path / "model.json"))
            with caplog.at_level(logging.WARNING, logger="root"):
                inf.load()
        assert "neutral" in caplog.text.lower(), (
            "Log message should mention 'neutral', not 'random'"
        )
        assert "random" not in caplog.text.lower(), (
            "Stale 'random' wording still present in log message"
        )


# ---------------------------------------------------------------------------
# load() when model file is missing (xgboost available but file absent)
# ---------------------------------------------------------------------------

class TestLoadWithMissingModelFile:
    """load() must catch file-not-found and leave _model=None."""

    def test_missing_file_sets_model_to_none(self, tmp_path):
        # xgb is only defined in the module when the import succeeded.
        # Simulate: xgboost available, but Booster.load_model raises (file absent).
        mock_booster = MagicMock()
        mock_booster.load_model.side_effect = Exception("file not found")

        import ml.models.xgboost_baseline.model as xgb_module
        real_xgb_available = xgb_module.XGBOOST_AVAILABLE

        if not real_xgb_available:
            pytest.skip("xgboost not installed — skipping file-missing path test")

        original_booster = xgb_module.xgb.Booster
        xgb_module.xgb.Booster = MagicMock(return_value=mock_booster)
        try:
            from ml.models.xgboost_baseline.model import XGBoostInference
            inf = XGBoostInference(model_path=str(tmp_path / "missing.json"))
            inf.load()
            assert inf._model is None
        finally:
            xgb_module.xgb.Booster = original_booster
