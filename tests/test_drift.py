import numpy as np
import pytest

from app.drift import DriftDetector

FEATURE_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def test_no_drift_on_same_distribution():
    rng = np.random.default_rng(42)
    ref = rng.normal(loc=5.0, scale=1.0, size=(500, 4))
    cur = rng.normal(loc=5.0, scale=1.0, size=(500, 4))
    detector = DriftDetector(ref, FEATURE_NAMES)
    result = detector.detect(cur, alpha=0.05)
    assert result["drift_detected"] is False
    assert result["n_drifted_features"] == 0


def test_drift_on_shifted_distribution():
    rng = np.random.default_rng(42)
    ref = rng.normal(loc=5.0, scale=1.0, size=(500, 4))
    cur = rng.normal(loc=8.0, scale=1.0, size=(500, 4))
    detector = DriftDetector(ref, FEATURE_NAMES)
    result = detector.detect(cur, alpha=0.05)
    assert result["drift_detected"] is True
    assert result["n_drifted_features"] == 4
    for feat in FEATURE_NAMES:
        assert result["per_feature"][feat]["p_value"] < 0.05


def test_drift_partial():
    rng = np.random.default_rng(0)
    ref = rng.normal(loc=5.0, scale=1.0, size=(500, 4))
    cur = ref.copy()
    cur[:, 2] = rng.normal(loc=10.0, scale=1.0, size=500)
    detector = DriftDetector(ref, FEATURE_NAMES)
    result = detector.detect(cur, alpha=0.05)
    assert result["drift_detected"] is True
    assert "petal_length" in result["drifted_features"]


def test_invalid_reference_shape():
    with pytest.raises(ValueError):
        DriftDetector(np.array([1.0, 2.0, 3.0]), FEATURE_NAMES)


def test_feature_names_mismatch():
    with pytest.raises(ValueError):
        DriftDetector(np.zeros((10, 3)), FEATURE_NAMES)


def test_current_columns_mismatch():
    detector = DriftDetector(np.zeros((10, 4)), FEATURE_NAMES)
    with pytest.raises(ValueError):
        detector.detect(np.zeros((10, 3)))
