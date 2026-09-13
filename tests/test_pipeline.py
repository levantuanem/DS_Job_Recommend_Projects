import pandas as pd
import pytest
from src.pipeline.pipeline import run_pipeline

# ============================================================
# BASIC PIPELINE TEST
# ============================================================
def test_pipeline_returns_result():
    result = run_pipeline()
    assert isinstance(result, dict)
    assert "X_train" in result
    assert "X_test" in result
    assert "y_train" in result
    assert "y_test" in result

# ============================================================
# TRAIN / TEST OUTPUT TEST
# ============================================================
def test_pipeline_train_test_outputs():
    result = run_pipeline()
    X_train = result["X_train"]
    X_test = result["X_test"]
    y_train = result["y_train"]
    y_test = result["y_test"]
    assert X_train is not None
    assert X_test is not None
    assert y_train is not None
    assert y_test is not None
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert len(y_train) > 0
    assert len(y_test) > 0

# ============================================================
# TRAIN / TEST SIZE CONSISTENCY
# ============================================================
def test_pipeline_train_test_size_consistency():
    result = run_pipeline()
    X_train = result["X_train"]
    X_test = result["X_test"]
    y_train = result["y_train"]
    y_test = result["y_test"]
    assert X_train.shape[0] == len(y_train)
    assert X_test.shape[0] == len(y_test)

# ============================================================
# TRAIN / TEST FEATURE CONSISTENCY
# ============================================================
def test_pipeline_feature_dimension_consistency():
    result = run_pipeline()
    X_train = result["X_train"]
    X_test = result["X_test"]
    assert X_train.shape[1] == X_test.shape[1]

# ============================================================
# TARGET VALIDATION
# ============================================================
def test_pipeline_target_is_not_empty():
    result = run_pipeline()
    y_train = result["y_train"]
    y_test = result["y_test"]
    assert not y_train.empty
    assert not y_test.empty

# ============================================================
# TARGET TYPE
# ============================================================
def test_pipeline_target_type():
    result = run_pipeline()
    y_train = result["y_train"]
    y_test = result["y_test"]
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)

# ============================================================
# FUTURE PIPELINE COMPONENTS
# ============================================================
def test_future_pipeline_components_are_empty():
    result = run_pipeline()
    assert result["model"] is None
    assert result["metrics"] is None
    assert result["predictions"] is None

