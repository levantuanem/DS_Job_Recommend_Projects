from types import SimpleNamespace

import pandas as pd

from src.pipeline import pipeline as pipeline_module


def test_pipeline_orchestration_without_training_side_effects(monkeypatch):
    x_train = pd.DataFrame({"feature": [1, 2]})
    x_test = pd.DataFrame({"feature": [3]})
    y_train = pd.Series([0, 1])
    y_test = pd.Series([0])
    label_encoder = SimpleNamespace(classes_=["ENTRY LEVEL", "DIRECTOR"])
    search = SimpleNamespace(best_params_={"clf__C": 1}, best_score_=0.7)

    monkeypatch.setattr(pipeline_module, "run_data_cleaning", lambda **kwargs: None)
    monkeypatch.setattr(
        pipeline_module,
        "get_train_test_data",
        lambda **kwargs: (x_train, x_test, y_train, y_test, label_encoder),
    )
    monkeypatch.setattr(
        pipeline_module,
        "train_all_models",
        lambda **kwargs: pd.DataFrame({"cv_f1_macro_mean": [0.7]}),
    )
    monkeypatch.setattr(pipeline_module, "tune_best_model", lambda: search)
    monkeypatch.setattr(
        pipeline_module,
        "evaluate_best_model",
        lambda: {"test_f1_macro": 0.7},
    )

    result = pipeline_module.run_pipeline()

    assert result["X_train"] is x_train
    assert result["X_test"] is x_test
    assert result["metrics"]["test_f1_macro"] == 0.7
    assert result["predictions"] is None

