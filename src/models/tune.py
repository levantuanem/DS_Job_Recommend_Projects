"""
src/models/tune.py

Member 4 - STEP 2: Hyperparameter tuning of the baseline (Logistic Regression only).

Reads models/best_model_info.json (written by train.py - always
"logistic_regression_baseline" now), runs RandomizedSearchCV with
StratifiedKFold cross-validation (scoring = Macro-F1, per README 9.6/9.9),
and saves the tuned estimator as models/best_model.pkl - the ONE file
evaluate.py and predict.py use downstream.

Run (from the repository root):
    python -m src.models.tune
"""

import json

import joblib
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression

from src.models.utils import (
    get_train_test_data,
    save_json,
    BEST_MODEL_INFO_PATH,
    BEST_MODEL_PATH,
    TUNING_RESULTS_PATH,
)

RANDOM_STATE = 42
CV_FOLDS = 5
N_ITER = 25  # RandomizedSearchCV budget - raise for a more thorough (slower) search

# =========================
# README 9.6 Hyperparameter search space - Logistic Regression only
# =========================
PARAM_GRIDS = {
    "logistic_regression_baseline": (
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        {
            "C": [0.01, 0.1, 1, 10, 100],
            "solver": ["lbfgs", "saga"],
        },
    ),
}


def tune_best_model():
    if not BEST_MODEL_INFO_PATH.exists():
        raise FileNotFoundError(
            f"{BEST_MODEL_INFO_PATH} not found. Run train.py first: python -m src.models.train"
        )

    with open(BEST_MODEL_INFO_PATH, "r", encoding="utf-8") as f:
        best_info = json.load(f)

    model_name = best_info["best_model_name"]
    print(f"Tuning best candidate from train.py: {model_name}")

    if model_name not in PARAM_GRIDS:
        raise ValueError(
            f"No hyperparameter grid defined for '{model_name}'. "
            f"Add one to PARAM_GRIDS in src/models/tune.py."
        )

    # Reuses the EXACT same cached train/test split train.py used (no refitting on test).
    x_train, x_test, y_train, y_test, label_encoder = get_train_test_data(
        apply_feature_selection=best_info.get("apply_feature_selection", False),
        k=best_info.get("k", 1000),
    )

    base_model, param_grid = PARAM_GRIDS[model_name]
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=N_ITER,
        scoring="f1_macro",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    print("Running RandomizedSearchCV (this can take a while)...")
    search.fit(x_train, y_train)

    print(f"\nBest CV Macro-F1: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")

    # ---- Save the tuned model as THE official best model used downstream ----
    joblib.dump(search.best_estimator_, BEST_MODEL_PATH)
    print(f"Tuned best model saved -> {BEST_MODEL_PATH}")

    tuning_results = {
        "model_name": model_name,
        "best_cv_f1_macro": float(search.best_score_),
        "best_params": search.best_params_,
        "cv_folds": CV_FOLDS,
        "n_iter": N_ITER,
    }
    save_json(tuning_results, TUNING_RESULTS_PATH)
    print(f"Tuning results saved -> {TUNING_RESULTS_PATH}")
    print("\nNext step: python -m src.models.evaluate")

    return search


if __name__ == "__main__":
    tune_best_model()