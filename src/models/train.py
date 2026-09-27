"""


 Train baseline + candidate models.


"""

import time

import joblib
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight

from src.models.utils import (
    get_train_test_data,
    compute_metrics,
    save_json,
    MODELS_DIR,
    MODEL_COMPARISON_PATH,
    BEST_MODEL_INFO_PATH,
)

# Optional boosting library - training still works fine without it.
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

RANDOM_STATE = 42
CV_FOLDS = 5


def build_candidate_models():
    """
    README 9.1 Baseline Model: Logistic Regression
    README 9.2 Candidate Models: RandomForest, XGBoost
    README 9.4 Class Imbalance: class_weight="balanced" wherever supported
    """
    models = {
        "logistic_regression_baseline": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    if HAS_XGB:
        models["xgboost"] = XGBClassifier(
            n_estimators=150,       
            tree_method="hist",     
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    else:
        print("[info] xgboost not installed - skipping (pip install xgboost to enable).")

    return models


def train_all_models(apply_feature_selection=False, k=1000):
    x_train, x_test, y_train, y_test, label_encoder = get_train_test_data(
        apply_feature_selection=apply_feature_selection, k=k
    )
    print(f"x_train: {x_train.shape}, x_test: {x_test.shape}")
    print(f"Classes ({len(label_encoder.classes_)}): {list(label_encoder.classes_)}")

    models = build_candidate_models()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # Sample weights for XGBoost, which doesn't accept class_weight="balanced" directly.
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    results = []

    for name, model in models.items():
        print(f"\n=== Training: {name} ===")
        start = time.time()

        # ---- README 9.5 Cross Validation (TRAIN only, never touches x_test) ----
        cv_scores = cross_val_score(
            model, x_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1
        )

        # ---- Fit on the full training set ----
        if name == "xgboost":
            model.fit(x_train, y_train, sample_weight=sample_weight)
        else:
            model.fit(x_train, y_train)

        train_time = time.time() - start

        # ---- README 9.7 / 9.8 Bias & Variance: compare train vs test performance ----
        train_pred = model.predict(x_train)
        test_pred = model.predict(x_test)

        metrics = {"model_name": name}
        metrics.update(compute_metrics(y_train, train_pred, prefix="train_"))
        metrics.update(compute_metrics(y_test, test_pred, prefix="test_"))
        metrics["cv_f1_macro_mean"] = cv_scores.mean()
        metrics["cv_f1_macro_std"] = cv_scores.std()
        metrics["training_time_sec"] = train_time

        results.append(metrics)

        model_path = MODELS_DIR / f"{name}.pkl"
        joblib.dump(model, model_path)
        print(f"Saved model -> {model_path}")
        print(
            f"CV Macro-F1: {metrics['cv_f1_macro_mean']:.4f} (+/- {metrics['cv_f1_macro_std']:.4f}) | "
            f"Test Macro-F1: {metrics['test_f1_macro']:.4f} | Test Acc: {metrics['test_accuracy']:.4f}"
        )

    # ---- README 9.10 Model Comparison table ----
    comparison_df = pd.DataFrame(results).set_index("model_name")
    comparison_df = comparison_df.sort_values("cv_f1_macro_mean", ascending=False)
    comparison_df.to_csv(MODEL_COMPARISON_PATH)
    print(f"\nModel comparison table saved -> {MODEL_COMPARISON_PATH}")
    print(comparison_df[["cv_f1_macro_mean", "test_f1_macro", "test_accuracy", "training_time_sec"]])

    # ---- Pick the candidate with the best CV Macro-F1 to hand off to tune.py ----
    best_name = comparison_df.index[0]
    best_info = {
        "best_model_name": best_name,
        "cv_f1_macro_mean": float(comparison_df.loc[best_name, "cv_f1_macro_mean"]),
        "test_f1_macro": float(comparison_df.loc[best_name, "test_f1_macro"]),
        "apply_feature_selection": apply_feature_selection,
        "k": k,
    }
    save_json(best_info, BEST_MODEL_INFO_PATH)
    print(f"\nBest candidate before tuning: {best_name} -> saved to {BEST_MODEL_INFO_PATH}")
    print("Next step: python -m src.models.tune")

    return comparison_df


if __name__ == "__main__":
    train_all_models(apply_feature_selection=False, k=1000)
