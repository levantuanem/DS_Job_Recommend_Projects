import time

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.models.utils import (
    get_train_test_data,
    compute_metrics,
    save_json,
    build_logistic_pipeline,
    MODELS_DIR,
    MODEL_COMPARISON_PATH,
    BEST_MODEL_INFO_PATH,
)

RANDOM_STATE = 42
CV_FOLDS = 3


def build_candidate_models(x_reference, apply_feature_selection=True, k=1000):

    return {
        "logistic_regression_baseline": build_logistic_pipeline(
            x_reference,
            random_state=RANDOM_STATE,
            apply_feature_selection=apply_feature_selection,
            k=k,
        ),
    }


def print_oversampling_summary(x_train, y_train, label_encoder):
    train_labels = label_encoder.inverse_transform(y_train)
    before_counts = pd.Series(train_labels).value_counts().sort_index()
    row_ids = np.arange(len(train_labels)).reshape(-1, 1)
    sampled_row_ids, sampled_labels = RandomOverSampler(
        random_state=RANDOM_STATE
    ).fit_resample(row_ids, train_labels)
    after_counts = pd.Series(sampled_labels).value_counts().sort_index()

    print("\nTraining class distribution before oversampling:")
    print(before_counts.to_string())
    print("\nTraining class distribution after oversampling:")
    print(after_counts.to_string())

    sample_positions = sampled_row_ids[-5:, 0]
    sample = x_train.iloc[sample_positions].copy()
    sample["formatted_experience_level"] = sampled_labels[-5:]
    sample_columns = [
        column
        for column in [
            "company_name",
            "title_length",
            "description_word_count",
            "formatted_experience_level",
        ]
        if column in sample.columns
    ]
    print("\nSample rows after oversampling (training data only):")
    print(sample[sample_columns].to_string(index=False))


def train_all_models(apply_feature_selection=True, k=1000):
    x_train, x_test, y_train, y_test, label_encoder = get_train_test_data(
        apply_feature_selection=apply_feature_selection, k=k
    )
    print(f"x_train: {x_train.shape}, x_test: {x_test.shape}")
    print(f"Classes ({len(label_encoder.classes_)}): {list(label_encoder.classes_)}")
    print_oversampling_summary(x_train, y_train, label_encoder)

    models = build_candidate_models(
        x_train,
        apply_feature_selection=apply_feature_selection,
        k=k,
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    results = []

    for name, model in models.items():
        print(f"\n=== Training: {name} ===")
        start = time.time()

        # Cross Validation (TRAIN only, never touches x_test) ----
        cv_scores = cross_val_score(
            model, x_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1
        )

        # ---- Fit on the full training set ----
        model.fit(x_train, y_train)

        train_time = time.time() - start

        # Bias & Variance: compare train vs test performance ----
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

    #  Model comparison table (1 row - Logistic Regression only) ----
    comparison_df = pd.DataFrame(results).set_index("model_name")
    comparison_df = comparison_df.sort_values("cv_f1_macro_mean", ascending=False)
    comparison_df.to_csv(MODEL_COMPARISON_PATH)
    print(f"\nModel comparison table saved -> {MODEL_COMPARISON_PATH}")
    print(comparison_df[["cv_f1_macro_mean", "test_f1_macro", "test_accuracy", "training_time_sec"]])

    # ---- Hand the (only) candidate off to tune.py ----
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
    train_all_models(apply_feature_selection=True, k=1000)
