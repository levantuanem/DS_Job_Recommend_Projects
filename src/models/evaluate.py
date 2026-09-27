"""
src/models/evaluate.py

Member 4 - STEP 3: Evaluate the tuned best model (models/best_model.pkl,
produced by tune.py) on train vs test.

Outputs (all under reports/):
    evaluation_report.json   - all metrics + bias/variance diagnosis + classification report
    confusion_matrix.csv/png - confusion matrix on the test set
    feature_importance.csv   - feature_importances_ / coef_ for model interpretation (9.11)

Run (from the repository root):
    python -m src.models.evaluate
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import joblib

from src.models.utils import (
    get_train_test_data,
    compute_metrics,
    save_json,
    BEST_MODEL_PATH,
    REPORTS_DIR,
)


def evaluate_best_model():
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{BEST_MODEL_PATH} not found. Run tune.py first: python -m src.models.tune"
        )

    model = joblib.load(BEST_MODEL_PATH)
    x_train, x_test, y_train, y_test, label_encoder = get_train_test_data()

    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)

    # ---- README 9.7 / 9.8 Bias & Variance ----
    train_metrics = compute_metrics(y_train, train_pred, prefix="train_")
    test_metrics = compute_metrics(y_test, test_pred, prefix="test_")

    print("=== Training performance ===")
    for k, v in train_metrics.items():
        print(f"{k}: {v:.4f}")

    print("\n=== Test performance ===")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")

    gap = train_metrics["train_f1_macro"] - test_metrics["test_f1_macro"]
    if train_metrics["train_f1_macro"] < 0.6 and test_metrics["test_f1_macro"] < 0.6:
        diagnosis = "High bias (underfitting): both train and test Macro-F1 are low."
    elif gap > 0.15:
        diagnosis = "High variance (overfitting): large gap between train and test Macro-F1."
    else:
        diagnosis = "Bias/variance look reasonably balanced."
    print(f"\nDiagnosis: {diagnosis}")

    # ---- README 9.9 Full evaluation on the test set ----
    class_names = [str(c) for c in label_encoder.classes_]
    report = classification_report(
        y_test, test_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    print("\n=== Classification report (test) ===")
    print(classification_report(y_test, test_pred, target_names=class_names, zero_division=0))

    cm = confusion_matrix(y_test, test_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(REPORTS_DIR / "confusion_matrix.csv")

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (Test Set)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    print(f"\nConfusion matrix saved -> {REPORTS_DIR / 'confusion_matrix.png'}")

    # ---- README 9.11 Model interpretation ----
    importance_path = REPORTS_DIR / "feature_importance.csv"
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            imp_df = pd.DataFrame({
                "feature_index": np.arange(len(importances)),
                "importance": importances,
            }).sort_values("importance", ascending=False)
            imp_df.to_csv(importance_path, index=False)
            print(
                f"Feature importances saved -> {importance_path} "
                f"(indices map to the ColumnTransformer's transformed columns; "
                f"use preprocessor.get_feature_names_out() from Member 3's preprocessor.pkl "
                f"to map indices back to real feature names)."
            )
        elif hasattr(model, "coef_"):
            coef_df = pd.DataFrame(
                model.coef_, columns=[f"feat_{i}" for i in range(model.coef_.shape[1])]
            )
            n_rows = coef_df.shape[0]
            coef_df.insert(0, "class", class_names[:n_rows] if n_rows > 1 else ["positive_class"])
            coef_df.to_csv(importance_path, index=False)
            print(f"Model coefficients saved -> {importance_path}")
        else:
            print("Selected model has no feature_importances_ / coef_ attribute; skipping interpretation export.")
    except Exception as e:
        print(f"Could not export feature importance: {e}")

    # ---- Save full evaluation summary ----
    summary = {
        **train_metrics,
        **test_metrics,
        "diagnosis": diagnosis,
        "classification_report": report,
    }
    save_json(summary, REPORTS_DIR / "evaluation_report.json")
    print(f"\nFull evaluation report saved -> {REPORTS_DIR / 'evaluation_report.json'}")
    print("\nNext step: python -m src.models.predict --input <new_data.csv>")

    return summary


if __name__ == "__main__":
    evaluate_best_model()
