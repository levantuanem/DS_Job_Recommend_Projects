"""

Gom:
- Tinh day du metric: Accuracy, Precision, Recall, F1, Macro-F1, Weighted-F1
- Confusion Matrix
- Bang so sanh nhieu model (Model Comparison Table)
- Phan tich Bias / Variance: so sanh Train vs Validation vs Test performance
- Ve TOAN BO ket qua danh gia duoi dang SUBPLOT (1 hinh, nhieu o) de de doc, de bao cao
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

from src.models.config import REPORTS_DIR, CLASS_LABELS


def compute_metrics(y_true, y_pred, average_list=("macro", "weighted")):
    """
    Tinh cac metric chinh theo muc 9.9. Khong chi dung Accuracy.
    """
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
    }
    for avg in average_list:
        metrics[f"Precision ({avg})"] = precision_score(y_true, y_pred, average=avg, zero_division=0)
        metrics[f"Recall ({avg})"] = recall_score(y_true, y_pred, average=avg, zero_division=0)
        metrics[f"F1 ({avg})"] = f1_score(y_true, y_pred, average=avg, zero_division=0)
    return metrics


def build_model_comparison_table(model_results: dict):
    """
    model_results: dict dang
        {
          "Random Forest": {
              "y_test": y_test, "y_pred": y_pred,
              "cv_mean": 0.81, "cv_std": 0.02, "train_time": 3.2,
          },
          ...
        }
    Tra ve DataFrame giong bang trong muc 9.10 cua README.
    """
    rows = []
    for name, res in model_results.items():
        m = compute_metrics(res["y_test"], res["y_pred"])
        rows.append({
            "Model": name,
            "Accuracy": round(m["Accuracy"], 4),
            "Precision (weighted)": round(m["Precision (weighted)"], 4),
            "Recall (weighted)": round(m["Recall (weighted)"], 4),
            "Macro-F1": round(m["F1 (macro)"], 4),
            "Weighted-F1": round(m["F1 (weighted)"], 4),
            "Training Time (s)": res.get("train_time", np.nan),
            "CV Mean": res.get("cv_mean", np.nan),
            "CV Std": res.get("cv_std", np.nan),
        })

    df = pd.DataFrame(rows).sort_values(by="Macro-F1", ascending=False).reset_index(drop=True)
    csv_path = os.path.join(REPORTS_DIR, "model_comparison_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"Da luu bang so sanh model vao: {csv_path}")
    return df


def bias_variance_scores(model, X_train, y_train, X_val, y_val, X_test, y_test, scoring_fn=f1_score):
    """
    Muc 9.7 & 9.8: So sanh Train / Validation / Test performance de phat hien
    Underfitting (High Bias) hay Overfitting (High Variance).
    """
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    train_score = scoring_fn(y_train, train_pred, average="macro", zero_division=0)
    val_score = scoring_fn(y_val, val_pred, average="macro", zero_division=0)
    test_score = scoring_fn(y_test, test_pred, average="macro", zero_division=0)

    diagnosis = _diagnose_bias_variance(train_score, val_score, test_score)

    return {
        "Train Macro-F1": round(train_score, 4),
        "Validation Macro-F1": round(val_score, 4),
        "Test Macro-F1": round(test_score, 4),
        "Diagnosis": diagnosis,
    }


def _diagnose_bias_variance(train_score, val_score, test_score, gap_threshold=0.08, low_threshold=0.55):
    """
    Quy tac don gian dua theo muc 9.7/9.8 README:
    - Train thap, Val thap, Test thap  -> High Bias (Underfitting)
    - Train rat cao, Val/Test thap hon nhieu -> High Variance (Overfitting)
    - Nguoc lai -> On dinh, khong dau hieu ro ret
    """
    if train_score < low_threshold and val_score < low_threshold:
        return "High Bias (Underfitting)"
    if (train_score - val_score) > gap_threshold and (train_score - test_score) > gap_threshold:
        return "High Variance (Overfitting)"
    return "On dinh (Good fit)"


def plot_evaluation_subplots(model_comparison_df, best_model_name, y_true, y_pred,
                              bias_variance_dict, class_labels=CLASS_LABELS,
                              save_name="evaluation_summary.png"):
    """
    Ve TAT CA ket qua danh gia trong MOT figure duy nhat gom 4 subplot:
        (1) So sanh Macro-F1 / Weighted-F1 / Accuracy giua cac model
        (2) Confusion Matrix cua model tot nhat
        (3) Bieu do Train vs Validation vs Test (Bias/Variance) cua model tot nhat
        (4) So sanh CV Mean +/- CV Std giua cac model (do on dinh)
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ---- (1) So sanh metric giua cac model ----
    ax1 = axes[0, 0]
    plot_df = model_comparison_df.set_index("Model")[["Accuracy", "Macro-F1", "Weighted-F1"]]
    plot_df.plot(kind="bar", ax=ax1, color=["#4C72B0", "#DD8452", "#55A868"])
    ax1.set_title("So sanh metric giua cac model")
    ax1.set_ylabel("Score")
    ax1.set_ylim(0, 1)
    ax1.legend(loc="lower right")
    ax1.tick_params(axis="x", rotation=30)

    # ---- (2) Confusion Matrix cua model tot nhat ----
    ax2 = axes[0, 1]
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_labels, yticklabels=class_labels, ax=ax2,
    )
    ax2.set_title(f"Confusion Matrix - {best_model_name}")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    ax2.tick_params(axis="x", rotation=45)
    ax2.tick_params(axis="y", rotation=0)

    # ---- (3) Bias / Variance: Train vs Val vs Test ----
    ax3 = axes[1, 0]
    stages = ["Train", "Validation", "Test"]
    scores = [
        bias_variance_dict["Train Macro-F1"],
        bias_variance_dict["Validation Macro-F1"],
        bias_variance_dict["Test Macro-F1"],
    ]
    bars = ax3.bar(stages, scores, color=["#4C72B0", "#DD8452", "#55A868"])
    ax3.set_ylim(0, 1)
    ax3.set_title(f"Bias/Variance - {best_model_name}\n"
                   f"Chan doan: {bias_variance_dict['Diagnosis']}")
    ax3.set_ylabel("Macro-F1")
    for bar, score in zip(bars, scores):
        ax3.text(bar.get_x() + bar.get_width() / 2, score + 0.02,
                  f"{score:.3f}", ha="center")

    # ---- (4) CV Mean +/- CV Std giua cac model ----
    ax4 = axes[1, 1]
    cv_df = model_comparison_df.dropna(subset=["CV Mean"]).sort_values("CV Mean", ascending=True)
    ax4.barh(
        cv_df["Model"], cv_df["CV Mean"],
        xerr=cv_df["CV Std"], color="#8172B2", capsize=4,
    )
    ax4.set_title("Do on dinh cua model (CV Mean +/- CV Std)")
    ax4.set_xlabel("f1_macro (Cross Validation)")
    ax4.set_xlim(0, 1)

    plt.suptitle("Model Evaluation Summary - Member 4", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = os.path.join(REPORTS_DIR, save_name)
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"\nDa luu bieu do tong hop danh gia vao: {out_path}")

    return fig


def print_classification_report(y_true, y_pred, class_labels=CLASS_LABELS):
    print(classification_report(y_true, y_pred, target_names=class_labels, zero_division=0))