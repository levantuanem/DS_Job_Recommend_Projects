"""
lazy_predict_runner.py
-----------------------
Buoc 1 cua Member 4: dung LazyPredict de nhanh chong so sanh hang chuc model
Machine Learning tren cung mot bo du lieu, tu do chon ra top model tiem nang
de di sau vao Train / Tune / Evaluate.



Luu y: LazyPredict train "mac dinh" (khong tune) rat nhieu model cung luc,
nen chi dung o buoc KHAM PHA, khong dung lam ket qua cuoi cung.
"""

import os
import scipy.sparse as sp
import matplotlib.pyplot as plt
from lazypredict.Supervised import LazyClassifier

from src.models.config import REPORTS_DIR, RANDOM_STATE

# Neu so chieu (so cot feature) vuot nguong nay, KHONG convert sparse -> dense
# (tranh tran RAM khi co TF-IDF hang nghin chieu). LazyPredict se tu bo qua
# nhung model khong ho tro sparse input.
DENSE_CONVERT_MAX_FEATURES = 2000


def _maybe_densify(X, name="X"):
    """
    LazyPredict/mot so model trong do hoat dong tot hon voi dense array.
    Chi convert neu so chieu du nho, tranh MemoryError voi TF-IDF lon.
    """
    if sp.issparse(X):
        n_features = X.shape[1]
        if n_features <= DENSE_CONVERT_MAX_FEATURES:
            print(f"  {name} la sparse ({X.shape}), convert sang dense "
                  f"(<= {DENSE_CONVERT_MAX_FEATURES} chieu).")
            return X.toarray()
        else:
            print(f"  CANH BAO: {name} la sparse voi {n_features} chieu "
                  f"(> {DENSE_CONVERT_MAX_FEATURES}). Giu nguyen dang sparse - "
                  "mot so model trong LazyPredict co the bi bo qua/loi, "
                  "day la hanh vi binh thuong.")
    return X


def run_lazy_predict(X_train, X_test, y_train, y_test, top_n: int = 10):
    """
    Chay LazyClassifier va tra ve bang xep hang model theo Accuracy / F1 / thoi gian.
    """
    X_train = _maybe_densify(X_train, "X_train")
    X_test = _maybe_densify(X_test, "X_test")

    clf = LazyClassifier(
        verbose=0,
        ignore_warnings=True,
        custom_metric=None,
        predictions=False,
    )

    models_df, predictions = clf.fit(X_train, X_test, y_train, y_test)

    # Sap xep theo F1 Score giam dan (uu tien F1 vi du lieu mat can bang class)
    if "F1 Score" in models_df.columns:
        models_df = models_df.sort_values(by="F1 Score", ascending=False)

    print("\n=== KET QUA LAZYPREDICT (TOP {}) ===".format(top_n))
    print(models_df.head(top_n))

    _plot_lazy_predict_results(models_df.head(top_n))

    csv_path = os.path.join(REPORTS_DIR, "lazy_predict_results.csv")
    models_df.to_csv(csv_path)
    print(f"\nDa luu ket qua LazyPredict vao: {csv_path}")

    return models_df


def _plot_lazy_predict_results(top_models_df):
    """
    Ve subplot so sanh Accuracy, F1 Score va Thoi gian train cua top model tu LazyPredict.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    metrics = [
        ("Accuracy", "Accuracy"),
        ("F1 Score", "F1 Score (Weighted)"),
        ("Time Taken", "Thoi gian train (s)"),
    ]

    for ax, (col, title) in zip(axes, metrics):
        if col not in top_models_df.columns:
            ax.axis("off")
            continue
        data = top_models_df[col].sort_values()
        ax.barh(data.index.astype(str), data.values, color="#4C72B0")
        ax.set_title(title)
        ax.set_xlabel(col)

    plt.suptitle("So sanh nhanh cac model bang LazyPredict", fontsize=14)
    plt.tight_layout()

    out_path = os.path.join(REPORTS_DIR, "lazy_predict_comparison.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Da luu bieu do LazyPredict vao: {out_path}")


def select_top_model_names(models_df, top_n: int = 5):
    """
    Lay danh sach ten model tiem nang nhat (theo F1 Score) de dua vao buoc train chinh thuc.
    """
    if "F1 Score" in models_df.columns:
        top = models_df.sort_values(by="F1 Score", ascending=False).head(top_n)
    else:
        top = models_df.head(top_n)
    return list(top.index)