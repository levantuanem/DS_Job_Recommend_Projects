
import os
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.models.config import MODELS_DIR, RANDOM_STATE, VAL_SIZE

# ============================================================
# IMPORT 
# ============================================================
try:
    from src.features.build_features import build_features
    HAS_BUILD_FEATURES = True
except ImportError:
    HAS_BUILD_FEATURES = False


def _maybe_encode_target(y_train, y_test, save_artifact: bool = True):
    """
    build_features() tra ve y dang pd.Series - co the la text (vd 'Entry Level')
    hoac da la so. Ham nay tu dong phat hien va encode neu can, roi luu
    LabelEncoder lam artifact (muc 9.12 README).
    """
    y_train = pd.Series(y_train).reset_index(drop=True)
    y_test = pd.Series(y_test).reset_index(drop=True)

    is_numeric = pd.api.types.is_numeric_dtype(y_train)

    if is_numeric:
        # Da la so san -> khong can encode, nhung van tao label_mapping "gia"
        classes = sorted(y_train.unique())
        label_mapping = {str(c): c for c in classes}
        return y_train.values, y_test.values, None, label_mapping

    le = LabelEncoder()
    le.fit(pd.concat([y_train, y_test], axis=0))
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)

    if save_artifact:
        joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.pkl"))

    label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    return y_train_enc, y_test_enc, le, label_mapping


def _split_out_validation(X_train, y_train):
    """
    build_features() chi tra ve Train/Test. Member 4 can them Validation set
    de phan tich Bias/Variance (Train vs Val vs Test) theo muc 9.7-9.8.
    Ham nay tach mot phan tu X_train, ho tro ca DataFrame, ndarray va sparse matrix.
    """
    n = X_train.shape[0]
    idx = np.arange(n)

    idx_train, idx_val = train_test_split(
        idx, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_train,
    )

    def _select(X, indices):
        if sp.issparse(X):
            return X[indices]
        if isinstance(X, (pd.DataFrame, pd.Series)):
            return X.iloc[indices]
        return np.asarray(X)[indices]

    X_tr = _select(X_train, idx_train)
    X_val = _select(X_train, idx_val)
    y_tr = np.asarray(y_train)[idx_train]
    y_val = np.asarray(y_train)[idx_val]

    return X_tr, X_val, y_tr, y_val


def load_and_split(apply_feature_selection: bool = True, k: int = 1000):
    """
    Ham chinh: goi build_features() that cua Member 3, tach them Validation,
    va encode target neu can. Mac dinh BAT feature selection (k=1000) de
    giam so chieu du lieu (rat huu ich khi co TF-IDF + One-Hot, giup
    LazyPredict/Train/Tune chay nhanh hon dang ke).

    Tra ve dict giong het cau truc cu, de cac module khac (lazy_predict_runner,
    train, tune, evaluate) khong can sua gi them.
    """
    if not HAS_BUILD_FEATURES:
        raise ImportError(
            "Khong import duoc `from src.features.build_features import build_features`.\n"
            "Hay dam bao:\n"
            "  1) Ban dang chay script tu THU MUC GOC cua repo "
            "(DS_Job_Recommend_Projects/), khong phai tu trong scripts/.\n"
            "  2) File src/features/build_features.py cua Member 3 da ton tai "
            "va co ham build_features(...).\n"
            "  3) Neu ten ham/tham so cua Member 3 khac, sua lai phan IMPORT "
            "o dau file src/models/data_loader.py nay."
        )

    print("Dang goi build_features() cua Member 3 "
          f"(apply_feature_selection={apply_feature_selection}, k={k})...")
    X_train, X_test, y_train, y_test = build_features(
        apply_feature_selection=apply_feature_selection, k=k,
    )

    # ---- Kiem tra nhanh dinh dang du lieu tra ve ----
    print(f"  Kieu X_train: {type(X_train)} | shape: {getattr(X_train, 'shape', 'N/A')}")
    print(f"  Kieu y_train: {type(y_train)} | so mau: {len(y_train)}")
    if sp.issparse(X_train):
        print("  -> X la sparse matrix (thuong gap khi co TF-IDF / One-Hot cardinality cao).")

    # ---- Encode target neu can ----
    y_train_enc, y_test_enc, label_encoder, label_mapping = _maybe_encode_target(
        y_train, y_test,
    )

    # ---- Tach them Validation tu Train ----
    X_tr, X_val, y_tr, y_val = _split_out_validation(X_train, y_train_enc)

    print("\nKich thuoc du lieu sau khi tach Validation:")
    print(f"  Train      : {X_tr.shape}")
    print(f"  Validation : {X_val.shape}")
    print(f"  Test       : {X_test.shape if hasattr(X_test, 'shape') else len(X_test)}")
    print(f"  Label mapping: {label_mapping}")

    return {
        "X_train": X_tr, "y_train": y_tr,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test_enc,
        "label_encoder": label_encoder,
        "label_mapping": label_mapping,
    }


if __name__ == "__main__":
    # Chay thu: python -m src.models.data_loader (tu thu muc goc repo)
    data = load_and_split()
    print("\nOK - da load va tach du lieu thanh cong.")