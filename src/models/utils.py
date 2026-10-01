import sys
import json
from pathlib import Path

import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# imbalanced-learn: data-level class imbalance handling (random oversampling),
# used TOGETHER with class_weight="balanced" (model-level handling) per team request.
# pip install imbalanced-learn
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.features.build_features import build_features  

# =========================
# PATHS
# =========================
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_SPLIT_PATH = MODELS_DIR / "data_split.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"          
FEATURE_SELECTOR_PATH = MODELS_DIR / "feature_selector.pkl"  
MODEL_COMPARISON_PATH = MODELS_DIR / "model_comparison.csv"
BEST_MODEL_INFO_PATH = MODELS_DIR / "best_model_info.json"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"            
TUNING_RESULTS_PATH = MODELS_DIR / "tuning_results.json"


def get_train_test_data(apply_feature_selection=True, k=1000, force_rebuild=False):

    if DATA_SPLIT_PATH.exists() and not force_rebuild:
        print(f"Loading cached train/test split from {DATA_SPLIT_PATH}")
        cache = joblib.load(DATA_SPLIT_PATH)
        return (
            cache["x_train"],
            cache["x_test"],
            cache["y_train"],
            cache["y_test"],
            cache["label_encoder"],
        )

    print("No cached split found. Building features via Member 3's pipeline (build_features)...")
    x_train, x_test, y_train, y_test = build_features(
        apply_feature_selection=apply_feature_selection, k=k
    )

    # Fit label encoder on TRAIN labels only, then just transform test (no leakage)
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)

    joblib.dump(
        {
            "x_train": x_train,
            "x_test": x_test,
            "y_train": y_train_enc,
            "y_test": y_test_enc,
            "label_encoder": label_encoder,
        },
        DATA_SPLIT_PATH,
    )
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    print(f"Cached train/test split -> {DATA_SPLIT_PATH}")

    return x_train, x_test, y_train_enc, y_test_enc, label_encoder


def compute_metrics(y_true, y_pred, prefix=""):
    """Accuracy / Precision / Recall / Macro-F1 / Weighted-F1 (see README 9.9)."""
    return {
        f"{prefix}accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        f"{prefix}recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        f"{prefix}f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        f"{prefix}f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def build_logistic_pipeline(random_state=42, **logreg_kwargs):

    params = dict(max_iter=2000, class_weight="balanced", random_state=random_state)
    params.update(logreg_kwargs)
    return ImbPipeline([
        ("oversample", RandomOverSampler(random_state=random_state)),
        ("clf", LogisticRegression(**params)),
    ])
