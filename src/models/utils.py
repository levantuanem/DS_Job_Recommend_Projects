

import sys
import json
from pathlib import Path

import joblib
import yaml
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ---------------------------------------------------------------------------
# Make sure the repo root is importable ("from src.features... import ...")
# regardless of the working directory the script is launched from.
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.features.build_features import build_features  # Member 3's function

# =========================
# PATHS
# =========================
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
CONFIG_PATH = ROOT_DIR / "configs" / "config.yaml"
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


def load_model_config():
    """Load project config with model.cv_folds and model.k_fold overrides."""
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return config.get("model", {})


def get_model_setting(name, default):
    value = load_model_config().get(name, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def get_train_test_data(apply_feature_selection=False, k=1000, force_rebuild=False):
    """
    Returns (x_train, x_test, y_train_enc, y_test_enc, label_encoder).

    First call: runs Member 3's build_features() (fits the preprocessor and,
    optionally, the SelectKBest selector on TRAIN only) and caches the resulting
    arrays + a LabelEncoder fit on the training labels. All later calls
    (tune.py, evaluate.py) reuse this exact cached split so nothing is
    re-fit on test data.

    Pass force_rebuild=True (or delete models/data_split.pkl) to regenerate
    with different `apply_feature_selection` / `k` settings.
    """
    if DATA_SPLIT_PATH.exists() and not force_rebuild:
        cache = joblib.load(DATA_SPLIT_PATH)
        cache_matches_config = (
            cache.get("apply_feature_selection") == apply_feature_selection
            and cache.get("k") == k
        )
        if cache_matches_config:
            print(f"Loading cached train/test split from {DATA_SPLIT_PATH}")
            return (
                cache["x_train"],
                cache["x_test"],
                cache["y_train"],
                cache["y_test"],
                cache["label_encoder"],
            )
        print("Cached split configuration differs; rebuilding features.")

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
            "apply_feature_selection": apply_feature_selection,
            "k": k,
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
