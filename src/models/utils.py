import hashlib
import sys
import json
from pathlib import Path

import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Class imbalance is handled with LogisticRegression(class_weight="balanced").
from sklearn.pipeline import Pipeline


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.features.build_features import DATA_PATH, build_features, build_preprocessor

# =========================
# PATHS
# =========================
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_SPLIT_PATH = MODELS_DIR / "data_split.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
MODEL_COMPARISON_PATH = MODELS_DIR / "model_comparison.csv"
BEST_MODEL_INFO_PATH = MODELS_DIR / "best_model_info.json"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"            
TUNING_RESULTS_PATH = MODELS_DIR / "tuning_results.json"
DATA_SPLIT_VERSION = 3


def _get_split_signature():
    source_paths = [
        Path(__file__).resolve().parents[1] / "features" / filename
        for filename in (
            "build_features.py",
            "text_features.py",
            "skill_extraction.py",
            "temporal_features.py",
        )
    ]
    source_hashes = {
        str(path.name): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_paths
    }
    data_stat = DATA_PATH.stat()
    return {
        "data_size": data_stat.st_size,
        "data_mtime_ns": data_stat.st_mtime_ns,
        "feature_code": source_hashes,
    }


def get_train_test_data(force_rebuild=False, include_groups=False):
    split_signature = _get_split_signature()

    if DATA_SPLIT_PATH.exists() and not force_rebuild:
        print(f"Loading cached train/test split from {DATA_SPLIT_PATH}")
        cache = joblib.load(DATA_SPLIT_PATH)
        if (
            cache.get("version") == DATA_SPLIT_VERSION
            and cache.get("signature") == split_signature
        ):
            result = (
                cache["x_train"],
                cache["x_test"],
                cache["y_train"],
                cache["y_test"],
                cache["label_encoder"],
            )
            if include_groups:
                return (*result, cache["groups_train"])
            return result
        print("Cached split is stale or uses an old format; rebuilding it.")

    print("No cached split found. Building features via Member 3's pipeline (build_features)...")
    x_train, x_test, y_train, y_test, groups_train, groups_test = build_features(
        return_groups=True
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
            "groups_train": groups_train,
            "groups_test": groups_test,
            "version": DATA_SPLIT_VERSION,
            "signature": split_signature,
        },
        DATA_SPLIT_PATH,
    )
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    print(f"Cached train/test split -> {DATA_SPLIT_PATH}")

    result = x_train, x_test, y_train_enc, y_test_enc, label_encoder
    if include_groups:
        return (*result, groups_train)
    return result


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


class BoundedSelectKBest(SelectKBest):
    def fit(self, X, y=None):
        self.k = min(self.k, X.shape[1])
        return super().fit(X, y)


def build_logistic_pipeline(
    x_reference,
    random_state=42,
    apply_feature_selection=False,
    k=1000,
    **logreg_kwargs,
):

    params = dict(max_iter=2000, class_weight="balanced", random_state=random_state)
    params.update(logreg_kwargs)
    steps = [("preprocessor", build_preprocessor(x_reference))]
    if apply_feature_selection:
        steps.append(("selector", BoundedSelectKBest(score_func=f_classif, k=k)))
    steps.append(("clf", LogisticRegression(**params)))
    return Pipeline(steps)
