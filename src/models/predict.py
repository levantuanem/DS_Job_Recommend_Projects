import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.models.utils import (
    ROOT_DIR,
    REPORTS_DIR,
    BEST_MODEL_PATH,
    BEST_MODEL_INFO_PATH,
    PREPROCESSOR_PATH,
    FEATURE_SELECTOR_PATH,
    LABEL_ENCODER_PATH,
)

from src.features.text_features import add_text_features
from src.features.skill_extraction import extract_skill_features
from src.features.temporal_features import add_temporal_features

# Same columns Member 3 drops in build_features() right before fitting the preprocessor
COLUMNS_TO_DROP = [
    "job_id", "company_id", "zip_code", "fips", "salary_id",
    "original_listed_time", "listed_time", "expiry", "closed_time",
    "job_posting_url", "application_url",
    "title", "description", "skills_desc",
]

TARGET_COLUMN = "formatted_experience_level"

INPUT_ALIASES = {
    "id": "job_id",
    "company": "company_name",
    "snippet": "description",
    "salary_min_usd": "min_salary",
    "salary_max_usd": "max_salary",
    "salary_avg_usd": "med_salary",
    "apply_type": "application_type",
    "remote": "remote_allowed",
    "url": "job_posting_url",
}

REQUIRED_INPUT_COLUMNS = [
    "title", "description", "skills_desc", "listed_time", "max_salary",
    "med_salary", "min_salary", "normalized_salary", "applies", "views",
    "company_name", "location", "pay_period", "formatted_work_type",
    "posting_domain", "application_type", "work_type", "currency",
    "compensation_type", "remote_allowed", "sponsored",
]

NUMERIC_INPUT_COLUMNS = {
    "max_salary", "med_salary", "min_salary", "normalized_salary",
    "applies", "views", "remote_allowed", "sponsored",
}


def normalize_input_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Map supported raw input variants to the training schema."""
    df = df.copy()

    for source, target in INPUT_ALIASES.items():
        if target not in df.columns and source in df.columns:
            df[target] = df[source]

    if "formatted_work_type" not in df.columns and "work_type" in df.columns:
        df["formatted_work_type"] = df["work_type"]

    if "listed_time" not in df.columns and "listed_at" in df.columns:
        numeric_time = pd.to_numeric(df["listed_at"], errors="coerce")
        parsed_time = pd.to_datetime(df["listed_at"], errors="coerce", utc=True)
        parsed_millis = parsed_time.astype("int64") // 1_000_000
        df["listed_time"] = numeric_time.where(
            numeric_time.notna(), parsed_millis.where(parsed_time.notna())
        )

    for column in ("remote_allowed", "sponsored"):
        if column in df.columns:
            boolean_values = (
                df[column]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({"yes": 1, "no": 0, "true": 1, "false": 0})
            )
            numeric_values = pd.to_numeric(df[column], errors="coerce")
            df[column] = boolean_values.fillna(numeric_values)

    for column in REQUIRED_INPUT_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan

    return df


def prepare_new_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the exact same feature-engineering pipeline Member 3 used in build_features()."""
    df = normalize_input_schema(df)

    if TARGET_COLUMN in df.columns:
        df = df.drop(columns=[TARGET_COLUMN])

    df = add_text_features(df)          
    df = extract_skill_features(df)     
    df = add_temporal_features(df)     

    df = df.drop(columns=[c for c in COLUMNS_TO_DROP if c in df.columns])
    return df


def predict(input_path: str, output_path: str):
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{BEST_MODEL_PATH} not found. Run train.py then tune.py first:\n"
            f"  python -m src.models.train\n  python -m src.models.tune"
        )
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"{PREPROCESSOR_PATH} not found. This is saved by Member 3's build_features() - "
            f"run train.py first (it triggers build_features())."
        )
    if not LABEL_ENCODER_PATH.exists():
        raise FileNotFoundError(f"{LABEL_ENCODER_PATH} not found. Run train.py first.")
    if not BEST_MODEL_INFO_PATH.exists():
        raise FileNotFoundError(
            f"{BEST_MODEL_INFO_PATH} not found. Run train.py and tune.py first."
        )

    model = joblib.load(BEST_MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)   
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    with open(BEST_MODEL_INFO_PATH, "r", encoding="utf-8") as file:
        model_info = json.load(file)
    use_feature_selection = model_info.get("apply_feature_selection", False)
    if use_feature_selection and not FEATURE_SELECTOR_PATH.exists():
        raise FileNotFoundError(
            f"{FEATURE_SELECTOR_PATH} not found, but the model was trained with feature selection."
        )
    selector = joblib.load(FEATURE_SELECTOR_PATH) if use_feature_selection else None

    raw_df = pd.read_csv(input_path, low_memory=False)
    print(f"Loaded {len(raw_df)} new records from {input_path}")

    x_new = prepare_new_data(raw_df)

    # transform ONLY - never fit on new/prediction data
    x_new_transformed = preprocessor.transform(x_new)
    if selector is not None:
        x_new_transformed = selector.transform(x_new_transformed)

    expected_features = getattr(model, "n_features_in_", None)
    if expected_features is not None and x_new_transformed.shape[1] != expected_features:
        raise ValueError(
            f"Prediction feature mismatch: model expects {expected_features}, "
            f"received {x_new_transformed.shape[1]}. Rebuild model artifacts together."
        )

    pred_encoded = model.predict(x_new_transformed)
    pred_labels = label_encoder.inverse_transform(pred_encoded)

    result = raw_df.copy()
    result["predicted_job_level"] = pred_labels

    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(x_new_transformed)
        except (AttributeError, ValueError) as error:
            print(
                "Probability output skipped because the saved model is incompatible "
                f"with the installed scikit-learn version: {error}"
            )
        else:
            for i, class_name in enumerate(label_encoder.classes_):
                result[f"proba_{class_name}"] = proba[:, i]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Predictions saved -> {output_path}")
    print("\nPrediction distribution:")
    print(result["predicted_job_level"].value_counts())

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict job level for new job postings.")
    parser.add_argument(
        "--input", type=str, default=str(ROOT_DIR / "data" / "raw" / "new_postings.csv"),
        help="Path to a CSV of new job postings (same raw columns as postings_clean.csv, minus the target).",
    )
    parser.add_argument(
        "--output", type=str, default=str(REPORTS_DIR / "predictions.csv"),
        help="Where to save predictions.",
    )
    args = parser.parse_args()
    predict(args.input, args.output)
