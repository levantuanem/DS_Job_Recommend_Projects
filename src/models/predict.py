import argparse
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd

from src.models.utils import (
    ROOT_DIR,
    REPORTS_DIR,
    BEST_MODEL_PATH,
    LABEL_ENCODER_PATH,
)

from src.features.text_features import add_text_features
from src.features.skill_extraction import extract_skill_features
from src.features.temporal_features import add_temporal_features


COLUMNS_TO_DROP = [
    "job_id", "company_id", "zip_code", "fips", "salary_id",
    "original_listed_time", "listed_time", "expiry", "closed_time",
    "job_posting_url", "application_url",
    "title", "description", "skills_desc",
]

NUMERIC_FEATURES = [
    "max_salary", "med_salary", "min_salary", "normalized_salary",
    "title_length", "description_length", "skills_length",
    "description_word_count", "skills_word_count", "posting_year", "posting_month",
    "posting_day", "posting_dayofweek", "posting_quarter", "posting_hour",
    "is_weekend", "skill_count",
]
CATEGORICAL_FEATURES = [
    "company_name", "location", "pay_period", "formatted_work_type", "posting_domain",
    "application_type", "work_type", "currency", "compensation_type",
]
BINARY_FEATURES = ["remote_allowed", "sponsored"]

TARGET_COLUMN = "formatted_experience_level"


def prepare_new_data(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    aliases = {
        "company": "company_name",
        "salary_min_usd": "min_salary",
        "salary_max_usd": "max_salary",
        "apply_type": "application_type",
        "listed_at": "listed_time",
    }
    for source, target in aliases.items():
        if target not in df.columns and source in df.columns:
            df[target] = df[source]

    if "formatted_work_type" not in df.columns and "work_type" in df.columns:
        df["formatted_work_type"] = df["work_type"]
    if "posting_domain" not in df.columns and "url" in df.columns:
        df["posting_domain"] = df["url"].map(
            lambda value: urlparse(value).netloc.lower() if isinstance(value, str) else "unknown"
        )
    if "remote_allowed" not in df.columns and "remote" in df.columns:
        remote_values = df["remote"].astype(str).str.strip().str.lower()
        df["remote_allowed"] = remote_values.map(
            {"yes": 1, "true": 1, "1": 1, "no": 0, "false": 0, "0": 0}
        )

    for column in NUMERIC_FEATURES:
        if column not in df.columns:
            df[column] = np.nan
    for column in CATEGORICAL_FEATURES:
        if column not in df.columns:
            df[column] = "unknown"
    for column in BINARY_FEATURES:
        if column not in df.columns:
            df[column] = np.nan
        else:
            binary_values = df[column].astype(str).str.strip().str.lower()
            normalized = binary_values.map(
                {
                    "yes": 1,
                    "true": 1,
                    "1": 1,
                    "no": 0,
                    "false": 0,
                    "0": 0,
                }
            )
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(normalized)

    if "skills_desc" not in df.columns:
        df["skills_desc"] = ""

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
    model = joblib.load(BEST_MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)

    raw_df = pd.read_csv(input_path, low_memory=False)
    print(f"Loaded {len(raw_df)} new records from {input_path}")

    x_new = prepare_new_data(raw_df)

    pred_encoded = model.predict(x_new)
    pred_labels = label_encoder.inverse_transform(pred_encoded)

    result = raw_df.copy()
    result["predicted_job_level"] = pred_labels

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_new)
        for i, class_name in enumerate(label_encoder.classes_):
            result[f"proba_{class_name}"] = proba[:, i]

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
