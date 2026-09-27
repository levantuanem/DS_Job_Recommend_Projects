import argparse

import joblib
import pandas as pd

from src.models.utils import (
    ROOT_DIR,
    REPORTS_DIR,
    BEST_MODEL_PATH,
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


def prepare_new_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the exact same feature-engineering pipeline Member 3 used in build_features()."""
    df = df.copy()

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

    model = joblib.load(BEST_MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)   
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    selector = joblib.load(FEATURE_SELECTOR_PATH) if FEATURE_SELECTOR_PATH.exists() else None

    raw_df = pd.read_csv(input_path, low_memory=False)
    print(f"Loaded {len(raw_df)} new records from {input_path}")

    x_new = prepare_new_data(raw_df)

    # transform ONLY - never fit on new/prediction data
    x_new_transformed = preprocessor.transform(x_new)
    if selector is not None:
        x_new_transformed = selector.transform(x_new_transformed)

    pred_encoded = model.predict(x_new_transformed)
    pred_labels = label_encoder.inverse_transform(pred_encoded)

    result = raw_df.copy()
    result["predicted_job_level"] = pred_labels

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_new_transformed)
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
