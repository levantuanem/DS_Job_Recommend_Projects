from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from src.features.text_features import add_text_features
from src.features.skill_extraction import extract_skill_features
from src.features.temporal_features import add_temporal_features

# =========================
# PATHS
# =========================
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "processed" / "postings_clean.csv"

# =========================
# BUILD FEATURES
# =========================
def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    numeric_features = [
        "max_salary",
        "med_salary",
        "min_salary",
        "normalized_salary",

        # Text statistics
        "title_length",
        "description_length",
        "skills_length",
        "description_word_count",
        "skills_word_count",

        # Temporal features
        "posting_year",
        "posting_month",
        "posting_day",
        "posting_dayofweek",
        "posting_quarter",
        "posting_hour",
        "is_weekend",

        # Skill count
        "skill_count"
    ]

    categorical_features = [
        "location",
        "pay_period",
        "formatted_work_type",
        "application_type",
        "work_type",
        "currency",
        "compensation_type"]

    binary_features = [
        "remote_allowed",
        "sponsored"]

    skill_features = [
        column
        for column in x.columns
        if column.startswith("skill_")
        and column != "skill_count"]
    numeric_features.extend(skill_features)

    all_structured_features = (
        numeric_features + categorical_features + binary_features)

    missing_features = [
        column
        for column in all_structured_features
        if column not in x.columns
    ]

    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")

    missing_text = [column for column in ("title", "description") if column not in x.columns]
    if missing_text:
        raise ValueError(f"Missing required text features: {missing_text}")

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    ])

    binary_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])

    title_vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.98,
        max_features=5000,
        dtype=np.float32,
    )
    description_vectorizer = TfidfVectorizer(
        stop_words="english",
        min_df=10,
        max_df=0.95,
        max_features=20000,
        dtype=np.float32,
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
            ("binary", binary_pipeline, binary_features),
            ("title_text", title_vectorizer, "title"),
            ("description_text", description_vectorizer, "description"),
        ]
    )

def _build_company_groups(data: pd.DataFrame) -> pd.Series:
    groups = pd.Series(pd.NA, index=data.index, dtype="string")

    if "company_id" in data.columns:
        company_ids = data["company_id"].astype("string").str.strip()
        company_ids = company_ids.mask(
            company_ids.str.lower().isin(["", "nan", "none", "<na>"])
        )
        groups = company_ids.map(
            lambda value: f"company_id:{value}" if pd.notna(value) else pd.NA
        )

    if "company_name" in data.columns:
        company_names = data["company_name"].astype("string").str.strip().str.lower()
        company_names = company_names.mask(
            company_names.isin(["", "nan", "none", "<na>", "unknown"])
        )
        groups = groups.fillna(
            company_names.map(
                lambda value: f"company_name:{value}" if pd.notna(value) else pd.NA
            )
        )

    if "job_id" in data.columns:
        job_ids = data["job_id"].astype("string").str.strip()
        job_ids = job_ids.mask(job_ids.str.lower().isin(["", "nan", "none", "<na>"]))
        groups = groups.fillna(job_ids.map(lambda value: f"job_id:{value}"))

    row_groups = pd.Series(data.index.astype(str), index=data.index).radd("row:")
    return groups.fillna(row_groups)


def build_features(data_path=DATA_PATH, return_groups=False):
    data_path = Path(data_path)
    data = pd.read_csv(data_path, low_memory=False)
    print("Data loaded successfully!")
    print("Shape:", data.shape)

    target = "formatted_experience_level"
    data = data.dropna(subset=[target]).copy()
    groups = _build_company_groups(data)
    x = data.drop(columns=[target]).copy()
    y = data[target].astype(str)
    print("\nTarget distribution:")
    print(y.value_counts())

    x = add_text_features(x)
    x = extract_skill_features(x)
    x = add_temporal_features(x)

    columns_to_drop = [
        "job_id", "company_id", "zip_code", "fips", "salary_id",
        "original_listed_time", "listed_time", "expiry", "closed_time",
        "job_posting_url", "application_url",
        "skills_desc"
    ]
    x = x.drop(columns=[column for column in columns_to_drop if column in x.columns])

    splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
    train_indices, test_indices = next(splitter.split(x, y, groups=groups))
    result = (
        x.iloc[train_indices],
        x.iloc[test_indices],
        y.iloc[train_indices],
        y.iloc[test_indices],
    )
    if return_groups:
        return (*result, groups.iloc[train_indices], groups.iloc[test_indices])
    return result

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    build_features()

