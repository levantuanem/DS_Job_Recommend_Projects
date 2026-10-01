from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
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
        "skill_count",
    ]

    categorical_features = [
        "company_name",
        "location",
        "pay_period",
        "formatted_work_type",
        "posting_domain",
        "application_type",
        "work_type",
        "currency",
        "compensation_type",
    ]

    binary_features = [
        "remote_allowed",
        "sponsored",
    ]

    skill_features = [
        column
        for column in x.columns
        if column.startswith("skill_")
        and column != "skill_count"
    ]
    numeric_features.extend(skill_features)

    all_structured_features = (
        numeric_features
        + categorical_features
        + binary_features
    )

    missing_features = [
        column
        for column in all_structured_features
        if column not in x.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing features: {missing_features}"
        )

    if "combined_text" not in x.columns:
        raise ValueError(
            "Missing required text feature: combined_text"
        )

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

    binary_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])

    tf_idf_vectorizer = TfidfVectorizer(
        stop_words="english",
        min_df=10,
        max_df=0.95,
        max_features=20000
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
            ("binary", binary_pipeline, binary_features),
            ("text", tf_idf_vectorizer, "combined_text")
        ]
    )

def build_features(apply_feature_selection=False, k=1000):
    data = pd.read_csv(DATA_PATH, low_memory=False)
    print("Data loaded successfully!")
    print("Shape:", data.shape)

    target = "formatted_experience_level"
    data = data.dropna(subset=[target]).copy()
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
        "title", "description", "skills_desc",
    ]
    x = x.drop(columns=[column for column in columns_to_drop if column in x.columns])

    return train_test_split(
        x,
        y,
        test_size=0.1,
        random_state=42,
        stratify=y,
    )

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    build_features(
        apply_feature_selection=True,
        k=1000
    )

