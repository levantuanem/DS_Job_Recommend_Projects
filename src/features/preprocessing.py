from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
from .nlp import build_tfidf_vectorizer

DEFAULT_NUMERICAL_FEATURES = [
    "salary_range",
    "log_salary",
    "application_rate",
    "view_to_apply_ratio",
    "skill_count",
    "text_length",
    "word_count",
    "sentence_count",
    "keyword_count",
    "technical_keyword_count",
    "management_keyword_count",
    "leadership_keyword_count",
    "duration_days",
    "post_month",
    "post_dayofweek",
    "post_hour",
    "is_weekend"]

DEFAULT_CATEGORICAL_FEATURES = [
    "work_type",
    "formatted_work_type",
    "remote_allowed",
    "pay_period",
    "currency",
    "location",
    "application_type"]


def build_preprocessor(
    data: pd.DataFrame,
    numerical_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    text_column: str = "combined_text") -> ColumnTransformer:
    numerical_features = [
        column
        for column in (numerical_features or DEFAULT_NUMERICAL_FEATURES)
        if column in data.columns]
    categorical_features = [
        column
        for column in (categorical_features or DEFAULT_CATEGORICAL_FEATURES)
        if column in data.columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )
    small_dataset = len(data) < 20
    text_vectorizer = build_tfidf_vectorizer(
        min_df=1 if small_dataset else 2,
        max_df=1.0 if small_dataset else 0.95
    )

    transformers = []
    if numerical_features:
        transformers.append(("num", numeric_pipeline, numerical_features))
    if categorical_features:
        transformers.append(("cat", categorical_pipeline, categorical_features))
    if text_column in data.columns:
        transformers.append(("text", text_vectorizer, text_column))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )