import pandas as pd

SUSPICIOUS_POSTING_ENGAGEMENT_FEATURES = {
    "views",
    "applies",
    "application_rate",
    "view_to_apply_ratio"}

def check_leakage(
    df: pd.DataFrame,
    target: str) -> dict:

    report = {
        "target": target,
        "target_exists": target in df.columns,
        "suspicious_features": [],
        "constant_features": [],
        "duplicate_columns": []}

    for column in SUSPICIOUS_POSTING_ENGAGEMENT_FEATURES:
        if column in df.columns:
            report["suspicious_features"].append(column)

    for column in df.columns:
        if column == target:
            continue
        if df[column].nunique(dropna=False) <= 1:
            report["constant_features"].append(column)

    duplicated_columns = df.columns[
        df.T.duplicated()
    ].tolist()

    report["duplicate_columns"] = duplicated_columns

    return report