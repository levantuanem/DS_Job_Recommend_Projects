import pandas as pd


def create_temporal_features(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()

    datetime_columns = [
        column
        for column in ["listed_time", "expiry"]
        if column in result.columns
    ]

    for column in datetime_columns:
        if not pd.api.types.is_datetime64_any_dtype(result[column]):
            result[column] = pd.to_datetime(
                result[column],
                unit="ms",
                errors="coerce",
            )

    if {"listed_time", "expiry"}.issubset(result.columns):
        result["duration_days"] = (
            result["expiry"] - result["listed_time"]
        ).dt.total_seconds() / (24 * 3600)

    if "listed_time" in result.columns:
        listed_time = result["listed_time"]
        result["post_month"] = listed_time.dt.month
        result["post_dayofweek"] = listed_time.dt.dayofweek
        result["post_hour"] = listed_time.dt.hour
        result["is_weekend"] = result["post_dayofweek"].ge(5).astype("Int64")

    return result