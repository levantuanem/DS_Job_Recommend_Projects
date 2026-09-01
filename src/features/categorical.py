import numpy as np
import pandas as pd

def prepare_categorical_features(
    df: pd.DataFrame,
    columns: list[str]
) -> pd.DataFrame:

    result = df.copy()

    existing_columns = [
        column
        for column in columns
        if column in result.columns
    ]

    for column in existing_columns:
        result[column] = (
            result[column]
            .astype("string")
            .fillna("Unknown")
            .str.strip()
        )

    return result