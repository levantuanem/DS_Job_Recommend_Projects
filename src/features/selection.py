import pandas as pd
from sklearn.feature_selection import mutual_info_classif


def calculate_mutual_information(
    X: pd.DataFrame,
    y: pd.Series) -> pd.Series:

    numeric_X = X.select_dtypes(include=["number"]).copy()
    numeric_X = numeric_X.fillna(0)
    scores = mutual_info_classif(numeric_X, y, random_state=42,)
    return pd.Series(
        scores,
        index=numeric_X.columns,
        name="mutual_information").sort_values(ascending=False)