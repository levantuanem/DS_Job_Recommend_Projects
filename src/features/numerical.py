import numpy as np
import pandas as pd

def create_numerical_features(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    # --------------------------------------------------------
    # Salary range
    # --------------------------------------------------------
    if {"min_salary", "max_salary"}.issubset(result.columns):
        result["salary_range"] = (result["max_salary"] - result["min_salary"])
    # --------------------------------------------------------
    # Log salary
    # --------------------------------------------------------
    if "normalized_salary" in result.columns:
        result["log_salary"] = np.log1p(result["normalized_salary"].clip(lower=0))
    # --------------------------------------------------------
    # Application rate
    # --------------------------------------------------------
    if {"applies", "views"}.issubset(result.columns):
        result["application_rate"] = np.where(
            result["views"] > 0, result["applies"] / result["views"], 0.0)
    # --------------------------------------------------------
    # View / apply ratio
    # --------------------------------------------------------
    if {"views", "applies"}.issubset(result.columns):
        result["view_to_apply_ratio"] = np.where(
            result["applies"] > 0, result["views"] / result["applies"], 0.0)
    return result