import pandas as pd
from .numerical import create_numerical_features
from .categorical import prepare_categorical_features
from .text_features import create_text_features
from .skill_extraction import extract_skill_features

TARGET_COLUMN = "formatted_experience_level"

CATEGORICAL_COLUMNS = [
    "work_type",
    "formatted_work_type",
    "remote_allowed",
    "pay_period",
    "currency",
    "location",
    "application_type"]

TEXT_COLUMNS = ["title", 
                "description", 
                "skills_desc"]

def build_features(data: pd.DataFrame ) -> tuple[pd.DataFrame, pd.Series]:
    # ========================================================
    # 1. Check target
    # ========================================================
    if TARGET_COLUMN not in data.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' "
            "does not exist in the dataset.")
    # ========================================================
    # 2. Copy data
    # ========================================================
    data = data.copy()
    # ========================================================
    # 3. Numerical Feature Engineering
    # ========================================================
    data = create_numerical_features(data)
    # ========================================================
    # 4. Categorical Feature Engineering
    # ========================================================
    existing_categorical = [
        column
        for column in CATEGORICAL_COLUMNS
        if column in data.columns]
    data = prepare_categorical_features(data, existing_categorical)

    # ========================================================
    # 5. Text Feature Engineering
    # ========================================================

    existing_text = [
        column
        for column in TEXT_COLUMNS
        if column in data.columns]
    data = create_text_features(data, existing_text)

    # ========================================================
    # 6. Skill Extraction
    # ========================================================

    data = extract_skill_features(data, existing_text)

    # ========================================================
    # 7. Separate X and y
    # ========================================================
    y = data[TARGET_COLUMN].copy()
    X = data.drop( columns=[TARGET_COLUMN]).copy()

    # ========================================================
    # 8. Remove helper columns
    # ========================================================

    helper_columns = ["skill_text"]
    X = X.drop(
        columns=[
            column
            for column in helper_columns
            if column in X.columns
        ]
    )

    # ========================================================
    # 9. Return
    # ========================================================
    return X, y