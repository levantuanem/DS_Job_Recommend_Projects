import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# 1. LOAD DATA
# ============================================================

def load_data(path):
    """
    Đọc dữ liệu postings_clean.csv.

    Target:
        formatted_experience_level

    Các dòng không có target (NaN) sẽ bị loại bỏ.
    """

    print("=" * 70)
    print("STEP 5 - MODEL COMPARISON")
    print("=" * 70)

    print("\n[1] Loading data...")

    df = pd.read_csv(path)

    print("Original shape:", df.shape)

    # Chỉ giữ những dòng có Job Level
    df = df.dropna(
        subset=["formatted_experience_level"]
    ).copy()

    print("Shape after removing NaN target:", df.shape)

    return df


# ============================================================
# 2. PREPARE X AND y
# ============================================================

def prepare_data(df):
    """
    Tạo:
        X = features
        y = target

    Target:
        formatted_experience_level
    """

    target = "formatted_experience_level"

    # --------------------------------------------------------
    # Các feature cơ bản
    # --------------------------------------------------------

    feature_columns = [
        "title",
        "description",
        "location",
        "company_name",
        "formatted_work_type",
        "work_type",
        "currency",
        "compensation_type",
        "skills_desc",
        "remote_allowed",
        "sponsored",
        "normalized_salary",
        "views",
        "applies",
        "min_salary",
        "max_salary",
        "med_salary"
    ]

    # Chỉ lấy những cột thực sự tồn tại
    feature_columns = [
        col for col in feature_columns
        if col in df.columns
    ]

    X = df[feature_columns].copy()
    y = df[target].copy()

    print("\n[2] Features:")
    print(feature_columns)

    print("\nTarget distribution:")
    print(y.value_counts())

    return X, y


# ============================================================
# 3. BUILD PREPROCESSOR
# ============================================================

def build_preprocessor(X):
    """
    Tiền xử lý dữ liệu:

    Numerical:
        - điền missing bằng median
        - StandardScaler

    Categorical:
        - điền missing bằng most_frequent
        - OneHotEncoder
    """

    numerical_columns = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    print("\n[3] Numerical features:")
    print(numerical_columns)

    print("\nCategorical features:")
    print(categorical_columns)

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numerical_pipeline,
                numerical_columns
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_columns
            )
        ]
    )

    return preprocessor


# ============================================================
# 4. BUILD MODELS
# ============================================================

def build_models():
    """
    Tạo 3 model cần so sánh.
    """

    models = {

        "Logistic Regression":
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced"
            ),

        "Linear SVM":
            LinearSVC(
                random_state=42,
                class_weight="balanced"
            ),

            
    }

    return models


# ============================================================
# 5. EVALUATE MODEL
# ============================================================

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Train model và tính các metric.
    """

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1_macro = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1_weighted = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    return {
        "Accuracy": accuracy,
        "Precision Macro": precision,
        "Recall Macro": recall,
        "F1 Macro": f1_macro,
        "F1 Weighted": f1_weighted
    }


# ============================================================
# 6. MODEL COMPARISON
# ============================================================

def compare_models(X_train, X_test, y_train, y_test):

    models = build_models()

    results = []

    for name, classifier in models.items():

        print("\n" + "=" * 70)
        print(f"TRAINING: {name}")
        print("=" * 70)

        # Mỗi model dùng cùng preprocessing
        preprocessor = build_preprocessor(X_train)

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessing",
                    preprocessor
                ),
                (
                    "model",
                    classifier
                )
            ]
        )

        metrics = evaluate_model(
            pipeline,
            X_train,
            X_test,
            y_train,
            y_test
        )

        metrics["Model"] = name

        results.append(metrics)

        print(
            f"Accuracy        : "
            f"{metrics['Accuracy']:.4f}"
        )

        print(
            f"Precision Macro : "
            f"{metrics['Precision Macro']:.4f}"
        )

        print(
            f"Recall Macro    : "
            f"{metrics['Recall Macro']:.4f}"
        )

        print(
            f"F1 Macro        : "
            f"{metrics['F1 Macro']:.4f}"
        )

        print(
            f"F1 Weighted     : "
            f"{metrics['F1 Weighted']:.4f}"
        )

    # Tạo bảng kết quả
    results_df = pd.DataFrame(results)

    results_df = results_df[
        [
            "Model",
            "Accuracy",
            "Precision Macro",
            "Recall Macro",
            "F1 Macro",
            "F1 Weighted"
        ]
    ]

    # Sắp xếp theo F1 Macro
    results_df = results_df.sort_values(
        by="F1 Macro",
        ascending=False
    )

    return results_df


# ============================================================
# 7. MAIN
# ============================================================

def main():

    data_path = (
        "data/processed/postings_clean.csv"                         # *
    )

    # Load
    df = load_data(data_path)

    # X, y
    X, y = prepare_data(df)

    # Train/Test split
    print("\n[4] Splitting train/test...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)

    print("y_train:", y_train.shape)
    print("y_test :", y_test.shape)

    # Compare
    results = compare_models(
        X_train,
        X_test,
        y_train,
        y_test
    )

    # Final table
    print("\n")
    print("=" * 90)
    print("FINAL MODEL COMPARISON")
    print("=" * 90)

    print(
        results.to_string(
            index=False
        )
    )

    # Save result
    results.to_csv(
        "data/processed/model_comparison_results.csv",
        index=False
    )

    print("\nResults saved to:")
    print(
        "data/processed/model_comparison_results.csv"
    )


if __name__ == "__main__":
    main()