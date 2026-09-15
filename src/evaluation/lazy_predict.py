import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# STEP 6 - MODEL SCREENING
# ============================================================

DATA_PATH = "data/processed/postings_clean.csv"
TARGET = "formatted_experience_level"


print("=" * 70)
print("STEP 6 - MODEL SCREENING")
print("=" * 70)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n[1] Loading data...")

df = pd.read_csv(DATA_PATH)

print("Original shape:", df.shape)


# ============================================================
# 2. REMOVE MISSING TARGET
# ============================================================

df = df.dropna(
    subset=[TARGET]
).copy()

print(
    "Shape after removing NaN target:",
    df.shape
)


# ============================================================
# 3. SELECT FEATURES
# ============================================================

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
    "med_salary",
]


feature_columns = [
    column
    for column in feature_columns
    if column in df.columns
]


X = df[feature_columns].copy()
y = df[TARGET].copy()


print("\n[2] Features:")
print(feature_columns)


print("\nTarget distribution:")
print(y.value_counts())


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

print("\n[3] Splitting train/test...")

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


# ============================================================
# 5. IDENTIFY FEATURE TYPES
# ============================================================

numerical_columns = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()


categorical_columns = X_train.select_dtypes(
    include=["object", "string"]
).columns.tolist()


print("\n[4] Numerical features:")
print(numerical_columns)


print("\nCategorical features:")
print(categorical_columns)


# ============================================================
# 6. NUMERICAL PREPROCESSING
# ============================================================

numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# ============================================================
# 7. CATEGORICAL PREPROCESSING
# ============================================================

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


# ============================================================
# 8. BUILD PREPROCESSOR
# ============================================================

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


# ============================================================
# 9. PREPROCESS TRAINING DATA
# ============================================================

print("\n[5] Preprocessing training data...")

X_train_processed = preprocessor.fit_transform(
    X_train
)


print(
    "Processed X_train shape:",
    X_train_processed.shape
)


# ============================================================
# 10. PREPROCESS TEST DATA
# ============================================================

print("\n[6] Preprocessing test data...")

X_test_processed = preprocessor.transform(
    X_test
)


print(
    "Processed X_test shape:",
    X_test_processed.shape
)


# ============================================================
# 11. KEEP SPARSE MATRIX
# ============================================================

print("\n[7] Preparing sparse matrix...")

print(
    "X_train:",
    X_train_processed.shape
)

print(
    "X_test :",
    X_test_processed.shape
)

print(
    "Sparse matrix is kept in memory."
)

print(
    "No .toarray() conversion."
)


# ============================================================
# 12. DEFINE MODELS
# ============================================================

print("\n")
print("=" * 70)
print("MODEL SCREENING")
print("=" * 70)


models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="saga",
        n_jobs=-1
    ),

    "Linear SVM": LinearSVC(
        class_weight="balanced",
        max_iter=5000,
        random_state=42
    ),

    "SGD Classifier": SGDClassifier(
        loss="hinge",
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Multinomial Naive Bayes": MultinomialNB()
}


# ============================================================
# 13. TRAIN MODELS
# ============================================================

results = []


for name, model in models.items():

    print("\n")
    print("=" * 70)
    print(
        f"TRAINING: {name}"
    )
    print("=" * 70)

    try:

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        print("Training...")

        model.fit(
            X_train_processed,
            y_train
        )


        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

        print("Predicting...")

        y_pred = model.predict(
            X_test_processed
        )


        # ----------------------------------------------------
        # CALCULATE METRICS
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            y_pred
        )


        balanced_accuracy = balanced_accuracy_score(
            y_test,
            y_pred
        )


        precision_macro = precision_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        )


        recall_macro = recall_score(
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


        # ----------------------------------------------------
        # PRINT RESULTS
        # ----------------------------------------------------

        print()

        print(
            f"Accuracy          : "
            f"{accuracy:.4f}"
        )

        print(
            f"Balanced Accuracy : "
            f"{balanced_accuracy:.4f}"
        )

        print(
            f"Precision Macro   : "
            f"{precision_macro:.4f}"
        )

        print(
            f"Recall Macro      : "
            f"{recall_macro:.4f}"
        )

        print(
            f"F1 Macro          : "
            f"{f1_macro:.4f}"
        )

        print(
            f"F1 Weighted       : "
            f"{f1_weighted:.4f}"
        )


        # ----------------------------------------------------
        # SAVE RESULT
        # ----------------------------------------------------

        results.append(
            {
                "Model": name,
                "Accuracy": accuracy,
                "Balanced Accuracy": balanced_accuracy,
                "Precision Macro": precision_macro,
                "Recall Macro": recall_macro,
                "F1 Macro": f1_macro,
                "F1 Weighted": f1_weighted
            }
        )


    except Exception as e:

        print()

        print(
            f"ERROR: {name}"
        )

        print(
            str(e)
        )


# ============================================================
# 14. RESULTS TABLE
# ============================================================

print("\n")
print("=" * 90)
print("FINAL MODEL SCREENING RESULTS")
print("=" * 90)


if len(results) == 0:

    print(
        "No model completed successfully."
    )

else:

    results_df = pd.DataFrame(
        results
    )


    # --------------------------------------------------------
    # SORT BY F1 MACRO
    # --------------------------------------------------------

    results_df = results_df.sort_values(
        by="F1 Macro",
        ascending=False
    )


    print()

    print(
        results_df.to_string(
            index=False
        )
    )


    # ========================================================
    # 15. SAVE RESULTS
    # ========================================================

    output_path = (
        "data/processed/"
        "lazypredict_results.csv"
    )


    results_df.to_csv(
        output_path,
        index=False
    )


    print("\n")

    print(
        "Results saved to:"
    )

    print(
        output_path
    )


# ============================================================
# 16. COMPLETED
# ============================================================

print("\n")
print("=" * 70)
print("STEP 6 COMPLETED")
print("=" * 70)