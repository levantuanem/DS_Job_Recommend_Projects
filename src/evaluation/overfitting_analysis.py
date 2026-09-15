import time
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# STEP 9 - UNDERFITTING / OVERFITTING ANALYSIS
# ============================================================

DATA_PATH = "data/processed/postings_clean.csv"
OUTPUT_PATH = "data/processed/overfitting_analysis.csv"

TARGET = "formatted_experience_level"


print("=" * 70)
print("STEP 9 - UNDERFITTING / OVERFITTING ANALYSIS")
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
# 3. FEATURES
# ============================================================

features = [
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

X = df[features]
y = df[TARGET]


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

print("\n[2] Splitting train/test...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)


# ============================================================
# 5. FEATURE TYPES
# ============================================================

numeric_features = [
    "remote_allowed",
    "sponsored",
    "normalized_salary",
    "views",
    "applies",
    "min_salary",
    "max_salary",
    "med_salary",
]

categorical_features = [
    "title",
    "description",
    "location",
    "company_name",
    "formatted_work_type",
    "work_type",
    "currency",
    "compensation_type",
    "skills_desc",
]


# ============================================================
# 6. PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
    ]
)


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
        ),
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_pipeline,
            numeric_features
        ),
        (
            "cat",
            categorical_pipeline,
            categorical_features
        ),
    ]
)


# ============================================================
# 7. MODEL
# ============================================================

print("\n[3] Creating Linear SVM...")

model = LinearSVC(
    C=1.0,
    class_weight="balanced",
    max_iter=5000,
    random_state=42
)


pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ]
)


# ============================================================
# 8. TRAINING
# ============================================================

print("\n[4] Training model...")

start_time = time.time()

pipeline.fit(
    X_train,
    y_train
)

training_time = time.time() - start_time

print(
    "Training time:",
    f"{training_time:.2f} seconds"
)


# ============================================================
# 9. TRAINING PREDICTION
# ============================================================

print("\n[5] Evaluating Training Set...")

y_train_pred = pipeline.predict(X_train)


train_accuracy = accuracy_score(
    y_train,
    y_train_pred
)

train_precision = precision_score(
    y_train,
    y_train_pred,
    average="macro",
    zero_division=0
)

train_recall = recall_score(
    y_train,
    y_train_pred,
    average="macro",
    zero_division=0
)

train_f1 = f1_score(
    y_train,
    y_train_pred,
    average="macro",
    zero_division=0
)


# ============================================================
# 10. TEST PREDICTION
# ============================================================

print("\n[6] Evaluating Test Set...")

y_test_pred = pipeline.predict(X_test)


test_accuracy = accuracy_score(
    y_test,
    y_test_pred
)

test_precision = precision_score(
    y_test,
    y_test_pred,
    average="macro",
    zero_division=0
)

test_recall = recall_score(
    y_test,
    y_test_pred,
    average="macro",
    zero_division=0
)

test_f1 = f1_score(
    y_test,
    y_test_pred,
    average="macro",
    zero_division=0
)


# ============================================================
# 11. PRINT RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("TRAIN vs TEST")
print("=" * 70)

print(
    f"{'Metric':<20}"
    f"{'Training':>15}"
    f"{'Test':>15}"
    f"{'Difference':>15}"
)

print("-" * 65)

print(
    f"{'Accuracy':<20}"
    f"{train_accuracy:>15.4f}"
    f"{test_accuracy:>15.4f}"
    f"{train_accuracy - test_accuracy:>15.4f}"
)

print(
    f"{'Precision Macro':<20}"
    f"{train_precision:>15.4f}"
    f"{test_precision:>15.4f}"
    f"{train_precision - test_precision:>15.4f}"
)

print(
    f"{'Recall Macro':<20}"
    f"{train_recall:>15.4f}"
    f"{test_recall:>15.4f}"
    f"{train_recall - test_recall:>15.4f}"
)

print(
    f"{'F1 Macro':<20}"
    f"{train_f1:>15.4f}"
    f"{test_f1:>15.4f}"
    f"{train_f1 - test_f1:>15.4f}"
)


# ============================================================
# 12. SIMPLE FIT ANALYSIS
# ============================================================

gap = train_f1 - test_f1


print("\n")
print("=" * 70)
print("FIT ANALYSIS")
print("=" * 70)


if train_f1 < 0.60 and test_f1 < 0.60:

    conclusion = "UNDERFITTING"

    print(
        "Conclusion: UNDERFITTING"
    )

    print(
        "Training and Test Macro-F1 are both relatively low."
    )


elif gap > 0.10:

    conclusion = "POSSIBLE OVERFITTING"

    print(
        "Conclusion: POSSIBLE OVERFITTING"
    )

    print(
        "Training Macro-F1 is considerably higher than Test Macro-F1."
    )


else:

    conclusion = "REASONABLE FIT"

    print(
        "Conclusion: REASONABLE FIT"
    )

    print(
        "Training and Test performance are reasonably close."
    )


# ============================================================
# 13. SAVE RESULTS
# ============================================================

results = pd.DataFrame([
    {
        "Model": "Linear SVM",
        "Train Accuracy": train_accuracy,
        "Test Accuracy": test_accuracy,
        "Train Precision Macro": train_precision,
        "Test Precision Macro": test_precision,
        "Train Recall Macro": train_recall,
        "Test Recall Macro": test_recall,
        "Train F1 Macro": train_f1,
        "Test F1 Macro": test_f1,
        "F1 Gap": gap,
        "Training Time": training_time,
        "Conclusion": conclusion,
    }
])


results.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)


print("\nResults saved to:")
print(OUTPUT_PATH)


print("\n")
print("=" * 70)
print("STEP 9 FINISHED")
print("=" * 70)