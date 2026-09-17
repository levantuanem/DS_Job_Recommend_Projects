import time
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier


# ============================================================
# STEP 8 - CROSS VALIDATION + GRID SEARCH
# ============================================================

DATA_PATH = "data/processed/postings_clean.csv"
OUTPUT_PATH = "data/processed/grid_search_results.csv"

TARGET = "formatted_experience_level"


print("=" * 70)
print("STEP 8 - CROSS VALIDATION + GRID SEARCH")
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

df = df.dropna(subset=[TARGET]).copy()

print(
    "Shape after removing NaN target:",
    df.shape
)


# ============================================================
# 3. SELECT FEATURES
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


print("\n[2] Features:")
print(features)


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

print("\n[3] Splitting train/test...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


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


print("\n[4] Numerical features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


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
# 7. CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# 8. MODELS + PARAMETER GRIDS
# ============================================================

models_and_grids = {

    "Logistic Regression": (
        LogisticRegression(
            max_iter=2000,
            random_state=42
        ),

        {
            "model__C": [
                0.1,
                1.0,
                10.0
            ],

            "model__class_weight": [
                None,
                "balanced"
            ]
        }
    ),


    "Linear SVM": (
        LinearSVC(
            max_iter=5000,
            random_state=42
        ),

        {
            "model__C": [
                0.01,
                0.1,
                1.0,
                10.0
            ],

            "model__class_weight": [
                None,
                "balanced"
            ]
        }
    ),


    "SGD Classifier": (
        SGDClassifier(
            max_iter=2000,
            random_state=42
        ),

        {
            "model__alpha": [
                0.00001,
                0.0001,
                0.001
            ],

            "model__class_weight": [
                None,
                "balanced"
            ]
        }
    ),
}


# ============================================================
# 9. GRID SEARCH
# ============================================================

results = []


for model_name, (model, param_grid) in models_and_grids.items():

    print("\n")
    print("=" * 70)
    print("GRID SEARCH:", model_name)
    print("=" * 70)

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


    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=2,
        return_train_score=True
    )


    start_time = time.time()


    print("\nTraining GridSearchCV...")


    grid.fit(
        X_train,
        y_train
    )


    elapsed_time = time.time() - start_time


    print("\nBest parameters:")
    print(grid.best_params__)


    print(
        "\nBest CV Macro-F1:",
        f"{grid.best_score_:.4f}"
    )


    print(
        "Training time:",
        f"{elapsed_time:.2f} seconds"
    )


    # ========================================================
    # STORE RESULT
    # ========================================================

    results.append({

        "Model":
        model_name,

        "Best CV Macro-F1":
        grid.best_score_,

        "Training Time":
        elapsed_time,

        "Best Parameters":
        str(grid.best_params_)
    })


# ============================================================
# 10. RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results)


results_df = results_df.sort_values(
    by="Best CV Macro-F1",
    ascending=False
)


print("\n")
print("=" * 70)
print("FINAL GRID SEARCH RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 11. SAVE RESULTS
# ============================================================

results_df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)


print("\nResults saved to:")
print(OUTPUT_PATH)


# ============================================================
# 12. FINAL BEST MODEL
# ============================================================

best_model_name = (
    results_df.iloc[0]["Model"]
)

best_score = (
    results_df.iloc[0]["Best CV Macro-F1"]
)


print("\n")
print("=" * 70)
print("BEST MODEL FROM GRID SEARCH")
print("=" * 70)

print(
    "Model:",
    best_model_name
)

print(
    "CV Macro-F1:",
    f"{best_score:.4f}"
)

print("\nSTEP 8 FINISHED.")