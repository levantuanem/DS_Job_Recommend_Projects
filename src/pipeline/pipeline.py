from pathlib import Path
from src.data.clean_data import run_pipeline as run_data_cleaning
from src.models.utils import (get_train_test_data)
from src.models.train import CV_FOLDS, train_all_models
from src.models.tune import tune_best_model
from src.models.evaluate import evaluate_best_model
from src.models.predict import predict

# ============================================================
# PROJECT PATHS
# ============================================================
ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# MAIN PIPELINE
# ============================================================
def run_pipeline(
    apply_feature_selection=False,
    k=1000,
    prediction_input=None,
    prediction_output=None,
    prediction_limit=100):

    print("=" * 70)
    print("DS JOB RECOMMENDATION - MAIN ML PIPELINE")
    print("=" * 70)

    # ========================================================
    # STEP 1 — DATA CLEANING
    # ========================================================
    print("\n[1/7] DATA CLEANING")
    print("-" * 70)

    run_data_cleaning(
        raw_dir=RAW_DIR,
        processed_dir=PROCESSED_DIR,
        balance_target=False,
    )
    print("✓ Data cleaning completed.")

    # ========================================================
    # STEP 2 — FEATURE ENGINEERING + PREPROCESSING
    # ========================================================
    print("\n[2/7] FEATURE ENGINEERING + PREPROCESSING")
    print("-" * 70)
    print("Building features through build_features()...")
    print(f"Feature selection : {apply_feature_selection}")

    if apply_feature_selection:
        print(f"Selected features : {k}")

    X_train, X_test, y_train, y_test, label_encoder = (
        get_train_test_data(force_rebuild=True)
    )

    print("✓ Feature engineering completed.")
    print("✓ Preprocessing completed.")
    print("✓ Train/test split completed.")
    print("✓ Feature cache created.")
    print(f"\nX_train shape : {X_train.shape}")
    print(f"X_test shape  : {X_test.shape}")
    print(f"y_train shape : {y_train.shape}")
    print(f"y_test shape  : {y_test.shape}")
    print(
        f"Classes ({len(label_encoder.classes_)}): "
        f"{list(label_encoder.classes_)}"
    )

    # ========================================================
    # STEP 3 — BASELINE MODEL TRAINING
    # ========================================================
    print("\n[3/7] BASELINE MODEL TRAINING")
    print("-" * 70)
    print("Running train.py...")
    print("Model: Logistic Regression")
    print("Class imbalance: class_weight='balanced'")
    print(f"Cross-validation: StratifiedGroupKFold ({CV_FOLDS} folds, grouped by company)")
    print("Metric: Macro-F1")

    comparison_df = train_all_models(
        apply_feature_selection=apply_feature_selection,
        k=k,
    )

    print("\n✓ Baseline model training completed.")

    # ========================================================
    # STEP 4 — HYPERPARAMETER TUNING
    # ========================================================
    print("\n[4/7] HYPERPARAMETER TUNING")
    print("-" * 70)
    print("Running tune.py...")
    print("Search method: RandomizedSearchCV")
    print("Scoring: Macro-F1")
    search = tune_best_model()
    print("\n✓ Hyperparameter tuning completed.")
    print(f"Best parameters: {search.best_params_}")
    print(f"Best CV Macro-F1: {search.best_score_:.4f}")

    # ========================================================
    # STEP 5 — FINAL MODEL EVALUATION
    # ========================================================
    print("\n[5/7] FINAL MODEL EVALUATION")
    print("-" * 70)
    print("Running evaluate.py...")
    print("Evaluation model: models/best_model.pkl")
    metrics = evaluate_best_model()
    print("\n✓ Final model evaluation completed.")
    if metrics is not None:
        print("\nFinal evaluation summary:")
        metric_keys = [
            "train_accuracy",
            "train_f1_macro",
            "test_accuracy",
            "test_f1_macro",
            "test_f1_weighted",
        ]
        for key in metric_keys:
            if key in metrics:
                print(f"{key}: {metrics[key]:.4f}")
        if "diagnosis" in metrics:
            print(f"Diagnosis: {metrics['diagnosis']}")

    # ========================================================
    # STEP 6 — PREDICTION ON NEW DATA
    # ========================================================
    print("\n[6/7] PREDICTION")
    print("-" * 70)
    predictions = None
    if prediction_input is not None:
        if prediction_output is None:
            prediction_output = REPORTS_DIR / "predictions.csv"
        prediction_input = Path(prediction_input)
        prediction_output = Path(prediction_output)
        print(f"Input : {prediction_input}")
        print(f"Output: {prediction_output}")
        print(f"Prediction sample limit: {prediction_limit}")
        predictions = predict(
            input_path=str(prediction_input),
            output_path=str(prediction_output),
            limit=prediction_limit,
        )
        print("✓ Prediction completed.")
    else:
        print("No new prediction data provided.")
        print("Prediction stage skipped.")
        print(
            "\nTo run prediction manually:"
        )
        print(
            "python -m src.models.predict "
            "--input data/raw/new_postings.csv "
            "--output reports/predictions.csv "
            "--limit 100"
        )

    # ========================================================
    # STEP 7 — PIPELINE STATUS
    # ========================================================
    print("\n[7/7] PIPELINE STATUS")
    print("-" * 70)
    print("  ✓ Data Cleaning")
    print("  ✓ Feature Engineering")
    print("  ✓ Preprocessing")
    print("  ✓ Train/Test Split")
    print("  ✓ Baseline Model Training")
    print("  ✓ Hyperparameter Tuning")
    print("  ✓ Final Model Evaluation")

    if predictions is not None:
        print("  ✓ Prediction on New Data")
    else:
        print("  - Prediction on New Data (skipped)")

    print("\nArtifacts generated:")
    print(f"  Models  : {MODELS_DIR}")
    print(f"  Reports : {REPORTS_DIR}")
    print("\n" + "=" * 70)
    print("PIPELINE FINISHED SUCCESSFULLY")
    print("=" * 70)

    # ========================================================
    # RETURN PIPELINE RESULTS
    # ========================================================
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "label_encoder": label_encoder,
        "model_comparison": comparison_df,
        "tuning_search": search,
        "metrics": metrics,
        "predictions": predictions,
    }

# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    run_pipeline(
        apply_feature_selection=False,
        k=1000,
        prediction_input=ROOT_DIR / "data" / "raw" / "new_postings.csv",
        prediction_output=REPORTS_DIR / "predictions.csv"
    )
