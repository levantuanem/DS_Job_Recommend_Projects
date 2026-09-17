"""
Pipeline flow:

Raw Data
    ↓
Data Cleaning
    ↓
Feature Engineering
    ↓
Preprocessing
    ↓
Train/Test Dataset
    ↓
Model Training      [future]
    ↓
Evaluation          [future]
    ↓
Prediction          [future]
"""

from pathlib import Path
from src.data.clean_data import run_pipeline as run_data_cleaning
from src.features.build_features import build_features

# ============================================================
# PROJECT PATHS
# ============================================================
ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

# ============================================================
# MAIN PIPELINE
# ============================================================
def run_pipeline( apply_feature_selection=False, k=1000):
    print("=" * 70)
    print("DS JOB RECOMMENDATION - MAIN PIPELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # STEP 1 — DATA CLEANING
    # --------------------------------------------------------
    print("\n[1/7] DATA CLEANING")
    print("-" * 70)
    run_data_cleaning(raw_dir=RAW_DIR, processed_dir=PROCESSED_DIR)
    print("Data cleaning completed.")

    # --------------------------------------------------------
    # STEP 2 — FEATURE ENGINEERING + PREPROCESSING
    # --------------------------------------------------------
    print("\n[2/7] FEATURE ENGINEERING + PREPROCESSING")
    print("-" * 70)
    X_train, X_test, y_train, y_test = build_features(apply_feature_selection=apply_feature_selection, k=k)
    print("Feature engineering completed.")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape:  {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape:  {y_test.shape}")

    # --------------------------------------------------------
    # STEP 3 — MODEL TRAINING
    # --------------------------------------------------------
    print("\n[3/7] MODEL TRAINING")
    print("-" * 70)
    print("Model training is not implemented yet.")

    # Future:
    #
    # model = train_model(
    #     X_train,
    #     y_train,
    # )
    #
    # save_model(model)

    model = None

    # --------------------------------------------------------
    # STEP 4 — EVALUATION
    # --------------------------------------------------------

    print("\n[4/7] EVALUATION")
    print("-" * 70)
    print("Evaluation is not implemented yet.")

    # Future:
    #
    # metrics = evaluate_model(
    #     model,
    #     X_test,
    #     y_test,
    # )

    metrics = None

    # --------------------------------------------------------
    # STEP 5 — PREDICTION
    # --------------------------------------------------------
    print("\n[5/7] PREDICTION")
    print("-" * 70)
    print("Prediction is not implemented yet.")

    # Future:
    #
    # predictions = predict(
    #     model,
    #     X_test,
    # )

    predictions = None

    # --------------------------------------------------------
    # STEP 6 — SAVE ARTIFACTS
    # --------------------------------------------------------
    print("\n[6/7] SAVE ARTIFACTS")
    print("-" * 70)
    print("Model/evaluation artifacts will be saved here later.")

    # Future:
    #
    # save_model(model)
    # save_metrics(metrics)
    # save_predictions(predictions)

    # --------------------------------------------------------
    # STEP 7 — PIPELINE COMPLETE
    # --------------------------------------------------------
    print("\n[7/7] PIPELINE STATUS")
    print("-" * 70)
    print("Current pipeline stages completed:")
    print("  ✓ Data Cleaning")
    print("  ✓ Feature Engineering")
    print("  ✓ Preprocessing")
    print("  ✓ Train/Test Split")
    print("  - Model Training (future)")
    print("  - Evaluation (future)")
    print("  - Prediction (future)")
    print("\n" + "=" * 70)
    print("PIPELINE FINISHED")
    print("=" * 70)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "model": model,
        "metrics": metrics,
        "predictions": predictions,
    }

if __name__ == "__main__":
    run_pipeline()