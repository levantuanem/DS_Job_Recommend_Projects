"""
train.py
--------
Buoc 2 cua Member 4: Train cac model chinh thuc (khong phai LazyPredict).

Gom:
- Baseline model: Logistic Regression
- Candidate models: Random Forest, Linear SVM, Gradient Boosting, XGBoost, LightGBM
- Xu ly Class Imbalance: class_weight hoac SMOTE (chi ap dung tren TRAIN)
- Cross Validation (StratifiedKFold) de danh gia do on dinh / variance cua model
- Ghi lai: CV Mean, CV Std, Training Time cho tung model
"""

import time
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

from imblearn.over_sampling import SMOTE

from src.models.config import RANDOM_STATE, CV_FOLDS, PRIMARY_SCORING, IMBALANCE_STRATEGY


def get_candidate_models(class_weight=None):
    """
    Dinh nghia Baseline + Candidate models theo dung muc 9.1 va 9.2 trong README.
    class_weight duoc truyen vao neu IMBALANCE_STRATEGY == "class_weight".
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight=class_weight,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight=class_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        # LinearSVC khong co predict_proba -> boc bang CalibratedClassifierCV
        # de dung duoc cho Log Loss / predict_proba khi can.
        "Linear SVM": CalibratedClassifierCV(
            LinearSVC(
                class_weight=class_weight,
                random_state=RANDOM_STATE,
                max_iter=5000,
            ),
            cv=3,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
        ),
    }

    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if HAS_LGBM:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=300,
            class_weight=class_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    return models


def apply_class_imbalance_strategy(X_train, y_train):
    """
    Ap dung chien luoc xu ly mat can bang class THEO README muc 9.4.
    QUAN TRONG: chi ap dung tren TRAIN, khong dung tren Validation/Test.
    Tra ve: (X_train_resampled, y_train_resampled, class_weight_param)
    """
    if IMBALANCE_STRATEGY == "smote":
        sm = SMOTE(random_state=RANDOM_STATE)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        print(f"Da ap dung SMOTE: {X_train.shape[0]} -> {X_res.shape[0]} mau.")
        return X_res, y_res, None

    if IMBALANCE_STRATEGY == "class_weight":
        return X_train, y_train, "balanced"

    return X_train, y_train, None


def train_and_cross_validate(X_train, y_train):
    """
    Train tung candidate model va danh gia bang StratifiedKFold Cross Validation.
    Tra ve DataFrame ket qua (CV Mean, CV Std, Training Time) + dict model da fit tren full train.
    """
    X_train_res, y_train_res, class_weight = apply_class_imbalance_strategy(X_train, y_train)
    models = get_candidate_models(class_weight=class_weight)

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    results = []
    fitted_models = {}

    for name, model in models.items():
        print(f"\n>>> Dang train & cross-validate: {name}")
        start = time.time()

        cv_scores = cross_val_score(
            model, X_train_res, y_train_res,
            cv=cv, scoring=PRIMARY_SCORING, n_jobs=-1,
        )

        # Fit lai tren toan bo train (sau khi da co CV score) de dung cho buoc evaluate/tune
        model.fit(X_train_res, y_train_res)
        elapsed = time.time() - start

        fitted_models[name] = model
        results.append({
            "Model": name,
            "CV Mean (f1_macro)": np.round(cv_scores.mean(), 4),
            "CV Std": np.round(cv_scores.std(), 4),
            "Scores by Fold": np.round(cv_scores, 4).tolist(),
            "Training Time (s)": np.round(elapsed, 2),
        })

        print(f"    CV Mean: {cv_scores.mean():.4f} | CV Std: {cv_scores.std():.4f} "
              f"| Time: {elapsed:.2f}s")

    results_df = pd.DataFrame(results).sort_values(
        by="CV Mean (f1_macro)", ascending=False
    ).reset_index(drop=True)

    return results_df, fitted_models