"""
tune.py
-------
Buoc 3 cua Member 4: Hyperparameter Tuning (muc 9.6 trong README).

Chi nen tune 1-3 model tiem nang nhat (lay tu ket qua train.py / lazy_predict_runner.py)
de tiet kiem thoi gian, khong nen GridSearch toan bo candidate models.
"""

import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold

from src.models.config import RANDOM_STATE, CV_FOLDS, PRIMARY_SCORING


# Param grid goi y cho tung model - co the chinh lai trong configs/model.yaml
PARAM_GRIDS = {
    "Logistic Regression": {
        "C": [0.01, 0.1, 1, 10, 100],
        "penalty": ["l2"],
        "solver": ["lbfgs"],
    },
    "Random Forest": {
        "n_estimators": [200, 300, 500],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "Gradient Boosting": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [2, 3, 4],
    },
    "XGBoost": {
        "n_estimators": [200, 300, 500],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
    },
    "LightGBM": {
        "n_estimators": [200, 300, 500],
        "num_leaves": [15, 31, 63],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.8, 1.0],
    },
}


def tune_model(model_name, base_model, X_train, y_train, search_type="random", n_iter=25):
    """
    Tune 1 model bang GridSearchCV hoac RandomizedSearchCV + Stratified Cross Validation.

    search_type:
        "grid"   -> GridSearchCV (tim toan bo, cham hon, chinh xac hon)
        "random" -> RandomizedSearchCV (nhanh hon, phu hop khi param grid lon)
    """
    if model_name not in PARAM_GRIDS:
        raise ValueError(
            f"Chua co param grid cho model '{model_name}'. "
            f"Hay them vao PARAM_GRIDS trong tune.py."
        )

    param_grid = PARAM_GRIDS[model_name]
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    if search_type == "grid":
        search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring=PRIMARY_SCORING,
            cv=cv,
            n_jobs=-1,
            verbose=1,
        )
    else:
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring=PRIMARY_SCORING,
            cv=cv,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=1,
        )

    print(f"\n>>> Dang tune: {model_name} ({search_type} search)")
    search.fit(X_train, y_train)

    print(f"    Best {PRIMARY_SCORING}: {search.best_score_:.4f}")
    print(f"    Best params: {search.best_params_}")

    return search.best_estimator_, search.best_params_, search.best_score_


def tune_top_models(top_model_names, fitted_models_dict, X_train, y_train, search_type="random"):
    """
    Tune nhieu model cung luc. fitted_models_dict lay tu train.train_and_cross_validate().
    Tra ve dict: {model_name: (best_estimator, best_params, best_score)}
    """
    tuned_results = {}
    for name in top_model_names:
        if name not in fitted_models_dict:
            print(f"Bo qua '{name}' vi khong co trong fitted_models_dict.")
            continue
        base_model = fitted_models_dict[name]
        best_est, best_params, best_score = tune_model(
            name, base_model, X_train, y_train, search_type=search_type
        )
        tuned_results[name] = {
            "best_estimator": best_est,
            "best_params": best_params,
            "best_cv_score": best_score,
        }
    return tuned_results