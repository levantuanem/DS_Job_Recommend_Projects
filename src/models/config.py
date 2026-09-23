"""
config.py
---------
Cau hinh dung chung cho toan bo module Modeling & Evaluation (Member 4).
Tuong ung voi configs/model.yaml trong README (co the sau nay tach ra file YAML
va doc bang src/utils/config_loader.py cua Member 5).
"""

import os

# ============ DUONG DAN ============
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")  # luu hinh, bang so sanh

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ============ TARGET ============
TARGET_COL = "formatted_experience_level"
CLASS_LABELS = [
    "Internship",
    "Entry Level",
    "Associate",
    "Mid-Senior Level",
    "Director",
    "Executive",
]

# ============ SPLIT ============
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.2  # trich tu phan train de co validation set rieng cho bias/variance
CV_FOLDS = 5

# ============ CLASS IMBALANCE ============
# "class_weight" | "smote" | "none"
IMBALANCE_STRATEGY = "class_weight"

# ============ SCORING CHINH (dung cho CV & tuning) ============
# Vi du can chu y Macro-F1 khi mat can bang class
PRIMARY_SCORING = "f1_macro"