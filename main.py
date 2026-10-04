"""Single-file CLI for the DS Job Recommendation pipeline.

Requires the project data files and third-party packages in requirements.txt.
"""

import argparse
import hashlib
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedGroupKFold,
    cross_val_score,
)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

def report_missing(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame(columns=["n_missing", "pct_missing"])
    miss = df.isna().sum()
    pct = (miss / len(df) * 100).round(2)
    rep = pd.DataFrame({"n_missing": miss, "pct_missing": pct})
    return rep[rep["n_missing"] > 0].sort_values("pct_missing", ascending=False)

def drop_missing_required(
    df: pd.DataFrame, subset: List[str]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    before = len(df)
    existing_cols = [c for c in subset if c in df.columns]
    if not existing_cols:
        return df, {
            "action": "drop_missing_required",
            "columns": subset,
            "n_before": before,
            "n_after": before,
            "n_removed": 0,
            "reason": "Không tìm thấy cột bắt buộc trong DataFrame",
        }
    df_clean = df.dropna(subset=existing_cols).copy()
    after = len(df_clean)
    report = {
        "action": "drop_missing_required",
        "columns": existing_cols,
        "n_before": before,
        "n_after": after,
        "n_removed": before - after,
        "reason": f"Các trường bắt buộc ({', '.join(existing_cols)}) không được để trống.",
    }
    return df_clean, report


def drop_duplicate_records(
    df: pd.DataFrame, subset: Optional[Union[str, List[str]]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    before = len(df)
    df_clean = df.drop_duplicates(subset=subset).copy()
    after = len(df_clean)
    report = {
        "action": "drop_duplicates",
        "subset": subset,
        "n_before": before,
        "n_after": after,
        "n_removed": before - after,
    }
    return df_clean, report


def fix_salary_range(
    df: pd.DataFrame,
    min_col: str = "min_salary",
    max_col: str = "max_salary",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = df.copy()
    if min_col not in df.columns or max_col not in df.columns:
        return df, {"action": "fix_salary_range_swap", "n_affected": 0, "reason": "Columns not found"}
    mask = df[min_col].notna() & df[max_col].notna() & (df[min_col] > df[max_col])
    n_affected = int(mask.sum())
    if n_affected > 0:
        df.loc[mask, [min_col, max_col]] = df.loc[mask, [max_col, min_col]].values
    report = {
        "action": "fix_salary_range_swap",
        "columns": [min_col, max_col],
        "n_affected": n_affected,
        "reason": "min_salary > max_salary là lỗi nhập liệu (đảo cột), xử lý bằng cách hoán đổi giá trị.",
    }
    return df, report


def flag_zero_negative_salary(
    df: pd.DataFrame,
    cols: Tuple[str, ...] = ("min_salary", "med_salary", "max_salary", "normalized_salary"),
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = df.copy()
    total_affected = 0
    affected_per_col = {}
    for col in cols:
        if col not in df.columns:
            continue
        mask = df[col].notna() & (df[col] <= 0)
        n = int(mask.sum())
        total_affected += n
        affected_per_col[col] = n
        if n > 0:
            df.loc[mask, col] = np.nan
    report = {
        "action": "flag_zero_negative_salary_as_nan",
        "columns": list(cols),
        "n_affected": total_affected,
        "details": affected_per_col,
        "reason": "Lương <= 0 là giá trị không hợp lệ (Data Error), gán NaN để không làm sai lệch phân phối và mô hình.",
    }
    return df, report


def fix_salary_units(
    df: pd.DataFrame,
    min_col: str = "min_salary",
    max_col: str = "max_salary",
    med_col: str = "med_salary",
    pay_period_col: str = "pay_period",
    norm_col: str = "normalized_salary",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = df.copy()
    if pay_period_col not in df.columns:
        return df, {"action": "fix_salary_units", "n_affected": 0, "reason": "pay_period column missing"}
    has_norm = norm_col in df.columns
    # Chuẩn hóa tạm thời để so sánh chuẩn xác
    period_s = df[pay_period_col].astype(str).str.strip().str.upper()
    # 1. Trường hợp HOURLY nhưng lương >= 1000
    hourly_mask = (period_s == "HOURLY") & (
        (df[min_col] >= 1000)
        | (df[max_col] >= 1000)
        | (df[med_col] >= 1000)
    )
    n_hourly_fixed = int(hourly_mask.sum())
    if n_hourly_fixed > 0:
        df.loc[hourly_mask, pay_period_col] = "YEARLY"
        if has_norm:
            # Recalculate normalized salary for YEARLY
            # Base yearly = med_salary if present else average of min & max
            base = df.loc[hourly_mask, med_col]
            avg_min_max = (df.loc[hourly_mask, min_col].fillna(df.loc[hourly_mask, max_col]) +
                           df.loc[hourly_mask, max_col].fillna(df.loc[hourly_mask, min_col])) / 2.0
            base = base.fillna(avg_min_max)
            df.loc[hourly_mask, norm_col] = base
    # 2. Trường hợp YEARLY nhưng lương <= 150 (chỉ áp dụng khi lương > 0)
    yearly_mask = (period_s == "YEARLY") & (
        ((df[max_col] > 0) & (df[max_col] <= 150))
        | ((df[med_col] > 0) & (df[med_col] <= 150))
        | ((df[min_col] > 0) & (df[min_col] <= 150))
    )
    n_yearly_fixed = int(yearly_mask.sum())
    if n_yearly_fixed > 0:
        df.loc[yearly_mask, pay_period_col] = "HOURLY"
        if has_norm:
            base = df.loc[yearly_mask, med_col]
            avg_min_max = (df.loc[yearly_mask, min_col].fillna(df.loc[yearly_mask, max_col]) +
                           df.loc[yearly_mask, max_col].fillna(df.loc[yearly_mask, min_col])) / 2.0
            base = base.fillna(avg_min_max)
            df.loc[yearly_mask, norm_col] = base * 2080.0
    report = {
        "action": "fix_salary_units",
        "n_hourly_to_yearly": n_hourly_fixed,
        "n_yearly_to_hourly": n_yearly_fixed,
        "n_total_affected": n_hourly_fixed + n_yearly_fixed,
        "reason": "Lệch đơn vị thời gian (nhập lương năm vào ô Hourly hoặc ngược lại). Sửa lại pay_period và recalculate normalized_salary.",
    }
    return df, report


def standardize_categorical(
    df: pd.DataFrame, col: str, mapping: Optional[Dict[str, str]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = df.copy()
    if col not in df.columns:
        return df, {"action": "standardize_categorical", "column": col, "n_affected": 0}
    before_unique = int(df[col].nunique(dropna=True))
    s = df[col].astype(str).str.strip().str.upper()
    s = s.replace({"NAN": np.nan, "NONE": np.nan, "": np.nan})
    if mapping:
        s = s.replace(mapping)
    df[col] = s
    after_unique = int(df[col].nunique(dropna=True))
    report = {
        "action": "standardize_categorical",
        "column": col,
        "n_unique_before": before_unique,
        "n_unique_after": after_unique,
    }
    return df, report


def clean_text_column(
    df: pd.DataFrame, col: str
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = df.copy()
    if col not in df.columns:
        return df, {"action": "clean_text_column", "column": col, "n_empty_or_whitespace_found": 0}
    # Đếm số dòng có khoảng trắng thừa đầu cuối hoặc rỗng
    str_series = df[col].dropna().astype(str)
    n_space_issues = int(((str_series != str_series.str.strip()) | (str_series.str.strip() == "")).sum())
    cleaned = (
        df[col]
        .dropna()
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    cleaned = cleaned.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    df[col] = cleaned
    report = {
        "action": "clean_text_column",
        "column": col,
        "n_empty_or_whitespace_found": n_space_issues,
    }
    return df, report


def detect_outliers_iqr(df: pd.DataFrame, col: str, k: float = 1.5) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    series = pd.to_numeric(df[col], errors="coerce")
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return (series < lower) | (series > upper)


def cast_dtype(
    df: pd.DataFrame, col: str, target_dtype: Any
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = df.copy()
    if col not in df.columns:
        return df, {"action": "cast_dtype", "column": col, "n_new_nan_from_coercion": 0}
    before_na = int(df[col].isna().sum())
    if target_dtype in (int, float, "float64", "int64"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    elif target_dtype == "datetime":
        df[col] = pd.to_datetime(df[col], errors="coerce")
    else:
        df[col] = df[col].astype(target_dtype, errors="ignore")
    after_na = int(df[col].isna().sum())
    report = {
        "action": "cast_dtype",
        "column": col,
        "target_dtype": str(target_dtype),
        "n_new_nan_from_coercion": after_na - before_na,
    }
    return df, report


def oversample_target_distribution(
    df: pd.DataFrame,
    target_col: str = "formatted_experience_level",
    random_state: int = 42,
    drop_na_target: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = df.copy()
    if target_col not in df.columns:
        return df, {
            "action": "oversample_target_distribution",
            "target_col": target_col,
            "status": "skipped",
            "reason": f"Column '{target_col}' not found in DataFrame",
        }
    # Phân tách dữ liệu hợp lệ và missing
    if drop_na_target:
        valid_mask = df[target_col].notna()
        df_valid = df[valid_mask].copy()
    else:
        df_valid = df.copy()
    if len(df_valid) == 0:
        return df, {
            "action": "oversample_target_distribution",
            "target_col": target_col,
            "status": "skipped",
            "reason": "No valid target rows found",
        }
    before_counts = df_valid[target_col].value_counts().to_dict()
    max_count = max(before_counts.values())
    # Thực hiện Oversampling bằng cách resample từng nhóm lên max_count
    resampled_groups = []
    for _, group in df_valid.groupby(target_col):
        if len(group) < max_count:
            sampled_group = group.sample(n=max_count, replace=True, random_state=random_state)
        else:
            sampled_group = group
        resampled_groups.append(sampled_group)
    df_resampled = (
        pd.concat(resampled_groups, axis=0)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )
    after_counts = df_resampled[target_col].value_counts().to_dict()
    report = {
        "action": "oversample_target_distribution",
        "target_col": target_col,
        "n_before": len(df_valid),
        "n_after": len(df_resampled),
        "n_added": len(df_resampled) - len(df_valid),
        "distribution_before": before_counts,
        "distribution_after": after_counts,
        "reason": "Cân bằng phân bố biến mục tiêu (target distribution) để tránh thiên lệch mô hình.",
    }
    return df_resampled, report


current_dir = Path(__file__).resolve().parent


if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


logger = logging.getLogger(__name__)


def clean_companies(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
    """Làm sạch bảng companies.csv"""
    logs = []
    logger.info("Cleaning companies...")
    # 1. Drop duplicates
    df_clean, r = drop_duplicate_records(df, subset=["company_id"])
    logs.append({**r, "table": "companies"})
    # 2. Text cleaning
    for col in ["name", "description", "state", "city", "address"]:
        if col in df_clean.columns:
            df_clean, r = clean_text_column(df_clean, col)
            logs.append({**r, "table": "companies"})
    # 3. Country categorical standardization
    if "country" in df_clean.columns:
        df_clean, r = standardize_categorical(df_clean, "country")
        logs.append({**r, "table": "companies"})
    # company_size: Giữ nguyên (1-7 là mã hợp lệ, Genuine Extreme Value)
    return df_clean, logs


def clean_salaries(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
    """Làm sạch bảng salaries.csv"""
    logs = []
    logger.info("Cleaning salaries...")
    # 1. Drop duplicates
    df_clean, r = drop_duplicate_records(df, subset=["salary_id"])
    logs.append({**r, "table": "salaries"})
    # 2. Standardize categories first
    for col in ["pay_period", "currency", "compensation_type"]:
        if col in df_clean.columns:
            df_clean, r = standardize_categorical(df_clean, col)
            logs.append({**r, "table": "salaries"})
    # 3. Fix inverted range
    df_clean, r = fix_salary_range(df_clean, min_col="min_salary", max_col="max_salary")
    logs.append({**r, "table": "salaries"})
    # 4. Flag <= 0
    df_clean, r = flag_zero_negative_salary(df_clean, cols=("min_salary", "med_salary", "max_salary"))
    logs.append({**r, "table": "salaries"})
    # 5. Fix units
    df_clean, r = fix_salary_units(df_clean, min_col="min_salary", max_col="max_salary", med_col="med_salary", pay_period_col="pay_period")
    logs.append({**r, "table": "salaries"})
    return df_clean, logs


def clean_postings(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
    """Làm sạch bảng chính postings.csv"""
    logs = []
    logger.info("Cleaning postings...")
    # 1. Drop duplicates by job_id
    df_clean, r = drop_duplicate_records(df, subset=["job_id"])
    logs.append({**r, "table": "postings"})
    # 2. Drop missing required fields (job_id, title, description)
    df_clean, r = drop_missing_required(df_clean, subset=["job_id", "title", "description"])
    logs.append({**r, "table": "postings"})
    # 3. Standardize categorical columns
    for col in ["work_type", "formatted_work_type", "formatted_experience_level", "pay_period", "currency", "compensation_type"]:
        if col in df_clean.columns:
            df_clean, r = standardize_categorical(df_clean, col)
            logs.append({**r, "table": "postings"})
    # 4. Flag <= 0 salary
    salary_cols = ("min_salary", "med_salary", "max_salary", "normalized_salary")
    df_clean, r = flag_zero_negative_salary(df_clean, cols=salary_cols)
    logs.append({**r, "table": "postings"})
    # 5. Fix inverted salary range
    df_clean, r = fix_salary_range(df_clean, min_col="min_salary", max_col="max_salary")
    logs.append({**r, "table": "postings"})
    # 6. Fix salary unit mismatches & recalculate normalized_salary
    df_clean, r = fix_salary_units(
        df_clean,
        min_col="min_salary",
        max_col="max_salary",
        med_col="med_salary",
        pay_period_col="pay_period",
        norm_col="normalized_salary",
    )
    logs.append({**r, "table": "postings"})
    # 7. Clean text columns
    for col in ["title", "description", "skills_desc", "company_name", "location"]:
        if col in df_clean.columns:
            df_clean, r = clean_text_column(df_clean, col)
            logs.append({**r, "table": "postings"})
    # 8. remote_allowed: chuẩn hóa NaN thành 0, 1.0 thành 1
    if "remote_allowed" in df_clean.columns:
        df_clean["remote_allowed"] = df_clean["remote_allowed"].fillna(0).astype(int)
        logs.append({
            "action": "fill_remote_allowed",
            "table": "postings",
            "column": "remote_allowed",
            "reason": "Chuyển NaN thành 0 (không phải remote hoàn toàn), 1.0 thành 1.",
        })
    return df_clean, logs


def clean_bridge_table(df: pd.DataFrame, table_name: str, key_cols: List[str]) -> Tuple[pd.DataFrame, List[Dict]]:
    """Làm sạch các bảng quan hệ / phụ"""
    logs = []
    logger.info(f"Cleaning {table_name}...")
    df_clean, r = drop_duplicate_records(df, subset=key_cols)
    logs.append({**r, "table": table_name})
    for col in df_clean.columns:
        if df_clean[col].dtype == object:
            df_clean, r = clean_text_column(df_clean, col)
            logs.append({**r, "table": table_name})
    return df_clean, logs


def run_data_cleaning(raw_dir: Path, processed_dir: Path, balance_target: bool = False):
    """Chạy toàn bộ pipeline làm sạch và lưu trữ kết quả"""
    processed_dir.mkdir(parents=True, exist_ok=True)
    all_logs = []
    before_counts = {}
    after_counts = {}
    # 1. Companies
    companies_path = raw_dir / "companies.csv"
    if companies_path.exists():
        companies = pd.read_csv(companies_path)
        before_counts["companies"] = len(companies)
        companies_clean, logs = clean_companies(companies)
        after_counts["companies"] = len(companies_clean)
        companies_clean.to_csv(processed_dir / "companies_clean.csv", index=False)
        all_logs.extend(logs)
    # 2. Company Industries
    ci_path = raw_dir / "company_industries.csv"
    if ci_path.exists():
        ci = pd.read_csv(ci_path)
        before_counts["company_industries"] = len(ci)
        ci_clean, logs = clean_bridge_table(ci, "company_industries", ["company_id", "industry"])
        after_counts["company_industries"] = len(ci_clean)
        ci_clean.to_csv(processed_dir / "company_industries_clean.csv", index=False)
        all_logs.extend(logs)
    # 3. Company Specialities
    cs_path = raw_dir / "company_specialities.csv"
    if cs_path.exists():
        cs = pd.read_csv(cs_path)
        before_counts["company_specialities"] = len(cs)
        cs_clean, logs = clean_bridge_table(cs, "company_specialities", ["company_id", "speciality"])
        after_counts["company_specialities"] = len(cs_clean)
        cs_clean.to_csv(processed_dir / "company_specialities_clean.csv", index=False)
        all_logs.extend(logs)
    # 4. Job Industries
    ji_path = raw_dir / "job_industries.csv"
    if ji_path.exists():
        ji = pd.read_csv(ji_path)
        before_counts["job_industries"] = len(ji)
        ji_clean, logs = clean_bridge_table(ji, "job_industries", ["job_id", "industry_id"])
        after_counts["job_industries"] = len(ji_clean)
        ji_clean.to_csv(processed_dir / "job_industries_clean.csv", index=False)
        all_logs.extend(logs)
    # 5. Job Skills
    js_path = raw_dir / "job_skills.csv"
    if js_path.exists():
        js = pd.read_csv(js_path)
        before_counts["job_skills"] = len(js)
        js_clean, logs = clean_bridge_table(js, "job_skills", ["job_id", "skill_abr"])
        after_counts["job_skills"] = len(js_clean)
        js_clean.to_csv(processed_dir / "job_skills_clean.csv", index=False)
        all_logs.extend(logs)
    # 6. Benefits
    bf_path = raw_dir / "benefits.csv"
    if bf_path.exists():
        bf = pd.read_csv(bf_path)
        before_counts["benefits"] = len(bf)
        bf_clean, logs = clean_bridge_table(bf, "benefits", ["job_id", "type"])
        after_counts["benefits"] = len(bf_clean)
        bf_clean.to_csv(processed_dir / "benefits_clean.csv", index=False)
        all_logs.extend(logs)
    # 7. Salaries
    salaries_path = raw_dir / "salaries.csv"
    if salaries_path.exists():
        salaries = pd.read_csv(salaries_path)
        before_counts["salaries"] = len(salaries)
        salaries_clean, logs = clean_salaries(salaries)
        after_counts["salaries"] = len(salaries_clean)
        salaries_clean.to_csv(processed_dir / "salaries_clean.csv", index=False)
        all_logs.extend(logs)
    # 8. Postings (bảng chính lớn nhất)
    postings_path = raw_dir / "postings.csv"
    if postings_path.exists():
        logger.info("Loading postings.csv...")
        postings = pd.read_csv(postings_path, low_memory=False)
        before_counts["postings"] = len(postings)
        postings_clean, logs = clean_postings(postings)
        after_counts["postings"] = len(postings_clean)
        logger.info("Saving postings_clean.csv...")
        postings_clean.to_csv(processed_dir / "postings_clean.csv", index=False)
        all_logs.extend(logs)
        # 8b. Cân bằng nhãn mục tiêu (Oversampling Target Distribution) nếu được yêu cầu
        if balance_target and "formatted_experience_level" in postings_clean.columns:
            logger.info("Balancing target distribution for formatted_experience_level (Oversampling)...")
            postings_balanced, sample_log = oversample_target_distribution(
                postings_clean, target_col="formatted_experience_level", random_state=42
            )
            balanced_path = processed_dir / "postings_balanced.csv"
            postings_balanced.to_csv(balanced_path, index=False)
            logger.info(f"Saved balanced postings to {balanced_path} ({len(postings_balanced)} records)")
            all_logs.append({**sample_log, "table": "postings_balanced"})
            after_counts["postings_balanced"] = len(postings_balanced)
    # Lưu cleaning log
    cleaning_log_df = pd.DataFrame(all_logs)
    cleaning_log_df.to_csv(processed_dir / "cleaning_log.csv", index=False)
    logger.info(f"Saved cleaning log with {len(cleaning_log_df)} entries to {processed_dir / 'cleaning_log.csv'}")
    # In và lưu bảng tóm tắt
    summary = pd.DataFrame({
        "Table": list(before_counts.keys()),
        "Records Before": list(before_counts.values()),
        "Records After": [after_counts.get(k, 0) for k in before_counts.keys()],
    })
    summary["Records Removed"] = summary["Records Before"] - summary["Records After"]
    summary.to_csv(processed_dir / "cleaning_summary.csv", index=False)
    logger.info(f"Saved cleaning summary to {processed_dir / 'cleaning_summary.csv'}")
    print("\n" + "=" * 50)
    print("PIPELINE SUMMARY: BEFORE vs AFTER CLEANING")
    print("=" * 50)
    print(summary.to_string(index=False))
    print("=" * 50)


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "description" not in df.columns and "snippet" in df.columns:
        df["description"] = df["snippet"]

    for col in ("title", "description", "skills_desc"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    df["title_length"] = df["title"].str.len()
    df["description_length"] = (df["description"].str.len())
    df["skills_length"] = (df["skills_desc"].str.len())
    df["description_word_count"] = (df["description"].str.split().str.len())
    df["skills_word_count"] = (df["skills_desc"].str.split().str.len())
    df["combined_text"] = (df["title"] + " " + df["description"] + " " + df["skills_desc"])
    return df


SKILL_VOCABULARY = [
    # Programming
    "python",
    "java",
    "c++",
    "c#",
    "javascript",
    "typescript",
    "r",
    "sql",

    # Data Science / ML
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "data science",
    "data analysis",
    "natural language processing",
    "computer vision",

    # ML frameworks
    "pytorch",
    "tensorflow",
    "keras",
    "scikit-learn",

    # Data
    "pandas",
    "numpy",
    "spark",
    "hadoop",

    # Cloud / DevOps
    "aws",
    "azure",
    "gcp",
    "docker",
    "kubernetes",

    # Databases
    "mysql",
    "postgresql",
    "mongodb",
    "oracle",

    # Visualization / BI
    "tableau",
    "power bi",
    "excel",

    # Software / Web
    "git",
    "github",
    "linux",
    "html",
    "css",
    "react",
    "node.js"
]


def _combine_text(df: pd.DataFrame) -> pd.Series:
    return (
        df["title"].fillna("").astype(str)
        + " "
        + df["description"].fillna("").astype(str)
        + " "
        + df["skills_desc"].fillna("").astype(str)
    ).str.lower()


def extract_skill_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    text = _combine_text(df)
    skill_columns = []
    for skill in SKILL_VOCABULARY:
        column_name = (
            "skill_"
            + re.sub(r"[^a-zA-Z0-9]+", "_", skill)
            .strip("_")
            .lower()
        )

        # Avoid duplicate feature names
        if column_name in skill_columns:
            continue

        df[column_name] = (
            text.str.contains(
                re.escape(skill),
                regex=True,
                na=False,
            )
            .astype(int)
        )

        skill_columns.append(column_name)

    # Total number of detected skills
    df["skill_count"] = df[skill_columns].sum(axis=1)

    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # =========================
    # PARSE POSTING TIME
    # =========================
    if "listed_time" not in df.columns:
        return df
    listed_time = df["listed_time"]
    if pd.api.types.is_numeric_dtype(listed_time):
        posting_time = pd.to_datetime(listed_time, unit="ms", errors="coerce", utc=True)
    else:
        posting_time = pd.to_datetime(listed_time, errors="coerce", utc=True)

    # =========================
    # CALENDAR FEATURES
    # =========================
    df["posting_year"] = posting_time.dt.year
    df["posting_month"] = posting_time.dt.month
    df["posting_day"] = posting_time.dt.day
    df["posting_dayofweek"] = posting_time.dt.dayofweek
    df["posting_quarter"] = posting_time.dt.quarter

    # =========================
    # TIME OF DAY
    # =========================
    df["posting_hour"] = posting_time.dt.hour

    # =========================
    # WEEKEND FEATURE
    # =========================
    df["is_weekend"] = (posting_time.dt.dayofweek >= 5).astype(int)

    return df


ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "data" / "processed" / "postings_clean.csv"


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    numeric_features = [
        "max_salary",
        "med_salary",
        "min_salary",
        "normalized_salary",

        # Text statistics
        "title_length",
        "description_length",
        "skills_length",
        "description_word_count",
        "skills_word_count",

        # Temporal features
        "posting_year",
        "posting_month",
        "posting_day",
        "posting_dayofweek",
        "posting_quarter",
        "posting_hour",
        "is_weekend",

        # Skill count
        "skill_count"
    ]

    categorical_features = [
        "location",
        "pay_period",
        "formatted_work_type",
        "application_type",
        "work_type",
        "currency",
        "compensation_type"]

    binary_features = [
        "remote_allowed",
        "sponsored"]

    skill_features = [
        column
        for column in x.columns
        if column.startswith("skill_")
        and column != "skill_count"]
    numeric_features.extend(skill_features)

    all_structured_features = (
        numeric_features + categorical_features + binary_features)

    missing_features = [
        column
        for column in all_structured_features
        if column not in x.columns
    ]

    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")

    missing_text = [column for column in ("title", "description") if column not in x.columns]
    if missing_text:
        raise ValueError(f"Missing required text features: {missing_text}")

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    ])

    binary_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])

    title_vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.98,
        max_features=5000,
        dtype=np.float32,
    )
    description_vectorizer = TfidfVectorizer(
        stop_words="english",
        min_df=10,
        max_df=0.95,
        max_features=20000,
        dtype=np.float32,
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
            ("binary", binary_pipeline, binary_features),
            ("title_text", title_vectorizer, "title"),
            ("description_text", description_vectorizer, "description"),
        ]
    )


def _build_company_groups(data: pd.DataFrame) -> pd.Series:
    groups = pd.Series(pd.NA, index=data.index, dtype="string")

    if "company_id" in data.columns:
        company_ids = data["company_id"].astype("string").str.strip()
        company_ids = company_ids.mask(
            company_ids.str.lower().isin(["", "nan", "none", "<na>"])
        )
        groups = company_ids.map(
            lambda value: f"company_id:{value}" if pd.notna(value) else pd.NA
        )

    if "company_name" in data.columns:
        company_names = data["company_name"].astype("string").str.strip().str.lower()
        company_names = company_names.mask(
            company_names.isin(["", "nan", "none", "<na>", "unknown"])
        )
        groups = groups.fillna(
            company_names.map(
                lambda value: f"company_name:{value}" if pd.notna(value) else pd.NA
            )
        )

    if "job_id" in data.columns:
        job_ids = data["job_id"].astype("string").str.strip()
        job_ids = job_ids.mask(job_ids.str.lower().isin(["", "nan", "none", "<na>"]))
        groups = groups.fillna(job_ids.map(lambda value: f"job_id:{value}"))

    row_groups = pd.Series(data.index.astype(str), index=data.index).radd("row:")
    return groups.fillna(row_groups)


def build_features(data_path=DATA_PATH, return_groups=False):
    data_path = Path(data_path)
    data = pd.read_csv(data_path, low_memory=False)
    print("Data loaded successfully!")
    print("Shape:", data.shape)

    target = "formatted_experience_level"
    data = data.dropna(subset=[target]).copy()
    groups = _build_company_groups(data)
    x = data.drop(columns=[target]).copy()
    y = data[target].astype(str)
    print("\nTarget distribution:")
    print(y.value_counts())

    x = add_text_features(x)
    x = extract_skill_features(x)
    x = add_temporal_features(x)

    columns_to_drop = [
        "job_id", "company_id", "zip_code", "fips", "salary_id",
        "original_listed_time", "listed_time", "expiry", "closed_time",
        "job_posting_url", "application_url",
        "skills_desc"
    ]
    x = x.drop(columns=[column for column in columns_to_drop if column in x.columns])

    splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
    train_indices, test_indices = next(splitter.split(x, y, groups=groups))
    result = (
        x.iloc[train_indices],
        x.iloc[test_indices],
        y.iloc[train_indices],
        y.iloc[test_indices],
    )
    if return_groups:
        return (*result, groups.iloc[train_indices], groups.iloc[test_indices])
    return result


ROOT_DIR = Path(__file__).resolve().parent


if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


MODELS_DIR = ROOT_DIR / "models"


REPORTS_DIR = ROOT_DIR / "reports"


MODELS_DIR.mkdir(parents=True, exist_ok=True)


REPORTS_DIR.mkdir(parents=True, exist_ok=True)


DATA_SPLIT_PATH = MODELS_DIR / "data_split.pkl"


LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"


MODEL_COMPARISON_PATH = MODELS_DIR / "model_comparison.csv"


BEST_MODEL_INFO_PATH = MODELS_DIR / "best_model_info.json"


BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"


TUNING_RESULTS_PATH = MODELS_DIR / "tuning_results.json"


DATA_SPLIT_VERSION = 3


def _get_split_signature():
    data_stat = DATA_PATH.stat()
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return {
        "data_size": data_stat.st_size,
        "data_mtime_ns": data_stat.st_mtime_ns,
        "feature_code": {"main.py": source_hash},
    }


def get_train_test_data(force_rebuild=False, include_groups=False):
    split_signature = _get_split_signature()

    if DATA_SPLIT_PATH.exists() and not force_rebuild:
        print(f"Loading cached train/test split from {DATA_SPLIT_PATH}")
        cache = joblib.load(DATA_SPLIT_PATH)
        if (
            cache.get("version") == DATA_SPLIT_VERSION
            and cache.get("signature") == split_signature
        ):
            result = (
                cache["x_train"],
                cache["x_test"],
                cache["y_train"],
                cache["y_test"],
                cache["label_encoder"],
            )
            if include_groups:
                return (*result, cache["groups_train"])
            return result
        print("Cached split is stale or uses an old format; rebuilding it.")

    print("No cached split found. Building features via Member 3's pipeline (build_features)...")
    x_train, x_test, y_train, y_test, groups_train, groups_test = build_features(
        return_groups=True
    )

    # Fit label encoder on TRAIN labels only, then just transform test (no leakage)
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)

    joblib.dump(
        {
            "x_train": x_train,
            "x_test": x_test,
            "y_train": y_train_enc,
            "y_test": y_test_enc,
            "label_encoder": label_encoder,
            "groups_train": groups_train,
            "groups_test": groups_test,
            "version": DATA_SPLIT_VERSION,
            "signature": split_signature,
        },
        DATA_SPLIT_PATH,
    )
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    print(f"Cached train/test split -> {DATA_SPLIT_PATH}")

    result = x_train, x_test, y_train_enc, y_test_enc, label_encoder
    if include_groups:
        return (*result, groups_train)
    return result


def compute_metrics(y_true, y_pred, prefix=""):
    """Accuracy / Precision / Recall / Macro-F1 / Weighted-F1 (see README 9.9)."""
    return {
        f"{prefix}accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        f"{prefix}recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        f"{prefix}f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        f"{prefix}f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


class BoundedSelectKBest(SelectKBest):
    def fit(self, X, y=None):
        self.k = min(self.k, X.shape[1])
        return super().fit(X, y)


def build_logistic_pipeline(
    x_reference,
    random_state=42,
    apply_feature_selection=False,
    k=1000,
    **logreg_kwargs,
):

    params = dict(max_iter=2000, class_weight="balanced", random_state=random_state)
    params.update(logreg_kwargs)
    steps = [("preprocessor", build_preprocessor(x_reference))]
    if apply_feature_selection:
        steps.append(("selector", BoundedSelectKBest(score_func=f_classif, k=k)))
    steps.append(("clf", LogisticRegression(**params)))
    return Pipeline(steps)


RANDOM_STATE = 42


CV_FOLDS = 3


def build_candidate_models(x_reference, apply_feature_selection=True, k=1000):

    return {
        "logistic_regression_baseline": build_logistic_pipeline(
            x_reference,
            random_state=RANDOM_STATE,
            apply_feature_selection=apply_feature_selection,
            k=k,
        ),
    }


def print_class_weight_summary(x_train, y_train, label_encoder):
    train_labels = label_encoder.inverse_transform(y_train)
    counts = pd.Series(train_labels).value_counts().sort_index()
    print("\nTraining class distribution:")
    print(counts.to_string())
    print("\nClass imbalance handling: LogisticRegression(class_weight='balanced')")
    print(f"No rows are duplicated; training rows: {len(x_train)}")


def train_all_models(apply_feature_selection=True, k=1000):
    x_train, x_test, y_train, y_test, label_encoder, groups_train = get_train_test_data(
        include_groups=True
    )
    print(f"x_train: {x_train.shape}, x_test: {x_test.shape}")
    print(f"Classes ({len(label_encoder.classes_)}): {list(label_encoder.classes_)}")
    print_class_weight_summary(x_train, y_train, label_encoder)

    models = build_candidate_models(
        x_train,
        apply_feature_selection=apply_feature_selection,
        k=k,
    )
    cv = StratifiedGroupKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )

    results = []

    for name, model in models.items():
        print(f"\n=== Training: {name} ===")
        start = time.time()

        # Cross Validation (TRAIN only, never touches x_test) ----
        cv_scores = cross_val_score(
            model,
            x_train,
            y_train,
            groups=groups_train,
            cv=cv,
            scoring="f1_macro",
            n_jobs=1,
        )

        # ---- Fit on the full training set ----
        model.fit(x_train, y_train)

        train_time = time.time() - start

        # Bias & Variance: compare train vs test performance ----
        train_pred = model.predict(x_train)
        test_pred = model.predict(x_test)

        metrics = {"model_name": name}
        metrics.update(compute_metrics(y_train, train_pred, prefix="train_"))
        metrics.update(compute_metrics(y_test, test_pred, prefix="test_"))
        metrics["cv_f1_macro_mean"] = cv_scores.mean()
        metrics["cv_f1_macro_std"] = cv_scores.std()
        metrics["training_time_sec"] = train_time

        results.append(metrics)

        model_path = MODELS_DIR / f"{name}.pkl"
        joblib.dump(model, model_path)
        print(f"Saved model -> {model_path}")
        print(
            f"CV Macro-F1: {metrics['cv_f1_macro_mean']:.4f} (+/- {metrics['cv_f1_macro_std']:.4f}) | "
            f"Test Macro-F1: {metrics['test_f1_macro']:.4f} | Test Acc: {metrics['test_accuracy']:.4f}"
        )

    #  Model comparison table (1 row - Logistic Regression only) ----
    comparison_df = pd.DataFrame(results).set_index("model_name")
    comparison_df = comparison_df.sort_values("cv_f1_macro_mean", ascending=False)
    comparison_df.to_csv(MODEL_COMPARISON_PATH)
    print(f"\nModel comparison table saved -> {MODEL_COMPARISON_PATH}")
    print(comparison_df[["cv_f1_macro_mean", "test_f1_macro", "test_accuracy", "training_time_sec"]])

    # ---- Hand the (only) candidate off to tune.py ----
    best_name = comparison_df.index[0]
    best_info = {
        "best_model_name": best_name,
        "cv_f1_macro_mean": float(comparison_df.loc[best_name, "cv_f1_macro_mean"]),
        "test_f1_macro": float(comparison_df.loc[best_name, "test_f1_macro"]),
        "apply_feature_selection": apply_feature_selection,
        "k": k,
    }
    save_json(best_info, BEST_MODEL_INFO_PATH)
    print(f"\nBest candidate before tuning: {best_name} -> saved to {BEST_MODEL_INFO_PATH}")
    print("Next step: python -m src.models.tune")

    return comparison_df


RANDOM_STATE = 42


CV_FOLDS = 3


N_ITER = 7


PARAM_GRIDS = {
    "logistic_regression_baseline": {
        "clf__C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
        "clf__solver": ["lbfgs"],
    },
}


def tune_best_model():
    if not BEST_MODEL_INFO_PATH.exists():
        raise FileNotFoundError(
            f"{BEST_MODEL_INFO_PATH} not found. Run train.py first: python -m src.models.train"
        )

    with open(BEST_MODEL_INFO_PATH, "r", encoding="utf-8") as f:
        best_info = json.load(f)

    model_name = best_info["best_model_name"]
    print(f"Tuning best candidate from train.py: {model_name}")

    if model_name not in PARAM_GRIDS:
        raise ValueError(
            f"No hyperparameter grid defined for '{model_name}'. "
            f"Add one to PARAM_GRIDS in src/models/tune.py."
        )


    x_train, x_test, y_train, y_test, label_encoder, groups_train = get_train_test_data(
        include_groups=True
    )

    base_model = build_logistic_pipeline(
        x_train,
        random_state=RANDOM_STATE,
        apply_feature_selection=best_info.get("apply_feature_selection", True),
        k=best_info.get("k", 1000),
    )
    param_grid = PARAM_GRIDS[model_name]
    cv = StratifiedGroupKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=N_ITER,
        scoring="f1_macro",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1,
        pre_dispatch=1,
        verbose=1,
        refit=True,
        error_score="raise",
    )

    print("Running RandomizedSearchCV (this can take a while)...")
    search.fit(x_train, y_train, groups=groups_train)

    print(f"\nBest CV Macro-F1: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")

    # ---- Save the tuned model as THE official best model used downstream ----
    joblib.dump(search.best_estimator_, BEST_MODEL_PATH)
    print(f"Tuned best model saved -> {BEST_MODEL_PATH}")

    tuning_results = {
        "model_name": model_name,
        "best_cv_f1_macro": float(search.best_score_),
        "best_params": search.best_params_,
        "cv_folds": CV_FOLDS,
        "n_iter": N_ITER,
    }
    save_json(tuning_results, TUNING_RESULTS_PATH)
    print(f"Tuning results saved -> {TUNING_RESULTS_PATH}")


    return search


matplotlib.use("Agg")


def evaluate_best_model():
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{BEST_MODEL_PATH} not found. Run tune.py first: python -m src.models.tune"
        )

    model = joblib.load(BEST_MODEL_PATH)
    x_train, x_test, y_train, y_test, label_encoder = get_train_test_data()

    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)

    #Bias & Variance ----
    train_metrics = compute_metrics(y_train, train_pred, prefix="train_")
    test_metrics = compute_metrics(y_test, test_pred, prefix="test_")

    print("=== Training performance ===")
    for k, v in train_metrics.items():
        print(f"{k}: {v:.4f}")

    print("\n=== Test performance ===")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")

    gap = train_metrics["train_f1_macro"] - test_metrics["test_f1_macro"]
    if train_metrics["train_f1_macro"] < 0.6 and test_metrics["test_f1_macro"] < 0.6:
        diagnosis = "High bias (underfitting): both train and test Macro-F1 are low."
    elif gap > 0.15:
        diagnosis = "High variance (overfitting): large gap between train and test Macro-F1."
    else:
        diagnosis = "Bias/variance look reasonably balanced."
    print(f"\nDiagnosis: {diagnosis}")

    # Full evaluation on the test set ----
    class_names = [str(c) for c in label_encoder.classes_]
    report = classification_report(
        y_test, test_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    print("\n=== Classification report (test) ===")
    print(classification_report(y_test, test_pred, target_names=class_names, zero_division=0))

    cm = confusion_matrix(y_test, test_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(REPORTS_DIR / "confusion_matrix.csv")

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (Test Set)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    print(f"\nConfusion matrix saved -> {REPORTS_DIR / 'confusion_matrix.png'}")


    final_estimator = model.steps[-1][1] if hasattr(model, "steps") else model

    importance_path = REPORTS_DIR / "feature_importance.csv"
    try:
        if hasattr(final_estimator, "feature_importances_"):
            importances = final_estimator.feature_importances_
            imp_df = pd.DataFrame({
                "feature_index": np.arange(len(importances)),
                "importance": importances,
            }).sort_values("importance", ascending=False)
            imp_df.to_csv(importance_path, index=False)
            print(
                f"Feature importances saved -> {importance_path} "
                f"(indices map to the ColumnTransformer's transformed columns; "
                f"use model.named_steps['preprocessor'].get_feature_names_out() "
                f"to map indices back to real feature names)."
            )
        elif hasattr(final_estimator, "coef_"):
            coef_df = pd.DataFrame(
                final_estimator.coef_, columns=[f"feat_{i}" for i in range(final_estimator.coef_.shape[1])]
            )
            n_rows = coef_df.shape[0]
            coef_df.insert(0, "class", class_names[:n_rows] if n_rows > 1 else ["positive_class"])
            coef_df.to_csv(importance_path, index=False)
            print(f"Model coefficients saved -> {importance_path}")
        else:
            print("Selected model has no feature_importances_ / coef_ attribute; skipping interpretation export.")
    except Exception as e:
        print(f"Could not export feature importance: {e}")

    # ---- Save full evaluation summary ----
    summary = {
        **train_metrics,
        **test_metrics,
        "diagnosis": diagnosis,
        "classification_report": report,
    }
    save_json(summary, REPORTS_DIR / "evaluation_report.json")
    print(f"\nFull evaluation report saved -> {REPORTS_DIR / 'evaluation_report.json'}")


    return summary


COLUMNS_TO_DROP = [
    "job_id", "company_id", "zip_code", "fips", "salary_id",
    "original_listed_time", "listed_time", "expiry", "closed_time",
    "job_posting_url", "application_url",
]


NUMERIC_FEATURES = [
    "max_salary", "med_salary", "min_salary", "normalized_salary",
    "title_length", "description_length", "skills_length",
    "description_word_count", "skills_word_count", "posting_year", "posting_month",
    "posting_day", "posting_dayofweek", "posting_quarter", "posting_hour",
    "is_weekend", "skill_count",
]


CATEGORICAL_FEATURES = [
    "company_name", "location", "pay_period", "formatted_work_type",
    "application_type", "work_type", "currency", "compensation_type",
]


BINARY_FEATURES = ["remote_allowed", "sponsored"]


TARGET_COLUMN = "formatted_experience_level"


def _select_prediction_rows(df: pd.DataFrame, limit=100, random_state=42) -> pd.DataFrame:
    if limit is None or len(df) <= limit:
        return df
    if limit < 1:
        raise ValueError("limit must be a positive integer or None")
    return df.sample(n=limit, random_state=random_state).sort_index()


def prepare_new_data(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    aliases = {
        "company": "company_name",
        "salary_min_usd": "min_salary",
        "salary_max_usd": "max_salary",
        "salary_avg_usd": "normalized_salary",
        "apply_type": "application_type",
        "listed_at": "listed_time",
    }
    for source, target in aliases.items():
        if target not in df.columns and source in df.columns:
            df[target] = df[source]

    if "application_type" in df.columns:
        application_types = df["application_type"].astype("string").str.strip()
        application_type_aliases = {
            "external": "OffsiteApply",
            "linkedin easy apply": "SimpleOnsiteApply",
            "linkedin (complex)": "ComplexOnsiteApply",
            "unknown": "UnknownApply",
        }
        mapped_application_types = application_types.str.casefold().map(
            application_type_aliases
        )
        df["application_type"] = mapped_application_types.fillna(application_types)

    if "formatted_work_type" not in df.columns and "work_type" in df.columns:
        df["formatted_work_type"] = df["work_type"]
    if "work_type" in df.columns:
        work_type = df["work_type"].astype("string").str.strip().str.upper()
        df["work_type"] = work_type.str.replace(r"[\s-]+", "_", regex=True)
    if "formatted_work_type" in df.columns:
        formatted_work_type = df["formatted_work_type"].astype("string").str.strip().str.upper()
        df["formatted_work_type"] = formatted_work_type.str.replace(r"\s+", "-", regex=True)
    if "remote_allowed" not in df.columns and "remote" in df.columns:
        remote_values = df["remote"].astype(str).str.strip().str.lower()
        df["remote_allowed"] = remote_values.map(
            {"yes": 1, "true": 1, "1": 1, "no": 0, "false": 0, "0": 0}
        )

    for column in NUMERIC_FEATURES:
        if column not in df.columns:
            df[column] = np.nan
    for column in CATEGORICAL_FEATURES:
        if column not in df.columns:
            df[column] = "unknown"
    for column in BINARY_FEATURES:
        if column not in df.columns:
            df[column] = np.nan
        else:
            binary_values = df[column].astype(str).str.strip().str.lower()
            normalized = binary_values.map(
                {
                    "yes": 1,
                    "true": 1,
                    "1": 1,
                    "no": 0,
                    "false": 0,
                    "0": 0,
                }
            )
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(normalized)

    if "skills_desc" not in df.columns:
        df["skills_desc"] = ""

    target_aliases = [column for column in (TARGET_COLUMN, "experience_level") if column in df.columns]
    if target_aliases:
        df = df.drop(columns=target_aliases)

    df = add_text_features(df)
    df = extract_skill_features(df)
    df = add_temporal_features(df)

    df = df.drop(columns=[c for c in COLUMNS_TO_DROP if c in df.columns])
    return df


def predict(input_path: str, output_path: str, limit=100, random_state=42):
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{BEST_MODEL_PATH} not found. Run train.py then tune.py first:\n"
            f"  python -m src.models.train\n  python -m src.models.tune"
        )
    model = joblib.load(BEST_MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)

    raw_df = pd.read_csv(input_path, low_memory=False)
    total_records = len(raw_df)
    raw_df = _select_prediction_rows(raw_df, limit=limit, random_state=random_state)
    print(f"Loaded {len(raw_df)} of {total_records} records from {input_path}")

    x_new = prepare_new_data(raw_df)

    pred_encoded = model.predict(x_new)
    pred_labels = label_encoder.inverse_transform(pred_encoded)

    result = raw_df.copy()
    result["predicted_job_level"] = pred_labels

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_new)
        for i, class_name in enumerate(label_encoder.classes_):
            result[f"proba_{class_name}"] = proba[:, i]

    result.to_csv(output_path, index=False)
    print(f"Predictions saved -> {output_path}")
    print("\nPrediction distribution:")
    print(result["predicted_job_level"].value_counts())

    return result


ROOT_DIR = Path(__file__).resolve().parent


RAW_DIR = ROOT_DIR / "data" / "raw"


PROCESSED_DIR = ROOT_DIR / "data" / "processed"


MODELS_DIR = ROOT_DIR / "models"


REPORTS_DIR = ROOT_DIR / "reports"


MODELS_DIR.mkdir(parents=True, exist_ok=True)


REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def run_project_pipeline(
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
    print(f"Cross-validation: StratifiedGroupKFold ({CV_FOLDS} folds, grouped by company)")

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
    search = tune_best_model()
    print("\n✓ Hyperparameter tuning completed.")

    # ========================================================
    # STEP 5 — FINAL MODEL EVALUATION
    # ========================================================
    print("\n[5/7] FINAL MODEL EVALUATION")
    print("-" * 70)
    metrics = evaluate_best_model()
    print("\n✓ Final model evaluation completed.")

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
    # STEP 7 — FINAL OUTPUTS
    # ========================================================
    print("\n[7/7] FINAL OUTPUTS")
    print("-" * 70)
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


ROOT_DIR = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DS Job Recommendation project CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Clean raw datasets into the processed folder."
    )
    preprocess_parser.add_argument(
        "--raw-dir",
        type=Path,
        default=ROOT_DIR / "data" / "raw",
        help="Path to the raw data directory."
    )
    preprocess_parser.add_argument(
        "--processed-dir",
        type=Path,
        default=ROOT_DIR / "data" / "processed",
        help="Path to the processed data directory."
    )
    preprocess_parser.add_argument(
        "--balance-target",
        action="store_true",
        help="Create an oversampled dataset for the target column."
    )

    features_parser = subparsers.add_parser(
        "build-features",
        help="Create the feature matrix and train/test split."
    )
    features_parser.add_argument(
        "--data-path",
        type=Path,
        default=ROOT_DIR / "data" / "processed" / "postings_clean.csv",
        help="CSV file used to build the feature matrix."
    )
    features_parser.add_argument(
        "--return-groups",
        action="store_true",
        help="Return grouped split metadata alongside train/test arrays."
    )

    train_parser = subparsers.add_parser(
        "train",
        help="Run the ML training workflow only."
    )
    train_parser.add_argument(
        "--feature-selection",
        action="store_true",
        help="Apply feature selection during model training."
    )
    train_parser.add_argument(
        "--k",
        type=int,
        default=1000,
        help="Number of features to keep when feature selection is used."
    )

    predict_parser = subparsers.add_parser(
        "predict",
        help="Predict job levels for a new CSV file."
    )
    predict_parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the input CSV file used for prediction."
    )
    predict_parser.add_argument(
        "--output",
        type=Path,
        default=ROOT_DIR / "reports" / "predictions.csv",
        help="Path for the prediction output CSV."
    )
    predict_parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of rows to predict."
    )

    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Run the full end-to-end data and model workflow."
    )
    pipeline_parser.add_argument(
        "--feature-selection",
        action="store_true",
        help="Apply feature selection during model training."
    )
    pipeline_parser.add_argument(
        "--k",
        type=int,
        default=1000,
        help="Number of features to keep when feature selection is used."
    )
    pipeline_parser.add_argument(
        "--prediction-input",
        type=Path,
        help="Optional CSV file for prediction after training."
    )
    pipeline_parser.add_argument(
        "--prediction-output",
        type=Path,
        default=ROOT_DIR / "reports" / "predictions.csv",
        help="Output file for prediction results."
    )
    pipeline_parser.add_argument(
        "--prediction-limit",
        type=int,
        default=100,
        help="Maximum number of rows to predict."
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "preprocess":
        run_data_cleaning(
            raw_dir=args.raw_dir,
            processed_dir=args.processed_dir,
            balance_target=args.balance_target,
        )
        return

    if args.command == "build-features":
        build_features(data_path=args.data_path, return_groups=args.return_groups)
        return

    if args.command == "train":
        get_train_test_data(force_rebuild=True)
        train_all_models(
            apply_feature_selection=args.feature_selection,
            k=args.k,
        )
        print("Training completed.")
        return

    if args.command == "predict":
        predict(
            input_path=str(args.input),
            output_path=str(args.output),
            limit=args.limit,
        )
        return

    if args.command == "pipeline":
        run_project_pipeline(
            apply_feature_selection=args.feature_selection,
            k=args.k,
            prediction_input=str(args.prediction_input) if args.prediction_input else None,
            prediction_output=str(args.prediction_output),
            prediction_limit=args.prediction_limit,
        )
        return

    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
