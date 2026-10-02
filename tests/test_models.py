import argparse
import re
import sys

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.models import tune as tune_module
from src.models.utils import (
    ROOT_DIR,
    REPORTS_DIR,
    LABEL_ENCODER_PATH,
    MODEL_COMPARISON_PATH,
    BEST_MODEL_INFO_PATH,
    BEST_MODEL_PATH,
    TUNING_RESULTS_PATH,
)
from src.models.predict import _select_prediction_rows, prepare_new_data
from src.models.train import print_class_weight_summary
from src.models.utils import build_logistic_pipeline

DATA_PATH = ROOT_DIR / "data" / "processed" / "postings_clean.csv"
TARGET_COLUMN = "formatted_experience_level"

REQUIRED_ARTIFACTS = {
    "Complete fitted model pipeline (tune.py)": BEST_MODEL_PATH,
    "Label encoder": LABEL_ENCODER_PATH,
    "Model comparison table (train.py)": MODEL_COMPARISON_PATH,
    "Best model info (train.py)": BEST_MODEL_INFO_PATH,
    "Tuning results (tune.py)": TUNING_RESULTS_PATH,
}
def check_artifacts():
    print("=" * 70)
    print("BƯỚC 1: Kiểm tra các file artifact bắt buộc")
    print("=" * 70)
    all_ok = True
    for label, path in REQUIRED_ARTIFACTS.items():
        ok = path.exists()
        status = "OK" if ok else "THIẾU"
        print(f"[{status:6}] {label}\n         -> {path}")
        all_ok = all_ok and ok

    if not all_ok:
        print("\nThiếu artifact bắt buộc. Hãy chạy tuần tự trước:")
        print("  python -m src.models.train")
        print("  python -m src.models.tune")
        print("  python -m src.models.evaluate")
    return all_ok


def sample_and_predict(n=20, random_state=123):
    print("\n" + "=" * 70)
    print(f"BƯỚC 2: Lấy {n} dòng dữ liệu thật ngẫu nhiên từ postings_clean.csv để test dự đoán")
    print("=" * 70)

    if not DATA_PATH.exists():
        print(f"Không tìm thấy {DATA_PATH} - bỏ qua bước này.")
        return None

    data = pd.read_csv(DATA_PATH, low_memory=False)
    data = data.dropna(subset=[TARGET_COLUMN])
    sample = data.sample(n=min(n, len(data)), random_state=random_state).copy()

    actual_labels = sample[TARGET_COLUMN].astype(str).values

    model = joblib.load(BEST_MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)

    x_new = prepare_new_data(sample)
    pred_encoded = model.predict(x_new)
    pred_labels = label_encoder.inverse_transform(pred_encoded)

    result = pd.DataFrame({
        "actual": actual_labels,
        "predicted": pred_labels,
        "correct": actual_labels == pred_labels,
    })

    accuracy = result["correct"].mean()
    print(result.to_string(index=False))
    print(f"\nĐộ chính xác trên {len(result)} dòng mẫu: {accuracy:.2%}")

    out_path = REPORTS_DIR / "test_pipeline_sample_predictions.csv"
    result.to_csv(out_path, index=False)
    print(f"Đã lưu -> {out_path}")

    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Kiểm tra lại toàn bộ pipeline sau khi train/tune/evaluate xong.")
    parser.add_argument("--n", type=int, default=20, help="Số dòng dữ liệu thật lấy mẫu để test dự đoán.")
    args = parser.parse_args()

    artifacts_ok = check_artifacts()
    if not artifacts_ok:
        sys.exit(1)

    accuracy = sample_and_predict(n=args.n)

    print("\n" + "=" * 70)
    print("TỔNG KẾT")
    print("=" * 70)
    print("Kiểm tra artifact : PASS")
    if accuracy is not None:
        print(f"Test dự đoán mẫu  : PASS (accuracy mẫu = {accuracy:.2%})")
    else:
        print("Test dự đoán mẫu  : BỎ QUA (không tìm thấy postings_clean.csv)")
    print("\nPipeline hoạt động bình thường. Giờ có thể dùng cho dữ liệu mới thật sự bằng:")
    print("  python -m src.models.predict --input <file.csv> --output <out.csv>")


def test_tuning_budget_is_lightweight():
    assert tune_module.CV_FOLDS <= 3, "Training CV budget must stay lightweight for large datasets."
    assert tune_module.N_ITER <= 8, "RandomizedSearchCV iteration count must be capped for faster runs."


def test_logistic_pipeline_uses_class_weights_without_oversampling():
    feature_columns = [
        "max_salary", "med_salary", "min_salary", "normalized_salary",
        "title_length", "description_length", "skills_length",
        "description_word_count", "skills_word_count", "posting_year",
        "posting_month", "posting_day", "posting_dayofweek", "posting_quarter",
        "posting_hour", "is_weekend", "skill_count", "location", "pay_period",
        "formatted_work_type", "posting_domain", "application_type", "work_type",
        "currency", "compensation_type", "remote_allowed", "sponsored",
        "title", "description",
    ]
    model = build_logistic_pipeline(pd.DataFrame(columns=feature_columns))

    assert list(model.named_steps) == ["preprocessor", "clf"]
    assert model.named_steps["clf"].class_weight == "balanced"


def test_prediction_adapter_keeps_text_and_maps_annual_salary():
    raw = pd.DataFrame(
        {
            "title": ["Senior Data Analyst"],
            "snippet": ["Lead reporting and analytics"],
            "company": ["Example Co"],
            "work_type": ["Full-time"],
            "apply_type": ["LinkedIn Easy Apply"],
            "url": ["https://www.linkedin.com/jobs/view/123"],
            "salary_min_usd": [80000],
            "salary_max_usd": [100000],
            "salary_avg_usd": [90000],
            "experience_level": ["Mid-Senior level"],
        }
    )

    prepared = prepare_new_data(raw)

    assert prepared.loc[0, "title"] == "Senior Data Analyst"
    assert prepared.loc[0, "description"] == "Lead reporting and analytics"
    assert prepared.loc[0, "work_type"] == "FULL_TIME"
    assert prepared.loc[0, "formatted_work_type"] == "FULL-TIME"
    assert prepared.loc[0, "application_type"] == "SimpleOnsiteApply"
    assert prepared.loc[0, "normalized_salary"] == 90000
    assert "experience_level" not in prepared.columns
    assert "formatted_experience_level" not in prepared.columns


def test_prediction_limit_samples_deterministically():
    data = pd.DataFrame({"record_id": range(250)})

    first = _select_prediction_rows(data, limit=100, random_state=42)
    second = _select_prediction_rows(data, limit=100, random_state=42)

    assert len(first) == 100
    assert first["record_id"].tolist() == second["record_id"].tolist()
    assert first["record_id"].is_monotonic_increasing
    assert _select_prediction_rows(data, limit=None) is data


def test_class_weight_summary_does_not_resample_training_rows(capsys):
    labels = ["Entry"] * 4 + ["Senior"] * 2
    label_encoder = LabelEncoder().fit(labels)
    y_train = label_encoder.transform(labels)
    x_train = pd.DataFrame({"company_name": ["Example Co"] * len(labels)})

    print_class_weight_summary(x_train, y_train, label_encoder)

    output = capsys.readouterr().out
    assert re.search(r"Entry\s+4", output)
    assert re.search(r"Senior\s+2", output)
    assert "class_weight='balanced'" in output
    assert "No rows are duplicated" in output


if __name__ == "__main__":
    main()
