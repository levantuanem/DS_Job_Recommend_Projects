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
from src.models.predict import prepare_new_data  # dùng lại đúng pipeline của Member 3
from src.models.train import print_oversampling_summary

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


def test_oversampling_summary_reports_balanced_training_counts(capsys):
    labels = ["Entry"] * 4 + ["Senior"] * 2
    label_encoder = LabelEncoder().fit(labels)
    y_train = label_encoder.transform(labels)
    x_train = pd.DataFrame({"company_name": ["Example Co"] * len(labels)})

    print_oversampling_summary(x_train, y_train, label_encoder)

    output = capsys.readouterr().out
    after_counts = output.split("Training class distribution after oversampling:")[1]
    assert re.search(r"Entry\s+4", after_counts)
    assert re.search(r"Senior\s+4", after_counts)
    assert "Sample rows after oversampling" in output


if __name__ == "__main__":
    main()
