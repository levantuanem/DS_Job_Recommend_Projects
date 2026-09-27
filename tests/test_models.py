import argparse
import sys

import joblib
import pandas as pd

from src.models.utils import (
    ROOT_DIR,
    REPORTS_DIR,
    PREPROCESSOR_PATH,
    FEATURE_SELECTOR_PATH,
    LABEL_ENCODER_PATH,
    MODEL_COMPARISON_PATH,
    BEST_MODEL_INFO_PATH,
    BEST_MODEL_PATH,
    TUNING_RESULTS_PATH,
)
from src.models.predict import prepare_new_data  # dùng lại đúng pipeline của Member 3

DATA_PATH = ROOT_DIR / "data" / "processed" / "postings_clean.csv"
TARGET_COLUMN = "formatted_experience_level"

REQUIRED_ARTIFACTS = {
    "Preprocessor (Member 3, fit on train)": PREPROCESSOR_PATH,
    "Label encoder": LABEL_ENCODER_PATH,
    "Model comparison table (train.py)": MODEL_COMPARISON_PATH,
    "Best model info (train.py)": BEST_MODEL_INFO_PATH,
    "Tuned best model (tune.py)": BEST_MODEL_PATH,
    "Tuning results (tune.py)": TUNING_RESULTS_PATH,
}
OPTIONAL_ARTIFACTS = {
    "Feature selector (chỉ có nếu apply_feature_selection=True)": FEATURE_SELECTOR_PATH,
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

    for label, path in OPTIONAL_ARTIFACTS.items():
        status = "OK" if path.exists() else "không dùng"
        print(f"[{status:10}] {label}\n         -> {path}")

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
    preprocessor = joblib.load(PREPROCESSOR_PATH)          
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    selector = joblib.load(FEATURE_SELECTOR_PATH) if FEATURE_SELECTOR_PATH.exists() else None

    x_new = prepare_new_data(sample)                        # gọi hàm 
    x_new_transformed = preprocessor.transform(x_new)
    if selector is not None:
        x_new_transformed = selector.transform(x_new_transformed)

    pred_encoded = model.predict(x_new_transformed)
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


if __name__ == "__main__":
    main()
