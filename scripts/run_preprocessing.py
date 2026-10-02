from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.clean_data import run_pipeline


def main() -> None:
    run_pipeline(
        raw_dir=PROJECT_ROOT / "data" / "raw",
        processed_dir=PROJECT_ROOT / "data" / "processed",
        balance_target=False,
    )
    print("Preprocessing completed without resampling the full dataset.")


if __name__ == "__main__":
    main()