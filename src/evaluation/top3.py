import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# STEP 7 - CLASS IMBALANCE + TOP 5 MODEL
# ============================================================

DATA_PATH = "data/processed/postings_clean.csv"

# Có thể thay đổi nếu file LazyPredict của bạn có tên khác
LAZY_RESULTS_PATH = "data/processed/lazypredict_results.csv"

OUTPUT_DIR = "data/processed"


print("=" * 70)
print("STEP 7 - CLASS IMBALANCE + TOP 3 MODEL")
print("=" * 70)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n[1] Loading data...")

df = pd.read_csv(DATA_PATH)

print(f"Original shape: {df.shape}")


# ============================================================
# 2. CHECK TARGET
# ============================================================

TARGET = "formatted_experience_level"

if TARGET not in df.columns:
    raise ValueError(
        f"Target '{TARGET}' does not exist in dataset."
    )

print(f"\nTarget: {TARGET}")


# ============================================================
# 3. REMOVE MISSING TARGET
# ============================================================

df_target = df.dropna(subset=[TARGET]).copy()

print(
    f"Shape after removing NaN target: "
    f"{df_target.shape}"
)


# ============================================================
# 4. CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("CLASS IMBALANCE ANALYSIS")
print("=" * 70)

class_counts = (
    df_target[TARGET]
    .value_counts()
    .sort_values(ascending=False)
)

total_samples = class_counts.sum()

class_percent = (
    class_counts / total_samples * 100
)

imbalance_table = pd.DataFrame({
    "Count": class_counts,
    "Percentage": class_percent
})

print("\nClass distribution:")
print(imbalance_table)


# ============================================================
# 5. IMBALANCE RATIO
# ============================================================

largest_class = class_counts.max()
smallest_class = class_counts.min()

imbalance_ratio = largest_class / smallest_class

print("\nLargest class:")
print(class_counts.idxmax())

print("\nSmallest class:")
print(class_counts.idxmin())

print(
    f"\nImbalance Ratio: "
    f"{imbalance_ratio:.2f}"
)


# ============================================================
# 6. SAVE CLASS IMBALANCE RESULTS
# ============================================================

imbalance_output = (
    f"{OUTPUT_DIR}/class_imbalance_results.csv"
)

imbalance_table.to_csv(
    imbalance_output,
    encoding="utf-8-sig"
)

print(
    f"\nClass imbalance results saved to:\n"
    f"{imbalance_output}"
)


# ============================================================
# 7. VISUALIZE CLASS DISTRIBUTION
# ============================================================

print("\n[7] Creating class distribution chart...")

plt.figure(figsize=(10, 6))

class_counts.plot(
    kind="bar"
)

plt.title(
    "Class Distribution - Job Experience Level"
)

plt.xlabel(
    "Experience Level"
)

plt.ylabel(
    "Number of Samples"
)

plt.xticks(
    rotation=30,
    ha="right"
)

plt.tight_layout()

chart_path = (
    f"{OUTPUT_DIR}/class_distribution.png"
)

plt.savefig(chart_path)

plt.close()

print(
    f"Chart saved to:\n{chart_path}"
)


# ============================================================
# 8. CHECK LAZYPREDICT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("TOP 5 MODEL FROM LAZYPREDICT")
print("=" * 70)

if not os.path.exists(LAZY_RESULTS_PATH):

    print(
        "\nWARNING:"
        "\nLazyPredict result file was not found:"
        f"\n{LAZY_RESULTS_PATH}"
    )

    print(
        "\nClass imbalance analysis is completed."
    )

    print(
        "\nPlease check the actual filename of your "
        "LazyPredict result CSV before running TOP 5."
    )

else:

    lazy_df = pd.read_csv(
        LAZY_RESULTS_PATH
    )

    print("\nLazyPredict results:")
    print(lazy_df)


    # ========================================================
    # 9. FIND F1 COLUMN
    # ========================================================

    possible_f1_columns = [
        "F1 Score",
        "F1",
        "F1 Macro",
        "F1_Macro",
        "f1_macro"
    ]

    f1_column = None

    for column in possible_f1_columns:

        if column in lazy_df.columns:
            f1_column = column
            break


    # ========================================================
    # 10. TOP 5
    # ========================================================

    if f1_column is not None:

        top5 = (
            lazy_df
            .sort_values(
                by=f1_column,
                ascending=False
            )
            .head(5)
            .copy()
        )

        print(
            f"\nRanking by: {f1_column}"
        )

        print("\nTOP 5 MODELS:")

        print(
            top5.to_string(
                index=False
            )
        )


        # ====================================================
        # 11. SAVE TOP 3
        # ====================================================

        top5_path = (
            f"{OUTPUT_DIR}/top5_models.csv"
        )

        top5.to_csv(
            top5_path,
            index=False,
            encoding="utf-8-sig"
        )

        print(
            f"\nTOP 5 saved to:\n{top5_path}"
        )

    else:

        print(
            "\nWARNING:"
            "\nCould not find an F1 column."
        )

        print(
            "\nAvailable columns:"
        )

        print(
            list(lazy_df.columns)
        )


# ============================================================
# 12. CONCLUSION
# ============================================================

print("\n" + "=" * 70)
print("STEP 7 SUMMARY")
print("=" * 70)

print(
    "\n[1] Class imbalance analysis: COMPLETED"
)

print(
    "[2] Class distribution saved: COMPLETED"
)

print(
    "[3] Class distribution chart: COMPLETED"
)

print(
    "[4] TOP 3 model selection:"
)

if os.path.exists(LAZY_RESULTS_PATH):
    print(
        "    Check the TOP 3 result above."
    )
else:
    print(
        "    Waiting for LazyPredict result file."
    )

print("\nSTEP 7 FINISHED.")