from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_features_geo.csv"
)


def main():

    df = pd.read_csv(DATA_PATH)

    print("=== FEATURE TABLE QUALITY ===")

    print(f"\nRows: {len(df)}")
    print(f"Columns: {df.shape[1]}")

    print(
        "\nDuplicate order IDs:",
        df["order_id"].duplicated().sum(),
    )

    print("\n=== TARGET DISTRIBUTION ===")
    print(df["late_delivery"].value_counts())

    print("\nTarget proportion:")
    print(
        df["late_delivery"]
        .value_counts(normalize=True)
    )

    print("\n=== MISSING VALUES ===")

    missing = (
        df.isna()
        .sum()
        .sort_values(ascending=False)
    )

    print(
        missing[
            missing > 0
        ]
    )

    print("\n=== NUMERICAL SUMMARY ===")

    numeric_cols = (
        df.select_dtypes(
            include=["int64", "float64"]
        )
        .columns
    )

    print(
        df[numeric_cols]
        .describe()
        .T
    )


if __name__ == "__main__":
    main()