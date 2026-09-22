from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def inspect_csv(file_path):
    df = pd.read_csv(file_path)

    return {
        "file": file_path.name,
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": df.columns.tolist(),
        "missing_values": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
    }


def main():
    csv_files = sorted(RAW_DATA_DIR.glob("*.csv"))

    print("=== OLIST DATASET INVENTORY ===")

    for file_path in csv_files:
        info = inspect_csv(file_path)

        print(f"\n--- {info['file']} ---")
        print(f"Rows: {info['rows']}")
        print(f"Columns: {info['columns']}")
        print(f"Missing values: {info['missing_values']}")
        print(f"Duplicate rows: {info['duplicates']}")

        print("Columns:")
        for col in info["column_names"]:
            print(f"  - {col}")


if __name__ == "__main__":
    main()