from pathlib import Path

import pandas as pd

from database import engine


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


TABLES = {
    "customers": "olist_customers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "order_payments": "olist_order_payments_dataset.csv",
    "order_reviews": "olist_order_reviews_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "product_category_translation":
        "product_category_name_translation.csv",
}


def load_table(table_name, filename):
    path = RAW_DIR / filename

    print(f"\nLoading {table_name}...")

    df = pd.read_csv(path)

    df.to_sql(
        table_name,
        engine,
        if_exists="replace",
        index=False,
        chunksize=5000,
        method="multi",
    )

    print(
        f"{table_name}: {len(df):,} rows loaded"
    )


def main():
    print(
        "=== LOADING OLIST DATA TO POSTGRESQL ==="
    )

    for table_name, filename in TABLES.items():
        load_table(
            table_name,
            filename,
        )

    print(
        "\nAll tables loaded successfully."
    )


if __name__ == "__main__":
    main()