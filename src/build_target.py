from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ORDERS_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "olist_orders_dataset.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "order_targets.csv"
)


DATE_COLUMNS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]


def build_target():

    orders = pd.read_csv(
        ORDERS_PATH,
        parse_dates=DATE_COLUMNS,
    )

    print("=== ORIGINAL ORDERS ===")
    print(f"Rows: {len(orders)}")

    print("\nOrder status:")
    print(orders["order_status"].value_counts())

    print("\nMissing delivery timestamps:")
    print(
        orders[
            [
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
            ]
        ]
        .isna()
        .sum()
    )

    # Target can only be determined for delivered orders
    target_df = orders[
        orders["order_status"].eq("delivered")
        & orders["order_delivered_customer_date"].notna()
        & orders["order_estimated_delivery_date"].notna()
    ].copy()

    target_df["late_delivery"] = (
        target_df["order_delivered_customer_date"]
        > target_df["order_estimated_delivery_date"]
    ).astype(int)

    print("\n=== MODELING TARGET DATA ===")
    print(f"Rows: {len(target_df)}")

    print("\nTarget counts:")
    print(
        target_df["late_delivery"].value_counts()
    )

    print("\nTarget proportions:")
    print(
        target_df["late_delivery"].value_counts(
            normalize=True
        )
    )

    # Useful timing feature known before delivery
    target_df["estimated_delivery_days"] = (
        (
            target_df["order_estimated_delivery_date"]
            - target_df["order_purchase_timestamp"]
        )
        .dt.total_seconds()
        / 86400
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    target_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nTarget dataset saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    build_target()