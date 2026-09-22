from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_features_geo.csv"
)


def late_rate_by_group(df, column):

    result = (
        df.groupby(column)["late_delivery"]
        .agg(["count", "mean"])
        .rename(
            columns={
                "count": "orders",
                "mean": "late_delivery_rate",
            }
        )
        .sort_values(
            "late_delivery_rate",
            ascending=False,
        )
    )

    result["late_delivery_rate"] *= 100

    return result


def main():

    df = pd.read_csv(DATA_PATH)

    print("=== OVERALL LATE DELIVERY RATE ===")

    print(
        f"{df['late_delivery'].mean() * 100:.2f}%"
    )

    categorical_features = [
        "customer_state",
        "primary_seller_state",
        "primary_product_category",
        "primary_payment_type",
        "purchase_month",
        "purchase_day_of_week",
        "customer_seller_same_state",
    ]

    for col in categorical_features:

        print(
            f"\n=== LATE DELIVERY BY "
            f"{col.upper()} ==="
        )

        print(
            late_rate_by_group(
                df,
                col,
            ).head(20)
        )

    print(
        "\n=== NUMERICAL FEATURES "
        "BY TARGET ==="
    )

    numerical_features = [
        "estimated_delivery_days",
        "approval_delay_hours",
        "item_count",
        "unique_sellers",
        "total_item_value",
        "total_freight_value",
        "average_product_weight_g",
        "average_product_volume_cm3",
        "shipping_limit_days",
        "customer_seller_distance_km",
        "max_payment_installments",
    ]

    print(
        df.groupby(
            "late_delivery"
        )[numerical_features]
        .mean()
        .T
        .rename(
            columns={
                0: "on_time_mean",
                1: "late_mean",
            }
        )
    )


if __name__ == "__main__":
    main()