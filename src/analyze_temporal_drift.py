import pandas as pd

features = pd.read_csv(
    "data/processed/model_features_geo.csv"
)

targets = pd.read_csv(
    "data/processed/order_targets.csv"
)

targets["order_purchase_timestamp"] = pd.to_datetime(
    targets["order_purchase_timestamp"]
)

df = features.merge(
    targets[
        [
            "order_id",
            "order_purchase_timestamp",
        ]
    ],
    on="order_id",
    how="left",
)

df["year_month"] = (
    df["order_purchase_timestamp"]
    .dt.to_period("M")
    .astype(str)
)

summary = (
    df.groupby("year_month")
    .agg(
        orders=("late_delivery", "size"),
        late_orders=("late_delivery", "sum"),
        late_rate=("late_delivery", "mean"),
        avg_estimated_days=(
            "estimated_delivery_days",
            "mean",
        ),
        avg_distance_km=(
            "customer_seller_distance_km",
            "mean",
        ),
    )
    .reset_index()
)

summary["late_rate"] *= 100

print(summary.to_string(index=False))

summary.to_csv(
    "reports/temporal_drift.csv",
    index=False,
)
