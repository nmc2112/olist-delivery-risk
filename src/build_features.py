from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TARGET_PATH = PROCESSED_DIR / "order_targets.csv"
OUTPUT_PATH = PROCESSED_DIR / "model_features.csv"


def load_data():

    targets = pd.read_csv(
        TARGET_PATH,
        parse_dates=[
            "order_purchase_timestamp",
            "order_approved_at",
            "order_estimated_delivery_date",
        ],
    )

    customers = pd.read_csv(
        RAW_DIR / "olist_customers_dataset.csv"
    )

    items = pd.read_csv(
        RAW_DIR / "olist_order_items_dataset.csv",
        parse_dates=["shipping_limit_date"],
    )

    products = pd.read_csv(
        RAW_DIR / "olist_products_dataset.csv"
    )

    sellers = pd.read_csv(
        RAW_DIR / "olist_sellers_dataset.csv"
    )

    payments = pd.read_csv(
        RAW_DIR / "olist_order_payments_dataset.csv"
    )

    translation = pd.read_csv(
        RAW_DIR / "product_category_name_translation.csv"
    )

    return (
        targets,
        customers,
        items,
        products,
        sellers,
        payments,
        translation,
    )


def build_order_item_features(
    items,
    products,
    sellers,
    translation,
):

    # Add product information
    items = items.merge(
        products,
        on="product_id",
        how="left",
        validate="many_to_one",
    )

    # Translate categories to English
    items = items.merge(
        translation,
        on="product_category_name",
        how="left",
        validate="many_to_one",
    )

    # Add seller information
    items = items.merge(
        sellers,
        on="seller_id",
        how="left",
        validate="many_to_one",
    )

    # Product volume
    items["product_volume_cm3"] = (
        items["product_length_cm"]
        * items["product_height_cm"]
        * items["product_width_cm"]
    )

    # Aggregate to ONE ROW PER ORDER
    order_items = (
        items.groupby("order_id")
        .agg(
            item_count=(
                "order_item_id",
                "count",
            ),
            unique_products=(
                "product_id",
                "nunique",
            ),
            unique_sellers=(
                "seller_id",
                "nunique",
            ),
            total_item_value=(
                "price",
                "sum",
            ),
            average_item_price=(
                "price",
                "mean",
            ),
            max_item_price=(
                "price",
                "max",
            ),
            total_freight_value=(
                "freight_value",
                "sum",
            ),
            average_product_weight_g=(
                "product_weight_g",
                "mean",
            ),
            max_product_weight_g=(
                "product_weight_g",
                "max",
            ),
            average_product_volume_cm3=(
                "product_volume_cm3",
                "mean",
            ),
            seller_state_count=(
                "seller_state",
                "nunique",
            ),
            latest_shipping_limit_date=(
                "shipping_limit_date",
                "max",
            ),
        )
        .reset_index()
    )

    # Use the most expensive item to represent
    # the dominant product category / seller
    primary_item = (
        items.sort_values(
            "price",
            ascending=False,
        )
        .drop_duplicates(
            subset="order_id"
        )
        [
            [
                "order_id",
                "product_category_name_english",
                "seller_state",
            ]
        ]
        .rename(
            columns={
                "product_category_name_english":
                    "primary_product_category",
                "seller_state":
                    "primary_seller_state",
            }
        )
    )

    order_items = order_items.merge(
        primary_item,
        on="order_id",
        how="left",
        validate="one_to_one",
    )

    return order_items


def build_payment_features(payments):

    payment_summary = (
        payments.groupby("order_id")
        .agg(
            payment_count=(
                "payment_sequential",
                "count",
            ),
            total_payment_value=(
                "payment_value",
                "sum",
            ),
            max_payment_installments=(
                "payment_installments",
                "max",
            ),
            payment_type_count=(
                "payment_type",
                "nunique",
            ),
        )
        .reset_index()
    )

    # Main payment = largest payment associated
    # with the order
    primary_payment = (
        payments.sort_values(
            "payment_value",
            ascending=False,
        )
        .drop_duplicates(
            subset="order_id"
        )
        [
            [
                "order_id",
                "payment_type",
            ]
        ]
        .rename(
            columns={
                "payment_type":
                    "primary_payment_type"
            }
        )
    )

    return payment_summary.merge(
        primary_payment,
        on="order_id",
        how="left",
        validate="one_to_one",
    )


def build_features():

    (
        targets,
        customers,
        items,
        products,
        sellers,
        payments,
        translation,
    ) = load_data()

    print("=== BUILDING MODEL FEATURES ===")
    print(f"Target orders: {len(targets)}")

    # -------------------------
    # Base order features
    # -------------------------

    features = targets[
        [
            "order_id",
            "customer_id",
            "order_purchase_timestamp",
            "order_approved_at",
            "order_estimated_delivery_date",
            "estimated_delivery_days",
            "late_delivery",
        ]
    ].copy()

    # Time features known at purchase time
    features["purchase_month"] = (
        features[
            "order_purchase_timestamp"
        ].dt.month
    )

    features["purchase_day_of_week"] = (
        features[
            "order_purchase_timestamp"
        ].dt.dayofweek
    )

    features["purchase_hour"] = (
        features[
            "order_purchase_timestamp"
        ].dt.hour
    )

    # Approval delay
    features["approval_delay_hours"] = (
        (
            features["order_approved_at"]
            - features["order_purchase_timestamp"]
        )
        .dt.total_seconds()
        / 3600
    )

    # -------------------------
    # Customer features
    # -------------------------

    features = features.merge(
        customers,
        on="customer_id",
        how="left",
        validate="one_to_one",
    )

    # -------------------------
    # Item/product/seller features
    # -------------------------

    item_features = build_order_item_features(
        items,
        products,
        sellers,
        translation,
    )

    features = features.merge(
        item_features,
        on="order_id",
        how="left",
        validate="one_to_one",
    )

    # Shipping window
    features["shipping_limit_days"] = (
        (
            features[
                "latest_shipping_limit_date"
            ]
            - features[
                "order_purchase_timestamp"
            ]
        )
        .dt.total_seconds()
        / 86400
    )

    # Customer vs primary seller geography
    features["customer_seller_same_state"] = (
        features["customer_state"]
        == features["primary_seller_state"]
    ).astype(int)

    # -------------------------
    # Payment features
    # -------------------------

    payment_features = build_payment_features(
        payments
    )

    features = features.merge(
        payment_features,
        on="order_id",
        how="left",
        validate="one_to_one",
    )

    # -------------------------
    # Drop IDs/timestamps that are not ML features
    # -------------------------

    features = features.drop(
        columns=[
            "customer_id",
            "customer_unique_id",
            "order_purchase_timestamp",
            "order_approved_at",
            "order_estimated_delivery_date",
            "latest_shipping_limit_date",
        ]
    )

    features[
        "primary_product_category"
    ] = features[
        "primary_product_category"
    ].fillna("unknown")

    features[
        "primary_seller_state"
    ] = features[
        "primary_seller_state"
    ].fillna("unknown")

    print("\n=== FINAL FEATURE TABLE ===")
    print(f"Rows: {len(features)}")
    print(f"Columns: {features.shape[1]}")

    print(
        f"Duplicate order IDs: "
        f"{features['order_id'].duplicated().sum()}"
    )

    print("\nMissing values:")
    missing = (
        features
        .isna()
        .sum()
        .sort_values(ascending=False)
    )

    print(
        missing[
            missing > 0
        ]
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    features.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nFeature table saved to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    build_features()