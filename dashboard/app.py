from pathlib import Path

import pandas as pd
import streamlit as st

from src.predict import predict_delivery_risk


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_features_geo.csv"
)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


df = load_data()


st.set_page_config(
    page_title="Olist Delivery Risk Dashboard",
    layout="wide",
)


st.title("Olist Delivery Risk Dashboard")

st.caption(
    "Operational analysis and late-delivery risk prediction"
)


# --------------------------------------------------
# KPI row
# --------------------------------------------------

total_orders = len(df)

late_orders = int(
    df["late_delivery"].sum()
)

late_rate = (
    df["late_delivery"].mean() * 100
)


col1, col2, col3 = st.columns(3)

col1.metric(
    "Orders",
    f"{total_orders:,}",
)

col2.metric(
    "Late deliveries",
    f"{late_orders:,}",
)

col3.metric(
    "Late-delivery rate",
    f"{late_rate:.2f}%",
)


# --------------------------------------------------
# Late delivery by customer state
# --------------------------------------------------

st.subheader(
    "Late delivery by customer state"
)

state_summary = (
    df.groupby("customer_state")
    ["late_delivery"]
    .agg(
        orders="count",
        late_delivery_rate="mean",
    )
    .reset_index()
)

state_summary[
    "late_delivery_rate"
] *= 100

state_summary = (
    state_summary
    .sort_values(
        "late_delivery_rate",
        ascending=False,
    )
)

st.dataframe(
    state_summary,
    use_container_width=True,
)


# --------------------------------------------------
# Monthly trend
# --------------------------------------------------

st.subheader(
    "Late delivery by purchase month"
)

monthly = (
    df.groupby("purchase_month")
    ["late_delivery"]
    .mean()
    .mul(100)
)

st.line_chart(monthly)


# --------------------------------------------------
# Same-state vs cross-state
# --------------------------------------------------

st.subheader(
    "Same-state vs cross-state deliveries"
)

same_state = (
    df.groupby(
        "customer_seller_same_state"
    )["late_delivery"]
    .mean()
    .mul(100)
    .rename(
        index={
            0: "Cross-state",
            1: "Same-state",
        }
    )
)

st.bar_chart(same_state)


# --------------------------------------------------
# Numeric profile
# --------------------------------------------------

st.subheader(
    "Average characteristics by delivery outcome"
)

numeric_features = [
    "estimated_delivery_days",
    "approval_delay_hours",
    "total_item_value",
    "total_freight_value",
    "average_product_weight_g",
    "average_product_volume_cm3",
    "customer_seller_distance_km",
]

profile = (
    df.groupby("late_delivery")
    [numeric_features]
    .mean()
    .T
    .rename(
        columns={
            0: "On time",
            1: "Late",
        }
    )
)

st.dataframe(
    profile,
    use_container_width=True,
)


# --------------------------------------------------
# Prediction form
# --------------------------------------------------

st.subheader(
    "Predict late-delivery risk"
)


with st.form(
    "prediction_form"
):

    col1, col2 = st.columns(2)

    with col1:

        estimated_delivery_days = (
            st.number_input(
                "Estimated delivery days",
                min_value=1.0,
                value=20.0,
            )
        )

        purchase_month = (
            st.number_input(
                "Purchase month",
                min_value=1,
                max_value=12,
                value=3,
            )
        )

        purchase_day_of_week = (
            st.number_input(
                "Purchase day of week",
                min_value=0,
                max_value=6,
                value=1,
            )
        )

        purchase_hour = (
            st.number_input(
                "Purchase hour",
                min_value=0,
                max_value=23,
                value=14,
            )
        )

        approval_delay_hours = (
            st.number_input(
                "Approval delay (hours)",
                min_value=0.0,
                value=8.0,
            )
        )

        customer_city = (
            st.text_input(
                "Customer city",
                value="rio de janeiro",
            )
        )

        customer_state = (
            st.text_input(
                "Customer state",
                value="RJ",
            )
        )

        item_count = (
            st.number_input(
                "Item count",
                min_value=1,
                value=1,
            )
        )

        unique_products = (
            st.number_input(
                "Unique products",
                min_value=1,
                value=1,
            )
        )

        unique_sellers = (
            st.number_input(
                "Unique sellers",
                min_value=1,
                value=1,
            )
        )

        total_item_value = (
            st.number_input(
                "Total item value",
                min_value=0.0,
                value=150.0,
            )
        )

        average_item_price = (
            st.number_input(
                "Average item price",
                min_value=0.0,
                value=150.0,
            )
        )

        max_item_price = (
            st.number_input(
                "Max item price",
                min_value=0.0,
                value=150.0,
            )
        )


    with col2:

        total_freight_value = (
            st.number_input(
                "Total freight value",
                min_value=0.0,
                value=25.0,
            )
        )

        average_product_weight_g = (
            st.number_input(
                "Average product weight (g)",
                min_value=0.0,
                value=2000.0,
            )
        )

        max_product_weight_g = (
            st.number_input(
                "Max product weight (g)",
                min_value=0.0,
                value=2000.0,
            )
        )

        average_product_volume_cm3 = (
            st.number_input(
                "Average product volume (cm³)",
                min_value=0.0,
                value=15000.0,
            )
        )

        seller_state_count = (
            st.number_input(
                "Seller state count",
                min_value=1,
                value=1,
            )
        )

        primary_product_category = (
            st.text_input(
                "Primary product category",
                value="health_beauty",
            )
        )

        primary_seller_state = (
            st.text_input(
                "Primary seller state",
                value="SP",
            )
        )

        shipping_limit_days = (
            st.number_input(
                "Shipping limit days",
                value=6.0,
            )
        )

        customer_seller_same_state = (
            st.selectbox(
                "Customer and seller same state",
                options=[0, 1],
            )
        )

        payment_count = (
            st.number_input(
                "Payment count",
                min_value=0,
                value=1,
            )
        )

        total_payment_value = (
            st.number_input(
                "Total payment value",
                min_value=0.0,
                value=175.0,
            )
        )

        max_payment_installments = (
            st.number_input(
                "Max payment installments",
                min_value=0,
                value=3,
            )
        )

        payment_type_count = (
            st.number_input(
                "Payment type count",
                min_value=0,
                value=1,
            )
        )

        primary_payment_type = (
            st.text_input(
                "Primary payment type",
                value="credit_card",
            )
        )

        customer_seller_distance_km = (
            st.number_input(
                "Customer-seller distance (km)",
                min_value=0.0,
                value=450.0,
            )
        )

    submitted = st.form_submit_button(
        "Predict risk"
    )


if submitted:

    payload = {
        "estimated_delivery_days":
            estimated_delivery_days,

        "purchase_month":
            purchase_month,

        "purchase_day_of_week":
            purchase_day_of_week,

        "purchase_hour":
            purchase_hour,

        "approval_delay_hours":
            approval_delay_hours,

        "customer_city":
            customer_city,

        "customer_state":
            customer_state,

        "item_count":
            item_count,

        "unique_products":
            unique_products,

        "unique_sellers":
            unique_sellers,

        "total_item_value":
            total_item_value,

        "average_item_price":
            average_item_price,

        "max_item_price":
            max_item_price,

        "total_freight_value":
            total_freight_value,

        "average_product_weight_g":
            average_product_weight_g,

        "max_product_weight_g":
            max_product_weight_g,

        "average_product_volume_cm3":
            average_product_volume_cm3,

        "seller_state_count":
            seller_state_count,

        "primary_product_category":
            primary_product_category,

        "primary_seller_state":
            primary_seller_state,

        "shipping_limit_days":
            shipping_limit_days,

        "customer_seller_same_state":
            customer_seller_same_state,

        "payment_count":
            payment_count,

        "total_payment_value":
            total_payment_value,

        "max_payment_installments":
            max_payment_installments,

        "payment_type_count":
            payment_type_count,

        "primary_payment_type":
            primary_payment_type,

        "customer_seller_distance_km":
            customer_seller_distance_km,
    }

    result = predict_delivery_risk(
        payload
    )

    st.write(
        "### Prediction result"
    )

    st.metric(
        "Late-delivery probability",
        f"{result['late_delivery_probability']:.2%}",
    )

    st.metric(
        "Risk",
        result["risk"].upper(),
    )

    st.caption(
        f"Decision threshold: "
        f"{result['threshold']:.3f}"
    )