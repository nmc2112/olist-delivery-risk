from src.predict import (
    MODEL_FEATURES,
    THRESHOLD,
    predict_delivery_risk,
)


SAMPLE_ORDER = {
    "estimated_delivery_days": 20,
    "purchase_month": 3,
    "purchase_day_of_week": 1,
    "purchase_hour": 14,
    "approval_delay_hours": 8,

    "customer_city": "rio de janeiro",
    "customer_state": "RJ",

    "item_count": 1,
    "unique_products": 1,
    "unique_sellers": 1,

    "total_item_value": 150,
    "average_item_price": 150,
    "max_item_price": 150,
    "total_freight_value": 25,

    "average_product_weight_g": 2000,
    "max_product_weight_g": 2000,
    "average_product_volume_cm3": 15000,

    "seller_state_count": 1,

    "primary_product_category": "health_beauty",
    "primary_seller_state": "SP",

    "shipping_limit_days": 6,
    "customer_seller_same_state": 0,

    "payment_count": 1,
    "total_payment_value": 175,
    "max_payment_installments": 3,
    "payment_type_count": 1,

    "primary_payment_type": "credit_card",

    "customer_seller_distance_km": 450,
}


def test_model_has_expected_features():
    assert len(MODEL_FEATURES) == 28


def test_threshold_is_probability():
    assert 0 < THRESHOLD < 1


def test_prediction_probability_range():

    result = predict_delivery_risk(
        SAMPLE_ORDER
    )

    assert (
        0
        <= result["late_delivery_probability"]
        <= 1
    )


def test_prediction_is_binary():

    result = predict_delivery_risk(
        SAMPLE_ORDER
    )

    assert result["prediction"] in [0, 1]


def test_risk_label():

    result = predict_delivery_risk(
        SAMPLE_ORDER
    )

    assert result["risk"] in [
        "lower",
        "high",
    ]