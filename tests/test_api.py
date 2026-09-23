from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


VALID_PAYLOAD = {
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


def test_health():

    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    assert response.json() == {
        "status": "ok"
    }


def test_predict():

    response = client.post(
        "/predict",
        json=VALID_PAYLOAD,
    )

    assert response.status_code == 200

    body = response.json()

    assert (
        0
        <= body[
            "late_delivery_probability"
        ]
        <= 1
    )

    assert body["prediction"] in [0, 1]

    assert body["risk"] in [
        "low",
        "high",
    ]


def test_invalid_month():

    invalid_payload = (
        VALID_PAYLOAD.copy()
    )

    invalid_payload[
        "purchase_month"
    ] = 15

    response = client.post(
        "/predict",
        json=invalid_payload,
    )

    assert response.status_code == 422


def test_invalid_item_count():

    invalid_payload = (
        VALID_PAYLOAD.copy()
    )

    invalid_payload[
        "item_count"
    ] = 0

    response = client.post(
        "/predict",
        json=invalid_payload,
    )

    assert response.status_code == 422