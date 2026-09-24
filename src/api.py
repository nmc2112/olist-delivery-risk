import time
from typing import Optional

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from prometheus_client import (
    Counter,
    Histogram,
    make_asgi_app,
)

from src.predict import predict_delivery_risk


# --------------------------------------------------
# Prometheus metrics
# --------------------------------------------------

REQUEST_COUNT = Counter(
    "olist_api_requests_total",
    "Total FastAPI requests",
    [
        "method",
        "path",
        "status",
    ],
)

REQUEST_LATENCY = Histogram(
    "olist_api_request_duration_seconds",
    "FastAPI request latency",
    [
        "method",
        "path",
    ],
)

PREDICTION_COUNT = Counter(
    "olist_predictions_total",
    "Number of delivery-risk predictions",
    ["risk"],
)

PREDICTION_PROBABILITY = Histogram(
    "olist_prediction_probability",
    "Distribution of predicted late-delivery probabilities",
    buckets=[
        0.05,
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        1.00,
    ],
)


# --------------------------------------------------
# API
# --------------------------------------------------

app = FastAPI(
    title="Olist Delivery Risk API",
    description=(
        "Predict the probability that an "
        "e-commerce order will be delivered "
        "after its estimated delivery date."
    ),
    version="1.0.0",
)


@app.middleware("http")
async def monitor_requests(
    request: Request,
    call_next,
):

    path = request.url.path

    # Do not count Prometheus itself
    if path == "/metrics":
        return await call_next(request)

    start = time.perf_counter()

    response = await call_next(
        request
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    REQUEST_COUNT.labels(
        method=request.method,
        path=path,
        status=str(
            response.status_code
        ),
    ).inc()

    REQUEST_LATENCY.labels(
        method=request.method,
        path=path,
    ).observe(elapsed)

    return response


# Prometheus endpoint
metrics_app = make_asgi_app()

app.mount(
    "/metrics",
    metrics_app,
)


class OrderFeatures(BaseModel):

    estimated_delivery_days: float = Field(
        gt=0
    )

    purchase_month: int = Field(
        ge=1,
        le=12,
    )

    purchase_day_of_week: int = Field(
        ge=0,
        le=6,
    )

    purchase_hour: int = Field(
        ge=0,
        le=23,
    )

    approval_delay_hours: Optional[
        float
    ] = None

    customer_city: str

    customer_state: str

    item_count: int = Field(
        ge=1
    )

    unique_products: int = Field(
        ge=1
    )

    unique_sellers: int = Field(
        ge=1
    )

    total_item_value: float = Field(
        ge=0
    )

    average_item_price: float = Field(
        ge=0
    )

    max_item_price: float = Field(
        ge=0
    )

    total_freight_value: float = Field(
        ge=0
    )

    average_product_weight_g: Optional[
        float
    ] = Field(
        default=None,
        ge=0,
    )

    max_product_weight_g: Optional[
        float
    ] = Field(
        default=None,
        ge=0,
    )

    average_product_volume_cm3: Optional[
        float
    ] = Field(
        default=None,
        ge=0,
    )

    seller_state_count: int = Field(
        ge=1
    )

    primary_product_category: str

    primary_seller_state: str

    shipping_limit_days: float

    customer_seller_same_state: int = (
        Field(
            ge=0,
            le=1,
        )
    )

    payment_count: Optional[
        float
    ] = Field(
        default=None,
        ge=0,
    )

    total_payment_value: Optional[
        float
    ] = Field(
        default=None,
        ge=0,
    )

    max_payment_installments: Optional[
        float
    ] = Field(
        default=None,
        ge=0,
    )

    payment_type_count: Optional[
        float
    ] = Field(
        default=None,
        ge=0,
    )

    primary_payment_type: str = (
        "unknown"
    )

    customer_seller_distance_km: Optional[
        float
    ] = Field(
        default=None,
        ge=0,
    )


class PredictionResponse(BaseModel):

    late_delivery_probability: float

    prediction: int

    risk: str

    threshold: float


@app.get("/")
def root():

    return {
        "service":
            "Olist Delivery Risk API",

        "docs":
            "/docs",
    }


@app.get("/health")
def health():

    return {
        "status": "ok"
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    order: OrderFeatures,
):

    result = predict_delivery_risk(
        order.model_dump()
    )

    PREDICTION_COUNT.labels(
        risk=result["risk"]
    ).inc()

    PREDICTION_PROBABILITY.observe(
        result[
            "late_delivery_probability"
        ]
    )

    return result