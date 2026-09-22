DROP VIEW IF EXISTS delivery_order_features;

CREATE VIEW delivery_order_features AS

WITH item_summary AS (

    SELECT
        order_id,

        COUNT(*) AS item_count,

        COUNT(DISTINCT product_id)
            AS unique_products,

        COUNT(DISTINCT seller_id)
            AS unique_sellers,

        SUM(price)
            AS total_item_value,

        AVG(price)
            AS average_item_price,

        SUM(freight_value)
            AS total_freight_value,

        MAX(shipping_limit_date::timestamp)
            AS latest_shipping_limit_date

    FROM order_items

    GROUP BY order_id
),

payment_summary AS (

    SELECT
        order_id,

        COUNT(*) AS payment_count,

        SUM(payment_value)
            AS total_payment_value,

        MAX(payment_installments)
            AS max_payment_installments

    FROM order_payments

    GROUP BY order_id
)

SELECT
    o.order_id,

    c.customer_state,

    EXTRACT(
        MONTH FROM o.order_purchase_timestamp::timestamp
    ) AS purchase_month,

    EXTRACT(
        DOW FROM o.order_purchase_timestamp::timestamp
    ) AS purchase_day_of_week,

    EXTRACT(
        HOUR FROM o.order_purchase_timestamp::timestamp
    ) AS purchase_hour,

    EXTRACT(
        EPOCH FROM (
            o.order_approved_at::timestamp
            - o.order_purchase_timestamp::timestamp
        )
    ) / 3600.0 AS approval_delay_hours,

    EXTRACT(
        EPOCH FROM (
            o.order_estimated_delivery_date::timestamp
            - o.order_purchase_timestamp::timestamp
        )
    ) / 86400.0 AS estimated_delivery_days,

    i.item_count,
    i.unique_products,
    i.unique_sellers,
    i.total_item_value,
    i.average_item_price,
    i.total_freight_value,

    p.payment_count,
    p.total_payment_value,
    p.max_payment_installments,

    CASE
        WHEN o.order_delivered_customer_date::timestamp
             > o.order_estimated_delivery_date::timestamp
        THEN 1
        ELSE 0
    END AS late_delivery

FROM orders o

JOIN customers c
    ON o.customer_id = c.customer_id

LEFT JOIN item_summary i
    ON o.order_id = i.order_id

LEFT JOIN payment_summary p
    ON o.order_id = p.order_id

WHERE o.order_status = 'delivered'
  AND o.order_delivered_customer_date IS NOT NULL
  AND o.order_estimated_delivery_date IS NOT NULL;