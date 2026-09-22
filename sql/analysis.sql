-- =========================================================
-- OLIST DELIVERY RISK — BUSINESS SQL ANALYSIS
-- =========================================================


-- 1. Total orders by status
SELECT
    order_status,
    COUNT(*) AS orders
FROM orders
GROUP BY order_status
ORDER BY orders DESC;


-- 2. Delivered orders and late-delivery rate
SELECT
    COUNT(*) AS delivered_orders,
    SUM(
        CASE
            WHEN order_delivered_customer_date::timestamp
                 > order_estimated_delivery_date::timestamp
            THEN 1
            ELSE 0
        END
    ) AS late_orders,
    ROUND(
        100.0 * SUM(
            CASE
                WHEN order_delivered_customer_date::timestamp
                     > order_estimated_delivery_date::timestamp
                THEN 1
                ELSE 0
            END
        ) / COUNT(*),
        2
    ) AS late_delivery_rate
FROM orders
WHERE order_status = 'delivered'
  AND order_delivered_customer_date IS NOT NULL
  AND order_estimated_delivery_date IS NOT NULL;


-- 3. Late-delivery rate by customer state
SELECT
    c.customer_state,
    COUNT(*) AS orders,
    ROUND(
        100.0 * AVG(
            CASE
                WHEN o.order_delivered_customer_date::timestamp
                     > o.order_estimated_delivery_date::timestamp
                THEN 1
                ELSE 0
            END
        ),
        2
    ) AS late_delivery_rate
FROM orders o
JOIN customers c
    ON o.customer_id = c.customer_id
WHERE o.order_status = 'delivered'
  AND o.order_delivered_customer_date IS NOT NULL
GROUP BY c.customer_state
HAVING COUNT(*) >= 100
ORDER BY late_delivery_rate DESC;


-- 4. Average item value and freight by delivery outcome
WITH order_values AS (
    SELECT
        order_id,
        SUM(price) AS total_item_value,
        SUM(freight_value) AS total_freight_value
    FROM order_items
    GROUP BY order_id
)

SELECT
    CASE
        WHEN o.order_delivered_customer_date::timestamp
             > o.order_estimated_delivery_date::timestamp
        THEN 'late'
        ELSE 'on_time'
    END AS delivery_status,

    ROUND(AVG(v.total_item_value)::numeric, 2)
        AS avg_item_value,

    ROUND(AVG(v.total_freight_value)::numeric, 2)
        AS avg_freight_value

FROM orders o

JOIN order_values v
    ON o.order_id = v.order_id

WHERE o.order_status = 'delivered'
  AND o.order_delivered_customer_date IS NOT NULL

GROUP BY delivery_status;


-- 5. Late-delivery rate by payment type
WITH primary_payment AS (
    SELECT DISTINCT ON (order_id)
        order_id,
        payment_type
    FROM order_payments
    ORDER BY
        order_id,
        payment_value DESC
)

SELECT
    p.payment_type,
    COUNT(*) AS orders,

    ROUND(
        100.0 * AVG(
            CASE
                WHEN o.order_delivered_customer_date::timestamp
                     > o.order_estimated_delivery_date::timestamp
                THEN 1
                ELSE 0
            END
        ),
        2
    ) AS late_delivery_rate

FROM orders o

JOIN primary_payment p
    ON o.order_id = p.order_id

WHERE o.order_status = 'delivered'

GROUP BY p.payment_type

ORDER BY late_delivery_rate DESC;


-- 6. Monthly late-delivery rate
SELECT
    EXTRACT(
        MONTH FROM order_purchase_timestamp::timestamp
    ) AS purchase_month,

    COUNT(*) AS orders,

    ROUND(
        100.0 * AVG(
            CASE
                WHEN order_delivered_customer_date::timestamp
                     > order_estimated_delivery_date::timestamp
                THEN 1
                ELSE 0
            END
        ),
        2
    ) AS late_delivery_rate

FROM orders

WHERE order_status = 'delivered'

GROUP BY purchase_month

ORDER BY purchase_month;


-- 7. Orders with multiple sellers
SELECT
    order_id,
    COUNT(DISTINCT seller_id) AS seller_count
FROM order_items
GROUP BY order_id
HAVING COUNT(DISTINCT seller_id) > 1
ORDER BY seller_count DESC;