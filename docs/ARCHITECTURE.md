Goal: the real-world problem

Imagine you work for Olist, a Brazilian e-commerce marketplace. Every order is given an estimated delivery date. About 8% of orders arrive late, and late deliveries lead to 1-star reviews, refund requests and customers who never come back.

The question this project answers: at the moment an order is approved, can we predict which orders are at risk of arriving late?

If we can, the business can act early: prioritise packing, switch carriers, or warn the customer in advance. The model outputs a probability, for example “this order has a 32% chance of being late”.

The most important constraint: the model may only use information available at order time, such as price, weight, distance and the estimated delivery window. It must never use information that is only known after delivery, like the actual delivery date or the customer review. Using those would be data leakage: the model would look great in testing and be useless in production.

Flow: 4 layers, from raw data to the user

Layer 1, data preparation (Data Engineering): “Turn 9 separate tables into 1 clean table”

Input: 9 raw Olist CSV files (orders, customers, sellers, products, payments, geolocation...).
Label each order as late or on time, then aggregate everything into one row per order, 32 columns, including customer–seller distance.
Great Expectations validates the data: no duplicate orders, no negative prices, valid distances...
Airflow runs these steps automatically, in the right order.

Layer 2, modelling (Machine Learning): “Learn from the past to predict the future, and evaluate honestly”

Chronological split: the oldest 80% of orders for development, the newest 20% as a “future” holdout, because in production the model always predicts the future.
V1, Random Forest: looked good during development but collapsed on the future period, doing worse than random. The cause: the late-delivery rate changes a lot over time, from 1% to 21% depending on the month. This is temporal drift.
V2, Logistic Regression trained only on the 20,000 most recent orders: recent data reflects current conditions better. It ranks risk much better and its probabilities are better calibrated.
The final model is saved as a single .joblib file.

Layer 3, serving: “Turn the model into a service anyone can call”

FastAPI: other systems send order details to /predict and get back a late-delivery probability and a risk level.
Streamlit: a dashboard for non-technical users to explore statistics and try predictions.

Layer 4, monitoring: “Know when something goes wrong”

Prometheus counts requests, measures response latency and tracks the distribution of predicted probabilities.
Grafana turns those metrics into charts. For example, a sudden rise in “high-risk” predictions is a sign of drift and a signal to retrain.