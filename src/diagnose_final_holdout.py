import pandas as pd
import joblib

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)


# --------------------------------------------------
# Load data
# --------------------------------------------------

features = pd.read_csv(
    "data/processed/model_features_geo.csv"
)

targets = pd.read_csv(
    "data/processed/order_targets.csv"
)

targets["order_purchase_timestamp"] = pd.to_datetime(
    targets["order_purchase_timestamp"]
)


# --------------------------------------------------
# Add timestamp used only for chronological split
# --------------------------------------------------

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

df = df.sort_values(
    "order_purchase_timestamp"
).reset_index(drop=True)


# --------------------------------------------------
# Same chronological split:
# oldest 80% = development
# newest 20% = final holdout
# --------------------------------------------------

split_idx = int(
    len(df) * 0.80
)

test = (
    df.iloc[split_idx:]
    .copy()
)


# --------------------------------------------------
# Prepare test features
# --------------------------------------------------

y_test = (
    test["late_delivery"]
    .astype(int)
)

DROP_COLUMNS = [
    "late_delivery",
    "order_id",
    "order_purchase_timestamp",
    "customer_zip_code_prefix",
    "primary_seller_zip_code_prefix",
]

X_test = test.drop(
    columns=[
        col
        for col in DROP_COLUMNS
        if col in test.columns
    ]
)


# --------------------------------------------------
# Load frozen model
# --------------------------------------------------

model = joblib.load(
    "models/final_random_forest.joblib"
)


# Make sure columns match model
expected_features = (
    model
    .named_steps["preprocessor"]
    .feature_names_in_
    .tolist()
)

X_test = X_test[
    expected_features
]


# --------------------------------------------------
# Predict
# --------------------------------------------------

p = model.predict_proba(
    X_test
)[:, 1]


# --------------------------------------------------
# Metrics
# --------------------------------------------------

print(
    "Rows:",
    len(y_test),
)

print(
    "Positive rate:",
    y_test.mean(),
)

print(
    "Mean prediction:",
    p.mean(),
)

print(
    "Average precision:",
    average_precision_score(
        y_test,
        p,
    ),
)

print(
    "ROC-AUC:",
    roc_auc_score(
        y_test,
        p,
    ),
)

print(
    "ROC-AUC if score reversed:",
    roc_auc_score(
        y_test,
        1 - p,
    ),
)


# --------------------------------------------------
# Probability distribution by actual class
# --------------------------------------------------

result = pd.DataFrame(
    {
        "y_true":
            y_test.to_numpy(),

        "probability":
            p,
    }
)

print(
    "\nPrediction probability by class:"
)

print(
    result
    .groupby("y_true")
    ["probability"]
    .describe()
)