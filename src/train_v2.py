from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


ROOT = Path(__file__).resolve().parents[1]

FEATURE_PATH = (
    ROOT
    / "data"
    / "processed"
    / "model_features_geo.csv"
)

TARGET_PATH = (
    ROOT
    / "data"
    / "processed"
    / "order_targets.csv"
)


# --------------------------------------------------
# Load + chronological ordering
# --------------------------------------------------

features = pd.read_csv(FEATURE_PATH)

targets = pd.read_csv(TARGET_PATH)

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

df = (
    df.sort_values("order_purchase_timestamp")
    .reset_index(drop=True)
)


# --------------------------------------------------
# Original chronological split
# --------------------------------------------------

split_idx = int(len(df) * 0.80)

dev = df.iloc[:split_idx].copy()

future_holdout = (
    df.iloc[split_idx:].copy()
)


DROP_COLUMNS = [
    "late_delivery",
    "order_id",
    "order_purchase_timestamp",
    "customer_zip_code_prefix",
    "primary_seller_zip_code_prefix",
]


def prepare(frame):

    y = (
        frame["late_delivery"]
        .astype(int)
    )

    X = frame.drop(
        columns=[
            c
            for c in DROP_COLUMNS
            if c in frame.columns
        ]
    )

    return X, y


def make_pipeline(X_train):

    categorical = (
        X_train
        .select_dtypes(include=["object"])
        .columns
        .tolist()
    )

    numeric = [
        c
        for c in X_train.columns
        if c not in categorical
    ]

    numeric_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                numeric_pipeline,
                numeric,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical,
            ),
        ]
    )

    model = LogisticRegression(
        max_iter=2000,
        random_state=2026,
    )

    return Pipeline(
        [
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )


# ==================================================
# 1. Threshold selection INSIDE development data
# ==================================================
#
# Train:
#   20k orders immediately before validation
#
# Validation:
#   last 10k orders of development period
# ==================================================

validation_size = 10000
training_window = 20000

threshold_validation = (
    dev.iloc[-validation_size:]
    .copy()
)

threshold_training = (
    dev.iloc[
        -(training_window + validation_size):
        -validation_size
    ]
    .copy()
)


X_train_thr, y_train_thr = prepare(
    threshold_training
)

X_val, y_val = prepare(
    threshold_validation
)


threshold_model = make_pipeline(
    X_train_thr
)

threshold_model.fit(
    X_train_thr,
    y_train_thr,
)

val_probability = (
    threshold_model
    .predict_proba(X_val)[:, 1]
)


# --------------------------------------------------
# Select threshold by maximum F1
# --------------------------------------------------

precision, recall, thresholds = (
    precision_recall_curve(
        y_val,
        val_probability,
    )
)

f1_values = (
    2
    * precision[:-1]
    * recall[:-1]
    / (
        precision[:-1]
        + recall[:-1]
        + 1e-12
    )
)

best_index = int(
    np.argmax(f1_values)
)

selected_threshold = float(
    thresholds[best_index]
)


print(
    "\n===== THRESHOLD VALIDATION ====="
)

print(
    "Training rows:",
    len(threshold_training),
)

print(
    "Validation rows:",
    len(threshold_validation),
)

print(
    "Validation actual rate:",
    y_val.mean(),
)

print(
    "Validation mean prediction:",
    val_probability.mean(),
)

print(
    "Validation AP:",
    average_precision_score(
        y_val,
        val_probability,
    ),
)

print(
    "Validation ROC-AUC:",
    roc_auc_score(
        y_val,
        val_probability,
    ),
)

print(
    "Selected threshold:",
    selected_threshold,
)

print(
    "Threshold precision:",
    precision[best_index],
)

print(
    "Threshold recall:",
    recall[best_index],
)

print(
    "Threshold F1:",
    f1_values[best_index],
)


# ==================================================
# 2. Train frozen V2 model on latest 20k DEV rows
# ==================================================

final_training = (
    dev.iloc[-training_window:]
    .copy()
)

X_final_train, y_final_train = prepare(
    final_training
)

final_model = make_pipeline(
    X_final_train
)

final_model.fit(
    X_final_train,
    y_final_train,
)


# --------------------------------------------------
# Save frozen model
# --------------------------------------------------

MODEL_PATH = (
    ROOT
    / "models"
    / "final_logistic_v2.joblib"
)

joblib.dump(
    final_model,
    MODEL_PATH,
)


# ==================================================
# 3. Post-hoc future-period diagnostic
# ==================================================

X_future, y_future = prepare(
    future_holdout
)

future_probability = (
    final_model
    .predict_proba(
        X_future
    )[:, 1]
)

future_prediction = (
    future_probability
    >= selected_threshold
).astype(int)


tn, fp, fn, tp = (
    confusion_matrix(
        y_future,
        future_prediction,
    )
    .ravel()
)


results = {
    "model":
        "logistic_regression",

    "training_strategy":
        "latest_20000_development_orders",

    "threshold_selection":
        "maximum_f1_on_last_10000_development_orders",

    "threshold":
        selected_threshold,

    "training_rows":
        len(final_training),

    "future_rows":
        len(future_holdout),

    "future_positive_rate":
        float(
            y_future.mean()
        ),

    "mean_prediction":
        float(
            future_probability.mean()
        ),

    "average_precision":
        float(
            average_precision_score(
                y_future,
                future_probability,
            )
        ),

    "roc_auc":
        float(
            roc_auc_score(
                y_future,
                future_probability,
            )
        ),

    "brier_score":
        float(
            brier_score_loss(
                y_future,
                future_probability,
            )
        ),

    "precision":
        float(
            precision_score(
                y_future,
                future_prediction,
                zero_division=0,
            )
        ),

    "recall":
        float(
            recall_score(
                y_future,
                future_prediction,
                zero_division=0,
            )
        ),

    "f1":
        float(
            f1_score(
                y_future,
                future_prediction,
                zero_division=0,
            )
        ),

    "confusion_matrix": {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    },
}


REPORT_PATH = (
    ROOT
    / "reports"
    / "model_v2_results.json"
)

with open(
    REPORT_PATH,
    "w",
) as f:

    json.dump(
        results,
        f,
        indent=2,
    )


THRESHOLD_PATH = (
    ROOT
    / "reports"
    / "model_v2_threshold.json"
)

with open(
    THRESHOLD_PATH,
    "w",
) as f:

    json.dump(
        {
            "threshold":
                selected_threshold,

            "selection_method":
                "maximum_f1_on_recent_development_validation",
        },
        f,
        indent=2,
    )


print(
    "\n===== MODEL V2 FUTURE DIAGNOSTIC ====="
)

print(
    json.dumps(
        results,
        indent=2,
    )
)

print(
    f"\nSaved model: {MODEL_PATH}"
)

print(
    f"Saved results: {REPORT_PATH}"
)