from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
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
# Preserve original final 20% holdout
# --------------------------------------------------

dev_end = int(len(df) * 0.80)

dev = df.iloc[:dev_end].copy()


DROP_COLUMNS = [
    "late_delivery",
    "order_id",
    "order_purchase_timestamp",
    "customer_zip_code_prefix",
    "primary_seller_zip_code_prefix",
]


def prepare(frame):

    y = frame[
        "late_delivery"
    ].astype(int)

    X = frame.drop(
        columns=[
            c
            for c in DROP_COLUMNS
            if c in frame.columns
        ]
    )

    return X, y


# --------------------------------------------------
# Build preprocessing separately for each backtest
# --------------------------------------------------

def make_pipeline(
    X_train,
    model_name,
):

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

    numeric_pipe = Pipeline(
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

    categorical_pipe = Pipeline(
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
                numeric_pipe,
                numeric,
            ),
            (
                "categorical",
                categorical_pipe,
                categorical,
            ),
        ]
    )

    if model_name == "logistic":

        model = LogisticRegression(
            max_iter=2000,
            random_state=2026,
        )

    elif model_name == "random_forest":

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=30,
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=-1,
            random_state=2026,
        )

    else:
        raise ValueError(model_name)

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


# --------------------------------------------------
# Backtest configuration
#
# Four later validation periods inside DEV.
# Each validation block = 10% of dev.
# --------------------------------------------------

n = len(dev)

block = int(n * 0.10)

validation_ends = [
    int(n * 0.70),
    int(n * 0.80),
    int(n * 0.90),
    n,
]

training_windows = {
    "expanding": None,
    "last_20000": 20000,
    "last_40000": 40000,
}

models = [
    "logistic",
    "random_forest",
]


results = []


for val_end in validation_ends:

    val_start = val_end - block

    validation = (
        dev.iloc[
            val_start:val_end
        ]
        .copy()
    )

    for window_name, window_size in training_windows.items():

        if window_size is None:

            train = (
                dev.iloc[
                    :val_start
                ]
                .copy()
            )

        else:

            train_start = max(
                0,
                val_start - window_size,
            )

            train = (
                dev.iloc[
                    train_start:val_start
                ]
                .copy()
            )

        X_train, y_train = prepare(
            train
        )

        X_val, y_val = prepare(
            validation
        )

        for model_name in models:

            pipeline = make_pipeline(
                X_train,
                model_name,
            )

            pipeline.fit(
                X_train,
                y_train,
            )

            p = pipeline.predict_proba(
                X_val
            )[:, 1]

            ap = (
                average_precision_score(
                    y_val,
                    p,
                )
            )

            auc = (
                roc_auc_score(
                    y_val,
                    p,
                )
            )

            brier = (
                brier_score_loss(
                    y_val,
                    p,
                )
            )

            result = {
                "model": model_name,

                "training_window":
                    window_name,

                "train_rows":
                    len(train),

                "train_start":
                    train[
                        "order_purchase_timestamp"
                    ].min(),

                "train_end":
                    train[
                        "order_purchase_timestamp"
                    ].max(),

                "validation_start":
                    validation[
                        "order_purchase_timestamp"
                    ].min(),

                "validation_end":
                    validation[
                        "order_purchase_timestamp"
                    ].max(),

                "train_positive_rate":
                    y_train.mean(),

                "validation_positive_rate":
                    y_val.mean(),

                "mean_prediction":
                    p.mean(),

                "average_precision":
                    ap,

                "roc_auc":
                    auc,

                "brier_score":
                    brier,
            }

            results.append(result)

            print(
                f"{model_name:14s} "
                f"{window_name:12s} "
                f"| AP={ap:.4f} "
                f"| AUC={auc:.4f} "
                f"| actual={y_val.mean():.4f} "
                f"| pred={p.mean():.4f}"
            )


# --------------------------------------------------
# Summary
# --------------------------------------------------

results_df = pd.DataFrame(
    results
)

summary = (
    results_df
    .groupby(
        [
            "model",
            "training_window",
        ]
    )
    .agg(
        mean_ap=(
            "average_precision",
            "mean",
        ),
        std_ap=(
            "average_precision",
            "std",
        ),
        mean_auc=(
            "roc_auc",
            "mean",
        ),
        std_auc=(
            "roc_auc",
            "std",
        ),
        mean_brier=(
            "brier_score",
            "mean",
        ),
        mean_actual_rate=(
            "validation_positive_rate",
            "mean",
        ),
        mean_predicted_rate=(
            "mean_prediction",
            "mean",
        ),
    )
    .sort_values(
        "mean_ap",
        ascending=False,
    )
)


print(
    "\n===== RECENCY BACKTEST SUMMARY ====="
)

print(
    summary.to_string()
)


results_df.to_csv(
    ROOT
    / "reports"
    / "recency_backtest_folds.csv",
    index=False,
)

summary.to_csv(
    ROOT
    / "reports"
    / "recency_backtest_summary.csv"
)