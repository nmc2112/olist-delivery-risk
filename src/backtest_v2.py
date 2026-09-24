from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_features_geo.csv"
)

TARGET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "order_targets.csv"
)


# --------------------------------------------------
# Load data
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
    df
    .sort_values("order_purchase_timestamp")
    .reset_index(drop=True)
)


# --------------------------------------------------
# Keep original 80% development period only
# --------------------------------------------------

split_idx = int(len(df) * 0.80)

dev = df.iloc[:split_idx].copy()


# --------------------------------------------------
# Additional safe calendar features
# --------------------------------------------------

dev["purchase_year"] = (
    dev["order_purchase_timestamp"].dt.year
)

dev["purchase_quarter"] = (
    dev["order_purchase_timestamp"].dt.quarter
)

start_date = (
    dev["order_purchase_timestamp"].min()
)

dev["days_since_start"] = (
    dev["order_purchase_timestamp"]
    - start_date
).dt.days


# --------------------------------------------------
# X / y
# --------------------------------------------------

DROP_COLUMNS = [
    "late_delivery",
    "order_id",
    "order_purchase_timestamp",
    "customer_zip_code_prefix",
    "primary_seller_zip_code_prefix",
]

y = dev["late_delivery"].astype(int)

X = dev.drop(
    columns=[
        col
        for col in DROP_COLUMNS
        if col in dev.columns
    ]
)


categorical_features = (
    X.select_dtypes(
        include=["object"]
    )
    .columns
    .tolist()
)

numerical_features = [
    col
    for col in X.columns
    if col not in categorical_features
]


# --------------------------------------------------
# Preprocessing
# --------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
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
    steps=[
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
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numerical_features,
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features,
        ),
    ]
)


# --------------------------------------------------
# Models
# --------------------------------------------------

models = {
    "dummy": DummyClassifier(
        strategy="prior"
    ),

    "logistic": LogisticRegression(
        max_iter=2000,
        random_state=2026,
    ),

    "logistic_balanced": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=2026,
    ),

    "random_forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=30,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=2026,
    ),

    "random_forest_balanced": RandomForestClassifier(
        n_estimators=200,
        max_depth=30,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=2026,
    ),

    "extra_trees": ExtraTreesClassifier(
        n_estimators=200,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=2026,
    ),
}


# --------------------------------------------------
# Time-aware backtest
# --------------------------------------------------

tscv = TimeSeriesSplit(
    n_splits=5
)

results = []


for model_name, model in models.items():

    print(
        f"\n===== {model_name} ====="
    )

    fold = 0

    for train_idx, val_idx in tscv.split(X):

        fold += 1

        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        train_dates = (
            dev.iloc[train_idx]
            ["order_purchase_timestamp"]
        )

        val_dates = (
            dev.iloc[val_idx]
            ["order_purchase_timestamp"]
        )

        pipeline = Pipeline(
            steps=[
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

        pipeline.fit(
            X_train,
            y_train,
        )

        probability = (
            pipeline.predict_proba(
                X_val
            )[:, 1]
        )

        ap = average_precision_score(
            y_val,
            probability,
        )

        auc = roc_auc_score(
            y_val,
            probability,
        )

        result = {
            "model": model_name,
            "fold": fold,

            "train_start":
                train_dates.min(),

            "train_end":
                train_dates.max(),

            "validation_start":
                val_dates.min(),

            "validation_end":
                val_dates.max(),

            "train_positive_rate":
                y_train.mean(),

            "validation_positive_rate":
                y_val.mean(),

            "mean_prediction":
                probability.mean(),

            "average_precision":
                ap,

            "roc_auc":
                auc,
        }

        results.append(result)

        print(
            f"Fold {fold}: "
            f"AP={ap:.4f} | "
            f"AUC={auc:.4f} | "
            f"actual={y_val.mean():.4f} | "
            f"pred={probability.mean():.4f}"
        )


# --------------------------------------------------
# Summary
# --------------------------------------------------

results_df = pd.DataFrame(
    results
)

summary = (
    results_df
    .groupby("model")
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
    )
    .sort_values(
        "mean_ap",
        ascending=False,
    )
)


print(
    "\n===== SUMMARY ====="
)

print(summary)


reports_dir = (
    PROJECT_ROOT
    / "reports"
)

results_df.to_csv(
    reports_dir
    / "backtest_v2_folds.csv",
    index=False,
)

summary.to_csv(
    reports_dir
    / "backtest_v2_summary.csv"
)