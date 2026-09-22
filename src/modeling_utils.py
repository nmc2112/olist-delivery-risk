from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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


TARGET_COLUMN = "late_delivery"


DROP_FEATURE_COLUMNS = [
    "order_id",

    # High-cardinality location identifiers.
    # We already derived state + geographic distance.
    "customer_zip_code_prefix",
    "primary_seller_zip_code_prefix",
]


def load_modeling_data():
    """
    Load feature table and recover purchase timestamp
    only for chronological splitting.
    """

    features = pd.read_csv(
        FEATURE_PATH
    )

    order_dates = pd.read_csv(
        TARGET_PATH,
        usecols=[
            "order_id",
            "order_purchase_timestamp",
        ],
        parse_dates=[
            "order_purchase_timestamp"
        ],
    )

    df = features.merge(
        order_dates,
        on="order_id",
        how="left",
        validate="one_to_one",
    )

    if df["order_purchase_timestamp"].isna().any():
        raise RuntimeError(
            "Some orders are missing purchase timestamps."
        )

    df = df.sort_values(
        "order_purchase_timestamp"
    ).reset_index(drop=True)

    return df


def chronological_split(
    df,
    test_size=0.20,
):
    """
    Last 20% chronologically = locked final test set.
    """

    split_index = int(
        len(df) * (1 - test_size)
    )

    dev = (
        df.iloc[:split_index]
        .copy()
    )

    test = (
        df.iloc[split_index:]
        .copy()
    )

    return dev, test


def prepare_xy(df):
    """
    Remove target, IDs and timestamp metadata.
    """

    y = df[TARGET_COLUMN].copy()

    X = df.drop(
        columns=[
            TARGET_COLUMN,
            "order_purchase_timestamp",
            *DROP_FEATURE_COLUMNS,
        ],
        errors="ignore",
    )

    return X, y


def build_preprocessor(X):
    """
    Numerical:
      median imputation + scaling

    Categorical:
      most-frequent imputation + one-hot encoding
    """

    numeric_columns = (
        X.select_dtypes(
            include=["number"]
        )
        .columns
        .tolist()
    )

    categorical_columns = (
        X.select_dtypes(
            exclude=["number"]
        )
        .columns
        .tolist()
    )

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
                numeric_columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
        ]
    )

    return (
        preprocessor,
        numeric_columns,
        categorical_columns,
    )


def print_split_summary(
    dev,
    test,
):
    print("=== CHRONOLOGICAL SPLIT ===")

    print(
        f"Total rows: {len(dev) + len(test):,}"
    )

    print(
        f"Development rows: {len(dev):,}"
    )

    print(
        f"Final test rows: {len(test):,}"
    )

    print(
        "\nDevelopment period:"
    )

    print(
        dev["order_purchase_timestamp"].min(),
        "→",
        dev["order_purchase_timestamp"].max(),
    )

    print(
        "\nFinal test period:"
    )

    print(
        test["order_purchase_timestamp"].min(),
        "→",
        test["order_purchase_timestamp"].max(),
    )

    print(
        "\nDevelopment late-delivery rate:"
    )

    print(
        f"{dev[TARGET_COLUMN].mean() * 100:.2f}%"
    )

    print(
        "\nFinal test late-delivery rate:"
    )

    print(
        f"{test[TARGET_COLUMN].mean() * 100:.2f}%"
    )


if __name__ == "__main__":

    df = load_modeling_data()

    dev, test = chronological_split(df)

    X_dev, y_dev = prepare_xy(dev)
    X_test, y_test = prepare_xy(test)

    (
        preprocessor,
        numeric_columns,
        categorical_columns,
    ) = build_preprocessor(X_dev)

    print_split_summary(
        dev,
        test,
    )

    print(
        f"\nModel input features: "
        f"{X_dev.shape[1]}"
    )

    print(
        f"Numerical features: "
        f"{len(numeric_columns)}"
    )

    print(
        f"Categorical features: "
        f"{len(categorical_columns)}"
    )

    print(
        "\nExcluded from model:"
    )

    print(
        [
            TARGET_COLUMN,
            "order_id",
            "order_purchase_timestamp",
            "customer_zip_code_prefix",
            "primary_seller_zip_code_prefix",
        ]
    )