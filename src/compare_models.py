from pathlib import Path

import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline

from modeling_utils import (
    load_modeling_data,
    chronological_split,
    prepare_xy,
    build_preprocessor,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "model_comparison.csv"
)


def main():

    print("=== MODEL COMPARISON ===")

    # --------------------------------------------------
    # Load and split
    # --------------------------------------------------

    df = load_modeling_data()

    dev, _ = chronological_split(df)

    X_dev, y_dev = prepare_xy(dev)

    (
        preprocessor,
        numeric_columns,
        categorical_columns,
    ) = build_preprocessor(X_dev)

    print(f"Development rows: {len(X_dev):,}")
    print(
        f"Positive rate: "
        f"{y_dev.mean() * 100:.2f}%"
    )

    print(
        f"Numerical features: "
        f"{len(numeric_columns)}"
    )

    print(
        f"Categorical features: "
        f"{len(categorical_columns)}"
    )

    # --------------------------------------------------
    # Time-aware CV
    # --------------------------------------------------

    cv = TimeSeriesSplit(
        n_splits=5
    )

    # --------------------------------------------------
    # Models
    # --------------------------------------------------

    models = {

        "Dummy":

            DummyClassifier(
                strategy="prior"
            ),

        "Logistic Regression":

            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=2026,
            ),

        "Random Forest":

            RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced_subsample",
                random_state=2026,
                n_jobs=-1,
                min_samples_leaf=2,
            ),
    }

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    scoring = {
        "average_precision":
            "average_precision",

        "roc_auc":
            "roc_auc",

        "precision":
            "precision",

        "recall":
            "recall",

        "f1":
            "f1",
    }

    results = []

    # --------------------------------------------------
    # Cross-validation
    # --------------------------------------------------

    for name, model in models.items():

        print(
            f"\nEvaluating: {name}"
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

        scores = cross_validate(
            pipeline,
            X_dev,
            y_dev,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )

        result = {
            "model": name,

            "average_precision_mean":
                scores[
                    "test_average_precision"
                ].mean(),

            "average_precision_std":
                scores[
                    "test_average_precision"
                ].std(),

            "roc_auc_mean":
                scores[
                    "test_roc_auc"
                ].mean(),

            "roc_auc_std":
                scores[
                    "test_roc_auc"
                ].std(),

            "precision_mean":
                scores[
                    "test_precision"
                ].mean(),

            "recall_mean":
                scores[
                    "test_recall"
                ].mean(),

            "f1_mean":
                scores[
                    "test_f1"
                ].mean(),
        }

        results.append(result)

        print(
            f"AP: "
            f"{result['average_precision_mean']:.4f}"
            f" ± "
            f"{result['average_precision_std']:.4f}"
        )

        print(
            f"ROC-AUC: "
            f"{result['roc_auc_mean']:.4f}"
        )

        print(
            f"Precision: "
            f"{result['precision_mean']:.4f}"
        )

        print(
            f"Recall: "
            f"{result['recall_mean']:.4f}"
        )

        print(
            f"F1: "
            f"{result['f1_mean']:.4f}"
        )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    results_df = (
        pd.DataFrame(results)
        .sort_values(
            "average_precision_mean",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print(
        "\n=== CROSS-VALIDATION SUMMARY ==="
    )

    print(
        results_df[
            [
                "model",
                "average_precision_mean",
                "roc_auc_mean",
                "precision_mean",
                "recall_mean",
                "f1_mean",
            ]
        ].to_string(
            index=False
        )
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        REPORT_PATH,
        index=False,
    )

    print(
        f"\nSaved to: {REPORT_PATH}"
    )

    print(
        "\nFinal test set has NOT been used."
    )


if __name__ == "__main__":
    main()