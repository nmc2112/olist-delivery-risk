from pathlib import Path
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    RandomizedSearchCV,
    TimeSeriesSplit,
)
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
    / "best_random_forest_params.json"
)


def main():

    print("=== RANDOM FOREST TUNING ===")

    # -----------------------------------------
    # Development data only
    # -----------------------------------------

    df = load_modeling_data()

    dev, _ = chronological_split(df)

    X_dev, y_dev = prepare_xy(dev)

    print(f"Development rows: {len(X_dev):,}")
    print(
        f"Positive rate: "
        f"{y_dev.mean() * 100:.2f}%"
    )

    preprocessor, _, _ = (
        build_preprocessor(X_dev)
    )

    # -----------------------------------------
    # Model
    # -----------------------------------------

    model = RandomForestClassifier(
        random_state=2026,

        # Avoid nested parallelism.
        # Search itself will parallelize.
        n_jobs=1,
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

    # -----------------------------------------
    # Search space
    # -----------------------------------------

    param_distributions = {

        "model__n_estimators": [
            200,
            300,
            500,
        ],

        "model__max_depth": [
            None,
            10,
            20,
            30,
        ],

        "model__min_samples_split": [
            2,
            5,
            10,
        ],

        "model__min_samples_leaf": [
            1,
            2,
            5,
            10,
        ],

        "model__max_features": [
            "sqrt",
            "log2",
            0.5,
        ],

        "model__class_weight": [
            "balanced",
            "balanced_subsample",
            None,
        ],
    }

    # -----------------------------------------
    # Time-aware CV
    # -----------------------------------------

    cv = TimeSeriesSplit(
        n_splits=4
    )

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,

        # Enough to explore without making
        # your Intel Mac suffer unnecessarily.
        n_iter=12,

        scoring="average_precision",

        cv=cv,

        random_state=2026,

        n_jobs=-1,

        verbose=2,

        return_train_score=False,
    )

    # -----------------------------------------
    # Tune
    # -----------------------------------------

    search.fit(
        X_dev,
        y_dev,
    )

    print(
        "\n=== BEST RESULT ==="
    )

    print(
        f"Best CV AP: "
        f"{search.best_score_:.4f}"
    )

    print(
        "\nBest parameters:"
    )

    for key, value in (
        search.best_params_.items()
    ):
        print(
            f"{key}: {value}"
        )

    # -----------------------------------------
    # Save
    # -----------------------------------------

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "best_cv_average_precision":
            float(search.best_score_),

        "best_params":
            search.best_params_,
    }

    with open(
        REPORT_PATH,
        "w",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
        )

    print(
        f"\nSaved to: {REPORT_PATH}"
    )

    print(
        "\nFinal test set has NOT been used."
    )


if __name__ == "__main__":
    main()