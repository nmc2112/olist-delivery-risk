from pathlib import Path

import joblib
import pandas as pd

from sklearn.inspection import permutation_importance

from modeling_utils import (
    load_modeling_data,
    chronological_split,
    prepare_xy,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "final_random_forest.joblib"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "permutation_importance.csv"
)


def main():

    print("=== MODEL INTERPRETATION ===")

    # --------------------------------------------------
    # Load locked test set
    # --------------------------------------------------

    df = load_modeling_data()

    _, test = chronological_split(df)

    X_test, y_test = prepare_xy(test)

    print(
        f"Final test rows: "
        f"{len(X_test):,}"
    )

    # --------------------------------------------------
    # Load frozen model
    # --------------------------------------------------

    pipeline = joblib.load(
        MODEL_PATH
    )

    # --------------------------------------------------
    # Permutation importance
    # --------------------------------------------------

    print(
        "\nCalculating permutation importance..."
    )

    result = permutation_importance(
        pipeline,
        X_test,
        y_test,

        # Same primary metric as model selection
        scoring="average_precision",

        n_repeats=5,

        random_state=2026,

        n_jobs=-1,
    )

    importance_df = pd.DataFrame(
        {
            "feature":
                X_test.columns,

            "importance_mean":
                result.importances_mean,

            "importance_std":
                result.importances_std,
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            "importance_mean",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print(
        "\n=== TOP FEATURES ==="
    )

    print(
        importance_df
        .head(20)
        .to_string(
            index=False
        )
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    importance_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nSaved to: "
        f"{OUTPUT_PATH}"
    )

    print(
        "\nNOTE:"
        "\nThese test-set importances are "
        "for interpretation only."
        "\nThey must not be used to "
        "retune or retrain the model."
    )


if __name__ == "__main__":
    main()