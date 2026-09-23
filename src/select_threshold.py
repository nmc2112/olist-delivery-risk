from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

from modeling_utils import (
    load_modeling_data,
    chronological_split,
    prepare_xy,
    build_preprocessor,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PARAMS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "best_random_forest_params.json"
)

THRESHOLD_PATH = (
    PROJECT_ROOT
    / "reports"
    / "selected_threshold.json"
)

THRESHOLD_TABLE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "threshold_candidates.csv"
)


def load_best_params():

    with open(PARAMS_PATH) as f:
        tuning_results = json.load(f)

    params = tuning_results["best_params"]

    # Remove Pipeline prefix
    rf_params = {
        key.replace("model__", ""): value
        for key, value in params.items()
    }

    return rf_params


def main():

    print("=== THRESHOLD SELECTION ===")

    # -----------------------------------------
    # Development data only
    # -----------------------------------------

    df = load_modeling_data()

    dev, _ = chronological_split(df)

    X_dev, y_dev = prepare_xy(dev)

    print(f"Development rows: {len(X_dev):,}")

    # -----------------------------------------
    # Tuned RF
    # -----------------------------------------

    rf_params = load_best_params()

    print("\nUsing tuned parameters:")

    for key, value in rf_params.items():
        print(f"{key}: {value}")

    # -----------------------------------------
    # Time-aware out-of-fold predictions
    # -----------------------------------------

    cv = TimeSeriesSplit(
        n_splits=5
    )

    all_y = []
    all_probabilities = []

    print(
        "\n=== GENERATING OUT-OF-TIME "
        "VALIDATION PREDICTIONS ==="
    )

    for fold, (
        train_index,
        valid_index,
    ) in enumerate(
        cv.split(X_dev),
        start=1,
    ):

        X_train = (
            X_dev.iloc[train_index]
        )

        y_train = (
            y_dev.iloc[train_index]
        )

        X_valid = (
            X_dev.iloc[valid_index]
        )

        y_valid = (
            y_dev.iloc[valid_index]
        )

        preprocessor, _, _ = (
            build_preprocessor(X_train)
        )

        model = RandomForestClassifier(
            **rf_params,
            random_state=2026,
            n_jobs=-1,
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

        probabilities = (
            pipeline.predict_proba(
                X_valid
            )[:, 1]
        )

        fold_ap = average_precision_score(
            y_valid,
            probabilities,
        )

        print(
            f"Fold {fold}: "
            f"train={len(train_index):,}, "
            f"validation={len(valid_index):,}, "
            f"AP={fold_ap:.4f}"
        )

        all_y.append(
            y_valid.to_numpy()
        )

        all_probabilities.append(
            probabilities
        )

    y_oof = np.concatenate(
        all_y
    )

    probabilities_oof = np.concatenate(
        all_probabilities
    )

    # -----------------------------------------
    # Ranking metrics
    # -----------------------------------------

    ap = average_precision_score(
        y_oof,
        probabilities_oof,
    )

    roc_auc = roc_auc_score(
        y_oof,
        probabilities_oof,
    )

    print(
        "\n=== OUT-OF-TIME VALIDATION METRICS ==="
    )

    print(
        f"Average Precision: {ap:.4f}"
    )

    print(
        f"ROC-AUC: {roc_auc:.4f}"
    )

    # -----------------------------------------
    # Precision / Recall curve
    # -----------------------------------------

    precision, recall, thresholds = (
        precision_recall_curve(
            y_oof,
            probabilities_oof,
        )
    )

    # precision/recall contain one extra element
    precision_at_threshold = (
        precision[:-1]
    )

    recall_at_threshold = (
        recall[:-1]
    )

    f1 = (
        2
        * precision_at_threshold
        * recall_at_threshold
        / (
            precision_at_threshold
            + recall_at_threshold
            + 1e-12
        )
    )

    table = pd.DataFrame(
        {
            "threshold": thresholds,
            "precision":
                precision_at_threshold,
            "recall":
                recall_at_threshold,
            "f1":
                f1,
        }
    )

    # -----------------------------------------
    # Best F1 threshold
    # -----------------------------------------

    best_index = (
        table["f1"].idxmax()
    )

    best_row = (
        table.loc[best_index]
    )

    print(
        "\n=== BEST F1 THRESHOLD ==="
    )

    print(
        f"Threshold: "
        f"{best_row['threshold']:.4f}"
    )

    print(
        f"Precision: "
        f"{best_row['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{best_row['recall']:.4f}"
    )

    print(
        f"F1: "
        f"{best_row['f1']:.4f}"
    )

    # -----------------------------------------
    # Business-oriented recall targets
    # -----------------------------------------

    print(
        "\n=== RECALL TARGET OPTIONS ==="
    )

    recall_targets = [
        0.40,
        0.50,
        0.60,
        0.70,
    ]

    recall_options = {}

    for target_recall in recall_targets:

        candidates = table[
            table["recall"]
            >= target_recall
        ]

        if candidates.empty:
            continue

        # Highest precision while meeting recall target
        selected = (
            candidates
            .sort_values(
                "precision",
                ascending=False,
            )
            .iloc[0]
        )

        key = (
            f"recall_{int(target_recall * 100)}"
        )

        recall_options[key] = {
            "threshold":
                float(
                    selected["threshold"]
                ),
            "precision":
                float(
                    selected["precision"]
                ),
            "recall":
                float(
                    selected["recall"]
                ),
            "f1":
                float(
                    selected["f1"]
                ),
        }

        print(
            f"Recall >= "
            f"{target_recall:.0%}: "
            f"threshold="
            f"{selected['threshold']:.4f}, "
            f"precision="
            f"{selected['precision']:.4f}, "
            f"recall="
            f"{selected['recall']:.4f}, "
            f"F1="
            f"{selected['f1']:.4f}"
        )

    # -----------------------------------------
    # Save results
    # -----------------------------------------

    THRESHOLD_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "selection_method":
            "maximum_f1_on_time_based_validation",

        "threshold":
            float(
                best_row["threshold"]
            ),

        "precision":
            float(
                best_row["precision"]
            ),

        "recall":
            float(
                best_row["recall"]
            ),

        "f1":
            float(
                best_row["f1"]
            ),

        "average_precision":
            float(ap),

        "roc_auc":
            float(roc_auc),

        "recall_target_options":
            recall_options,
    }

    with open(
        THRESHOLD_PATH,
        "w",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
        )

    table.to_csv(
        THRESHOLD_TABLE_PATH,
        index=False,
    )

    print(
        f"\nSaved threshold to: "
        f"{THRESHOLD_PATH}"
    )

    print(
        f"Saved threshold table to: "
        f"{THRESHOLD_TABLE_PATH}"
    )

    print(
        "\nFinal test set has NOT been used."
    )


if __name__ == "__main__":
    main()