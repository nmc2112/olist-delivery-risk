from pathlib import Path
import json

import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
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

RESULTS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "final_test_results.json"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "final_random_forest.joblib"
)


def load_best_params():

    with open(PARAMS_PATH) as f:
        tuning_results = json.load(f)

    params = tuning_results["best_params"]

    return {
        key.replace("model__", ""): value
        for key, value in params.items()
    }


def load_threshold():

    with open(THRESHOLD_PATH) as f:
        threshold_results = json.load(f)

    return float(
        threshold_results["threshold"]
    )


def main():

    print("=== FINAL MODEL EVALUATION ===")

    # --------------------------------------------------
    # Locked chronological split
    # --------------------------------------------------

    df = load_modeling_data()

    dev, test = chronological_split(df)

    X_dev, y_dev = prepare_xy(dev)
    X_test, y_test = prepare_xy(test)

    print(
        f"Development rows: {len(X_dev):,}"
    )

    print(
        f"Final test rows: {len(X_test):,}"
    )

    print(
        f"Development positive rate: "
        f"{y_dev.mean() * 100:.2f}%"
    )

    print(
        f"Final test positive rate: "
        f"{y_test.mean() * 100:.2f}%"
    )

    # --------------------------------------------------
    # Frozen configuration
    # --------------------------------------------------

    rf_params = load_best_params()
    threshold = load_threshold()

    print(
        f"\nSelected threshold: "
        f"{threshold:.4f}"
    )

    print(
        "\nTuned parameters:"
    )

    for key, value in rf_params.items():
        print(
            f"{key}: {value}"
        )

    # --------------------------------------------------
    # Final pipeline
    # --------------------------------------------------

    preprocessor, _, _ = (
        build_preprocessor(X_dev)
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

    print(
        "\nTraining final model "
        "on all development data..."
    )

    pipeline.fit(
        X_dev,
        y_dev,
    )

    # --------------------------------------------------
    # ONE evaluation on locked test set
    # --------------------------------------------------

    probabilities = (
        pipeline.predict_proba(
            X_test
        )[:, 1]
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    # Ranking metrics
    average_precision = (
        average_precision_score(
            y_test,
            probabilities,
        )
    )

    roc_auc = (
        roc_auc_score(
            y_test,
            probabilities,
        )
    )

    # Threshold-dependent metrics
    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_test,
        predictions,
    )

    print(
        "\n=== FINAL TEST RESULTS ==="
    )

    print(
        f"Average Precision: "
        f"{average_precision:.4f}"
    )

    print(
        f"ROC-AUC: "
        f"{roc_auc:.4f}"
    )

    print(
        f"Precision: "
        f"{precision:.4f}"
    )

    print(
        f"Recall: "
        f"{recall:.4f}"
    )

    print(
        f"F1: "
        f"{f1:.4f}"
    )

    print(
        "\nConfusion matrix:"
    )

    print(cm)

    print(
        "\nClassification report:"
    )

    print(
        classification_report(
            y_test,
            predictions,
            digits=4,
            zero_division=0,
        )
    )

    # --------------------------------------------------
    # Save model + metrics
    # --------------------------------------------------

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        pipeline,
        MODEL_PATH,
    )

    tn, fp, fn, tp = (
        cm.ravel()
    )

    output = {
        "threshold":
            threshold,

        "test_rows":
            int(len(X_test)),

        "test_positive_rate":
            float(y_test.mean()),

        "average_precision":
            float(average_precision),

        "roc_auc":
            float(roc_auc),

        "precision":
            float(precision),

        "recall":
            float(recall),

        "f1":
            float(f1),

        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },

        "best_params":
            rf_params,
    }

    with open(
        RESULTS_PATH,
        "w",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    print(
        f"\nModel saved to: "
        f"{MODEL_PATH}"
    )

    print(
        f"Results saved to: "
        f"{RESULTS_PATH}"
    )

    print(
        "\nFinal test evaluation complete."
    )


if __name__ == "__main__":
    main()