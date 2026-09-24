from pathlib import Path
import json

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "final_logistic_v2.joblib"
)

THRESHOLD_PATH = (
    PROJECT_ROOT
    / "reports"
    / "model_v2_threshold.json"
)


def load_artifacts():
    """
    Load the frozen sklearn pipeline and
    the selected decision threshold.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(
            f"Threshold file not found: "
            f"{THRESHOLD_PATH}"
        )

    model = joblib.load(
        MODEL_PATH
    )

    with open(
        THRESHOLD_PATH,
        "r",
    ) as f:
        threshold_data = json.load(f)

    threshold = float(
        threshold_data["threshold"]
    )

    return model, threshold


# --------------------------------------------------
# Load frozen artifacts once when the service starts
# --------------------------------------------------

MODEL, THRESHOLD = load_artifacts()


# --------------------------------------------------
# Read expected features DIRECTLY from fitted model
# --------------------------------------------------

try:
    MODEL_FEATURES = (
        MODEL
        .named_steps["preprocessor"]
        .feature_names_in_
        .tolist()
    )

except (
    AttributeError,
    KeyError,
):
    raise RuntimeError(
        "Unable to recover model input features "
        "from the fitted pipeline."
    )


def predict_delivery_risk(
    order_features: dict,
):
    """
    Predict late-delivery probability for one order.

    The input dictionary must contain exactly the
    features expected by the frozen model pipeline.
    """

    # --------------------------------------------------
    # Validate required features
    # --------------------------------------------------

    missing_features = [
        feature
        for feature in MODEL_FEATURES
        if feature not in order_features
    ]

    if missing_features:
        raise ValueError(
            "Missing required model features: "
            f"{missing_features}"
        )

    # --------------------------------------------------
    # Build one-row DataFrame in the exact feature order
    # used during training.
    # --------------------------------------------------

    row = pd.DataFrame(
        [
            {
                feature:
                    order_features.get(feature)

                for feature
                in MODEL_FEATURES
            }
        ],
        columns=MODEL_FEATURES,
    )

    # --------------------------------------------------
    # Probability prediction
    # --------------------------------------------------

    probability = float(
        MODEL.predict_proba(
            row
        )[0, 1]
    )

    prediction = int(
        probability >= THRESHOLD
    )

    # --------------------------------------------------
    # Response
    # --------------------------------------------------

    return {
        "late_delivery_probability":
            probability,

        "prediction":
            prediction,

        "risk":
            (
                "high"
                if prediction == 1
                else "lower"
            ),

        "threshold":
            THRESHOLD,
    }


if __name__ == "__main__":

    print(
        "Model loaded successfully."
    )

    print(
        f"Threshold: {THRESHOLD:.4f}"
    )

    print(
        f"Expected features: "
        f"{len(MODEL_FEATURES)}"
    )

    for feature in MODEL_FEATURES:
        print(
            f"  - {feature}"
        )