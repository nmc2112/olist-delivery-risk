from modeling_utils import (
    load_modeling_data,
    chronological_split,
    prepare_xy,
)


FORBIDDEN_FEATURES = {
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "review_score",
    "review_comment_title",
    "review_comment_message",
    "review_creation_date",
    "review_answer_timestamp",
    "order_status",
}


def main():

    df = load_modeling_data()

    dev, test = chronological_split(df)

    X_dev, _ = prepare_xy(dev)

    present_forbidden = (
        FORBIDDEN_FEATURES
        .intersection(
            X_dev.columns
        )
    )

    print(
        "=== LEAKAGE AUDIT ==="
    )

    print(
        f"Features checked: "
        f"{X_dev.shape[1]}"
    )

    print(
        "Forbidden post-outcome features found:",
        sorted(present_forbidden),
    )

    if present_forbidden:
        raise RuntimeError(
            "Leakage detected."
        )

    print(
        "\nLeakage audit passed."
    )


if __name__ == "__main__":
    main()