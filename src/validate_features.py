from pathlib import Path

import pandas as pd
import great_expectations as gx


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_features_geo.csv"
)


def main():

    print("=== GREAT EXPECTATIONS VALIDATION ===")

    # --------------------------------------------------
    # Load final feature table
    # --------------------------------------------------

    df = pd.read_csv(DATA_PATH)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {df.shape[1]}")

    # --------------------------------------------------
    # Create temporary GX context
    # --------------------------------------------------

    context = gx.get_context(
        mode="ephemeral"
    )

    # --------------------------------------------------
    # Data Source
    # --------------------------------------------------

    data_source = (
        context.data_sources.add_pandas(
            name="olist_features_source"
        )
    )

    # --------------------------------------------------
    # Data Asset
    # --------------------------------------------------

    data_asset = (
        data_source.add_dataframe_asset(
            name="delivery_features"
        )
    )

    # --------------------------------------------------
    # Batch Definition
    # --------------------------------------------------

    batch_definition = (
        data_asset
        .add_batch_definition_whole_dataframe(
            "full_feature_table"
        )
    )

    # --------------------------------------------------
    # Expectation Suite
    # --------------------------------------------------

    suite = gx.ExpectationSuite(
        name="delivery_feature_quality"
    )

    suite = context.suites.add(suite)

    # --------------------------------------------------
    # Core structural expectations
    # --------------------------------------------------

    suite.add_expectation(
        gx.expectations.ExpectTableRowCountToBeBetween(
            min_value=96000,
            max_value=97000,
        )
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="order_id"
        )
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeUnique(
            column="order_id"
        )
    )

    # --------------------------------------------------
    # Target validation
    # --------------------------------------------------

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="late_delivery"
        )
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="late_delivery",
            value_set=[0, 1],
        )
    )

    # --------------------------------------------------
    # Business-valid numerical ranges
    # --------------------------------------------------

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="item_count",
            min_value=1,
        )
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="total_item_value",
            min_value=0,
        )
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="total_freight_value",
            min_value=0,
        )
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="estimated_delivery_days",
            min_value=0,
        )
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="customer_seller_distance_km",
            min_value=0,
            severity="warning",
        )
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="max_payment_installments",
            min_value=0,
        )
    )

    # --------------------------------------------------
    # Validation Definition
    # --------------------------------------------------

    validation_definition = gx.ValidationDefinition(
        name="delivery_features_validation",
        data=batch_definition,
        suite=suite,
    )

    validation_definition = (
        context.validation_definitions.add(
            validation_definition
        )
    )

    # --------------------------------------------------
    # Run validation
    # --------------------------------------------------

    result = validation_definition.run(
        batch_parameters={
            "dataframe": df
        }
    )

    print("\n=== VALIDATION RESULT ===")
    print(
        f"Success: {result.success}"
    )

    print(
        f"Expectations evaluated: "
        f"{len(result.results)}"
    )

    failed = [
        r
        for r in result.results
        if not r.success
    ]

    print(
        f"Failed expectations: "
        f"{len(failed)}"
    )

    if failed:

        print("\n=== FAILURES ===")

        for failure in failed:

            config = (
                failure
                .expectation_config
            )

            print(
                f"- {config.type}"
            )

            print(
                f"  kwargs: "
                f"{config.kwargs}"
            )

    if not result.success:
        raise RuntimeError(
            "Data validation failed."
        )

    print(
        "\nData quality checks passed."
    )


if __name__ == "__main__":
    main()