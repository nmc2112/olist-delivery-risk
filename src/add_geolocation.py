from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

FEATURE_PATH = PROCESSED_DIR / "model_features.csv"

GEO_PATH = (
    RAW_DIR
    / "olist_geolocation_dataset.csv"
)

OUTPUT_PATH = (
    PROCESSED_DIR
    / "model_features_geo.csv"
)


def haversine_distance(
    lat1,
    lon1,
    lat2,
    lon2,
):
    """
    Calculate distance in kilometers between
    two latitude/longitude coordinates.
    """

    earth_radius_km = 6371.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    a = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(delta_lon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    return earth_radius_km * c


def build_zip_coordinates(geolocation):
    """
    Convert multiple geolocation records per ZIP code
    into one representative coordinate per ZIP.
    """

    zip_coordinates = (
        geolocation
        .groupby(
            "geolocation_zip_code_prefix"
        )
        .agg(
            latitude=(
                "geolocation_lat",
                "median",
            ),
            longitude=(
                "geolocation_lng",
                "median",
            ),
        )
        .reset_index()
    )

    return zip_coordinates


def main():

    features = pd.read_csv(
        FEATURE_PATH
    )

    geolocation = pd.read_csv(
        GEO_PATH
    )

    print("=== GEOLOCATION PROCESSING ===")

    print(
        f"Raw geolocation rows: "
        f"{len(geolocation)}"
    )

    print(
        f"Raw duplicate rows: "
        f"{geolocation.duplicated().sum()}"
    )

    zip_coordinates = (
        build_zip_coordinates(
            geolocation
        )
    )

    print(
        f"Unique ZIP codes: "
        f"{len(zip_coordinates)}"
    )

    # -------------------------
    # Customer coordinates
    # -------------------------

    customer_geo = (
        zip_coordinates.rename(
            columns={
                "geolocation_zip_code_prefix":
                    "customer_zip_code_prefix",
                "latitude":
                    "customer_latitude",
                "longitude":
                    "customer_longitude",
            }
        )
    )

    features = features.merge(
        customer_geo,
        on="customer_zip_code_prefix",
        how="left",
        validate="many_to_one",
    )

    # -------------------------
    # Seller coordinates
    # -------------------------

    seller_geo = (
        zip_coordinates.rename(
            columns={
                "geolocation_zip_code_prefix":
                    "primary_seller_zip_code_prefix",
                "latitude":
                    "seller_latitude",
                "longitude":
                    "seller_longitude",
            }
        )
    )

    features = features.merge(
        seller_geo,
        on="primary_seller_zip_code_prefix",
        how="left",
        validate="many_to_one",
    )

    # -------------------------
    # Distance
    # -------------------------

    features["customer_seller_distance_km"] = (
        haversine_distance(
            features["customer_latitude"],
            features["customer_longitude"],
            features["seller_latitude"],
            features["seller_longitude"],
        )
    )

    print("\n=== DISTANCE FEATURE ===")

    print(
        features[
            "customer_seller_distance_km"
        ].describe()
    )

    print(
        "\nMissing distances:",
        features[
            "customer_seller_distance_km"
        ].isna().sum(),
    )

    print(
        "\nDuplicate order IDs:",
        features[
            "order_id"
        ].duplicated().sum(),
    )

    # Coordinates were useful to calculate distance,
    # but we do not need them as model inputs.
    features = features.drop(
        columns=[
            "customer_latitude",
            "customer_longitude",
            "seller_latitude",
            "seller_longitude",
        ]
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    features.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nFinal rows: {len(features)}"
    )

    print(
        f"Final columns: "
        f"{features.shape[1]}"
    )

    print(
        f"\nSaved to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()