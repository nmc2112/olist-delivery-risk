from pathlib import Path
import subprocess

import pendulum

from airflow.sdk import dag, task


PROJECT_ROOT = Path(
    "/opt/airflow/project"
)

PROJECT_PYTHON = Path(
    "/opt/airflow/project-venv/bin/python"
)


def run_script(relative_path):

    script_path = (
        PROJECT_ROOT
        / relative_path
    )

    print(
        f"Running {script_path}"
    )

    subprocess.run(
        [
            str(PROJECT_PYTHON),
            str(script_path),
        ],
        cwd=str(PROJECT_ROOT),
        check=True,
    )


@dag(
    dag_id="olist_delivery_data_pipeline",

    description=(
        "Build and validate the Olist "
        "late-delivery modeling dataset."
    ),

    schedule=None,

    start_date=pendulum.datetime(
        2026,
        1,
        1,
        tz="UTC",
    ),

    catchup=False,

    tags=[
        "olist",
        "data-engineering",
        "machine-learning",
    ],
)
def olist_delivery_pipeline():

    @task
    def load_postgres():
        run_script(
            "src/load_to_postgres.py"
        )

    @task
    def create_sql_feature_view():
        run_script(
            "src/run_feature_view.py"
        )

    @task
    def build_target():
        run_script(
            "src/build_target.py"
        )

    @task
    def build_features():
        run_script(
            "src/build_features.py"
        )

    @task
    def add_geolocation():
        run_script(
            "src/add_geolocation.py"
        )

    @task
    def validate_features():
        run_script(
            "src/validate_features.py"
        )

    postgres = load_postgres()

    sql_view = create_sql_feature_view()

    target = build_target()

    features = build_features()

    geo = add_geolocation()

    validation = validate_features()

    postgres >> sql_view

    postgres >> target

    target >> features

    features >> geo

    geo >> validation


olist_delivery_pipeline()