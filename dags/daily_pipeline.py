"""
Daily pipeline DAG — 07:00 Europe/Berlin.

Fetches today's live station snapshot and weather forecast, appends to the
demand parquet incrementally, rebuilds the feature matrix, and generates
tomorrow's district-level predictions.

Task order:
    fetch_live → run_etl → build_features → predict
"""

from datetime import timedelta
from pathlib import Path

import pendulum
from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
PYTHON = f"{PROJECT_ROOT}/.venv/bin/python"

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="daily_pipeline",
    schedule="0 7 * * *",
    start_date=pendulum.datetime(2026, 6, 1, tz="Europe/Berlin"),
    catchup=False,
    default_args=default_args,
    tags=["bike-sharing"],
    doc_md=__doc__,
) as dag:

    fetch_live = BashOperator(
        task_id="fetch_live",
        bash_command=f"cd {PROJECT_ROOT} && {PYTHON} -m src.data.collection.fetch_live",
    )

    run_etl = BashOperator(
        task_id="run_etl",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            f"{PYTHON} -m src.data.processing.pipeline --incremental --no-plots"
        ),
    )

    build_features = BashOperator(
        task_id="build_features",
        bash_command=f"cd {PROJECT_ROOT} && {PYTHON} -m src.features.build_features",
    )

    predict = BashOperator(
        task_id="predict",
        bash_command=f"cd {PROJECT_ROOT} && {PYTHON} -m src.models.predict",
    )

    fetch_live >> run_etl >> build_features >> predict
