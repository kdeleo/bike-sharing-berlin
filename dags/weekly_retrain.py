"""
Weekly retrain DAG — 02:00 Europe/Berlin every Sunday.

Rebuilds the feature matrix from all accumulated data and retrains the
LightGBM model with 50 Optuna trials. Replaces models/best_model.txt.

One week adds ~63 district-rows — enough to capture the latest seasonal
patterns without the cost of daily retraining.

Task order:
    build_features → train_model
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
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="weekly_retrain",
    schedule="0 2 * * 0",
    start_date=pendulum.datetime(2026, 6, 1, tz="Europe/Berlin"),
    catchup=False,
    default_args=default_args,
    tags=["bike-sharing"],
    doc_md=__doc__,
) as dag:

    build_features = BashOperator(
        task_id="build_features",
        bash_command=f"cd {PROJECT_ROOT} && {PYTHON} -m src.features.build_features",
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command=f"cd {PROJECT_ROOT} && {PYTHON} -m src.models.train --trials 50",
    )

    build_features >> train_model
