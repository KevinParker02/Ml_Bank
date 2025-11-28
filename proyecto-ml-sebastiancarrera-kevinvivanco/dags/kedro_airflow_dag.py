from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "kevinvivanco",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 0,
}

with DAG(
    dag_id="kedro_pipeline_ml_bank",
    default_args=default_args,
    description="Orquestación de pipelines Kedro para ML_Bank (clasificación, regresión y clustering)",
    schedule_interval=None,  # ejecución manual
    start_date=datetime(2025, 10, 1),
    catchup=False,
    tags=["kedro", "ml_bank", "airflow"],
) as dag:

    data_engineering = BashOperator(
        task_id="data_engineering",
        bash_command="cd /app && kedro run --pipeline=data_engineering",
    )

    feature_engineering = BashOperator(
        task_id="feature_engineering",
        bash_command="cd /app && kedro run --pipeline=feature_engineering",
    )

    feature_selection = BashOperator(
        task_id="feature_selection",
        bash_command="cd /app && kedro run --pipeline=feature_selection",
    )

    classification_models = BashOperator(
        task_id="classification_models",
        bash_command="cd /app && kedro run --pipeline=classification_models",
    )

    regression_models = BashOperator(
        task_id="regression_models",
        bash_command="cd /app && kedro run --pipeline=regression_models",
    )

    unsupervised_learning = BashOperator(
        task_id="unsupervised_learning",
        bash_command="cd /app && kedro run --pipeline=unsupervised_learning",
    )

    reporting = BashOperator(
        task_id="reporting",
        bash_command="cd /app && kedro run --pipeline=reporting",
    )


    # Dependencias
    data_engineering >> feature_engineering >> feature_selection
    feature_selection >> [classification_models, regression_models] >> unsupervised_learning >> reporting
