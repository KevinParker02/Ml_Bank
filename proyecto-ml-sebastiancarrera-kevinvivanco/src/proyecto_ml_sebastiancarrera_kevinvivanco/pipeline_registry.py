"""Project pipelines."""


from typing import Dict
from kedro.pipeline import Pipeline
from proyecto_ml_sebastiancarrera_kevinvivanco.pipelines import data_engineering

def register_pipelines() -> Dict[str, Pipeline]:
    data_engineering_pipeline = data_engineering.create_pipeline()

    return {
        "__default__": data_engineering_pipeline,  # corre si pones `kedro run`
        "data_engineering": data_engineering_pipeline,
    }
