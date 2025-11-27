"""Project pipelines."""


from typing import Dict
from kedro.pipeline import Pipeline
from proyecto_ml_sebastiancarrera_kevinvivanco.pipelines import data_engineering, reporting
from proyecto_ml_sebastiancarrera_kevinvivanco.pipelines import feature_engineering as fe
from proyecto_ml_sebastiancarrera_kevinvivanco.pipelines import feature_selection as fs
from proyecto_ml_sebastiancarrera_kevinvivanco.pipelines import classification_models as cm
from proyecto_ml_sebastiancarrera_kevinvivanco.pipelines import regression_models as rm
from proyecto_ml_sebastiancarrera_kevinvivanco.pipelines import unsupervised_learning as ul
def register_pipelines() -> Dict[str, Pipeline]:
    data_engineering_pipeline = data_engineering.create_pipeline()
    reporting_pipeline = reporting.create_pipeline()
    feature_engineering_pipeline = fe.create_pipeline()
    feature_selection_pipeline = fs.create_pipeline()
    classification_models_pipeline = cm.create_pipeline()
    regression_models_pipeline = rm.create_pipeline()
    unsupervised_learning_pipeline= ul.create_pipeline()

    return {
        "data_engineering": data_engineering_pipeline,
        "feature_engineering": feature_engineering_pipeline,
        "feature_selection": feature_selection_pipeline,
        "classification_models": classification_models_pipeline,
        "regression_models": regression_models_pipeline,
        "unsupervised-learning": unsupervised_learning_pipeline,
        "reporting": reporting_pipeline,
        "__default__": data_engineering_pipeline + feature_engineering_pipeline + feature_selection_pipeline + classification_models_pipeline + regression_models_pipeline + unsupervised_learning_pipeline + reporting_pipeline  # corre si pones `kedro run`
    }
