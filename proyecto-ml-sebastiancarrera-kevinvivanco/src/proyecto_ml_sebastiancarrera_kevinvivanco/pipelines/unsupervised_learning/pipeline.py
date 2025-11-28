from kedro.pipeline import Pipeline, node, pipeline
from .nodes import run_unsupervised_learning, run_best_xgb_model

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        # === Task 1: Clustering (Notebook 05) ===
        node(
            func=run_unsupervised_learning,
            inputs=["Features_training_v1", "params:unsupervised_learning"],
            outputs="Features_clustering_v1",
            name="run_unsupervised_learning_full"
        ),
        # === Task 2: Clasificación (Notebook 06) ===
        node(
            func=run_best_xgb_model,
            inputs=["Features_clustering_v1", "params:unsupervised_learning"],
            outputs=None,
            name="run_best_xgb_model_task2"
        )
    ])
