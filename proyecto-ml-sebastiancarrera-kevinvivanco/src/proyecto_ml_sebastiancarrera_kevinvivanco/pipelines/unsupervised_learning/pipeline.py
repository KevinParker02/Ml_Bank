from kedro.pipeline import Pipeline, node, pipeline
from .nodes import run_unsupervised_learning

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=run_unsupervised_learning,
            inputs=["Features_training_v1", "params:unsupervised_learning"],
            outputs="Features_clustering_v1", 
            name="run_unsupervised_learning_full"
        )
    ])
