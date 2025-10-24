from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_knn_classifier

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=train_knn_classifier,
            inputs=["Features_training_v1", "params:knn_model_path"],
            outputs="knn_results",
            name="train_knn_classifier_node",
        ),
    ])