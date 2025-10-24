from kedro.pipeline import Pipeline, node, pipeline
from .nodes import select_classification_features, select_regression_features

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=select_classification_features,
            inputs="PRI_FULL_BASE",
            outputs="Features_training_v1",
            name="select_classification_features_node",
        ),
        node(
            func=select_regression_features,
            inputs="PRI_FULL_BASE",
            outputs="Features_training_v2",
            name="select_regression_features_node",
        ),
    ])