from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    train_knn_classifier,
    train_decision_tree,
    train_random_forest,
    train_xgboost_classifier,
    train_mlp_classifier,
)

def create_pipeline(**kwargs) -> Pipeline:
    """Pipeline de entrenamiento de modelos de clasificación (detección de fraude)."""
    return pipeline([
        node(
            func=train_knn_classifier,
            inputs=["Features_training_v1", "params:knn_model_path"],
            outputs="knn_results",
            name="train_knn_classifier_node",
        ),
        node(
            func=train_decision_tree,
            inputs=["Features_training_v1", "params:dt_model_path"],
            outputs="dt_results",
            name="train_decision_tree_node",
        ),
        node(
            func=train_random_forest,
            inputs=["Features_training_v1", "params:rf_model_path"],
            outputs="rf_results",
            name="train_random_forest_node",
        ),
        node(
            func=train_xgboost_classifier,
            inputs=["Features_training_v1", "params:xgb_model_path"],
            outputs="xgb_results",
            name="train_xgboost_classifier_node",
        ),
        node(
            func=train_mlp_classifier,
            inputs=["Features_training_v1", "params:mlp_model_path"],
            outputs="mlp_results",
            name="train_mlp_classifier_node",
        ),
    ])