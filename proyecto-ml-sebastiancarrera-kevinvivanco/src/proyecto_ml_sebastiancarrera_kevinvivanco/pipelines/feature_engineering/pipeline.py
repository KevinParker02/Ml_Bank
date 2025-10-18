from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    create_rfm_features,
    create_rfm_segments,
    create_customer_value_features,
    create_fraud_detection_features,
    encode_temporal_features,
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=create_rfm_features,
                inputs=["PRI_cleaned", "params:feature_engineering"],
                outputs="rfm_features",
                name="create_rfm_features",
            ),
            node(
                func=create_rfm_segments,
                inputs=["rfm_features", "params:feature_engineering"],
                outputs="rfm_with_segments",
                name="create_rfm_segments",
            ),
            node(
                func=create_customer_value_features,
                inputs=["PRI_cleaned", "rfm_with_segments", "params:feature_engineering"],
                outputs="customer_features_enriched",
                name="create_customer_value_features",
            ),
            node(
                func=create_fraud_detection_features,
                inputs=["PRI_cleaned", "params:feature_engineering"],
                outputs="transactions_with_fraud_features",
                name="create_fraud_detection_features",
            ),
            node(
                func=encode_temporal_features,
                inputs="transactions_with_fraud_features",
                outputs="transactions_model_ready",
                name="encode_temporal_features",
            ),
        ],
        tags=["features", "rfm", "fraud", "phase3"],
    )
