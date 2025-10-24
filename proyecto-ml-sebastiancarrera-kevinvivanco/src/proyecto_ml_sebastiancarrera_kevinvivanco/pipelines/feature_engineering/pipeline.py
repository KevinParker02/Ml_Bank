from kedro.pipeline import Pipeline, node, pipeline
from .nodes import compare_datasets, create_is_fraud, generate_features, clean_and_save

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=compare_datasets,
            inputs=["fraud_dataset", "PRI_FULL_V2"],   # PRI_FULL_V2 es la entrada
            outputs="pri_full_compared",
            name="compare_datasets_node",
        ),
        node(
            func=create_is_fraud,
            inputs=["pri_full_compared", "fraud_dataset"],
            outputs="pri_full_with_fraud",
            name="create_is_fraud_node",
        ),
        node(
            func=generate_features,
            inputs="pri_full_with_fraud",
            outputs="pri_full_engineered",
            name="generate_features_node",
        ),
        node(
            func=clean_and_save,
            inputs="pri_full_engineered",
            outputs="PRI_FULL_BASE",   # hora el dataset PRI_FULL_V2 sale como BASE
            name="clean_and_save_node",
        ),
    ])