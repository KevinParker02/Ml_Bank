"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 1.0.0
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import load_and_validate_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_and_validate_data,
                inputs="cleaned_dataset",
                outputs="validated_cleaned_dataset",  # ⚠️ distinto nombre
                name="validate_cleaned_dataset_node",
            ),
        ]
    )

