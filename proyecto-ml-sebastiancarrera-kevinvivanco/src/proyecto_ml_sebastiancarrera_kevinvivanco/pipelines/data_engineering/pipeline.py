"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 1.0.0
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import *

def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=clean_datasets,
            inputs=["cleaned_dataset","customer_agg","RFM"],
            outputs=["NIV_cleaned_dataset","NIV_customer_agg","NIV_RFM"],
            name="clean_datasets_node"
        )
    ])

