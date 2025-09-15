"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 1.0.0
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import *

def create_pipeline(**kwargs):
    return Pipeline([
        # 1) Limpieza inicial -> NIV
        node(
            func=clean_datasets,
            inputs=["cleaned_dataset","customer_agg","RFM"],
            outputs=["NIV_cleaned_dataset","NIV_customer_agg","NIV_RFM"],
            name="clean_datasets_node"
        ),

        # 2) Transformación -> PRI
        node(
            func=transform_datasets,
            inputs=["NIV_cleaned_dataset","NIV_customer_agg","NIV_RFM"],
            outputs=["PRI_cleaned","PRI_customer","PRI_rfm"],
            name="transform_datasets_node"
        ),

        # 3) Integración -> PRI_full
        node(
            func=integrate_datasets,
            inputs=["PRI_cleaned","PRI_customer","PRI_rfm"],
            outputs="PRI_full",
            name="integrate_datasets_node"
        )
    ])