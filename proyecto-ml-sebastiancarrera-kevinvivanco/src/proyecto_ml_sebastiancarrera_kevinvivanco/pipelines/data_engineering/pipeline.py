"""
Pipeline 'data_engineering' actualizado con tratamiento de outliers
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import *

def create_pipeline(**kwargs):
    return Pipeline([
        # 1️⃣ Limpieza inicial
        node(
            func=clean_datasets,
            inputs=["cleaned_dataset","customer_agg","RFM"],
            outputs=["NIV_cleaned_dataset","NIV_customer_agg","NIV_RFM"],
            name="clean_datasets_node"
        ),

        # 2️⃣ Transformación
        node(
            func=transform_datasets,
            inputs=["NIV_cleaned_dataset","NIV_customer_agg","NIV_RFM"],
            outputs=["PRI_cleaned","PRI_customer","PRI_rfm"],
            name="transform_datasets_node"
        ),

        # 3️⃣ Tratamiento de outliers
        node(
            func=treat_outliers,
            inputs=["PRI_cleaned","PRI_customer","PRI_rfm"],
            outputs=["PRI_v2_cleaned","PRI_v2_customer","PRI_v2_rfm"],
            name="treat_outliers_node"
        ),

        # 4️⃣ Integración final
        node(
            func=integrate_datasets_v2,
            inputs=["PRI_v2_cleaned","PRI_v2_customer","PRI_v2_rfm"],
            outputs="PRI_FULL_V2",
            name="integrate_datasets_v2_node"
        ),
    ])