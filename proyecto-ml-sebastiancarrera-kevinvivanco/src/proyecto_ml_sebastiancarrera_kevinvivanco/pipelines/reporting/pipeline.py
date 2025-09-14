"""
This is a boilerplate pipeline 'reporting'
generated using Kedro 1.0.0
"""

from kedro.pipeline import node, Pipeline  # noqa
from .nodes import generate_reporting

def create_pipeline(**kwargs):
    return Pipeline([
        # Reporte inicial (datasets crudos)
        node(
            func=generate_reporting,
            inputs=dict(df="cleaned_dataset", dataset_name="params:dataset_name_cleaned", stage="params:stage_inicial"),
            outputs="report_cleaned_inicial",
            name="report_cleaned_inicial_node",
        ),
        node(
            func=generate_reporting,
            inputs=dict(df="customer_agg", dataset_name="params:dataset_name_customer", stage="params:stage_inicial"),
            outputs="report_customer_inicial",
            name="report_customer_inicial_node",
        ),
        node(
            func=generate_reporting,
            inputs=dict(df="RFM", dataset_name="params:dataset_name_rfm", stage="params:stage_inicial"),
            outputs="report_rfm_inicial",
            name="report_rfm_inicial_node",
        ),

        # Aquí va tu pipeline de data_engineering normal

        # Reporte final (datasets ya procesados)
        node(
            func=generate_reporting,
            inputs=dict(df="NIV_cleaned_dataset", dataset_name="params:dataset_name_cleaned", stage="params:stage_final"),
            outputs="report_cleaned_VarEliminadas",
            name="report_cleaned_VarEliminadas",
        ),
        node(
            func=generate_reporting,
            inputs=dict(df="NIV_customer_agg", dataset_name="params:dataset_name_customer", stage="params:stage_final"),
            outputs="report_customer_VarEliminadas",
            name="report_customer_VarEliminadas",
        ),
        node(
            func=generate_reporting,
            inputs=dict(df="NIV_RFM", dataset_name="params:dataset_name_rfm", stage="params:stage_final"),
            outputs="report_rfm_VarEliminadas",
            name="report_rfm_VarEliminadas",
        )
    ])