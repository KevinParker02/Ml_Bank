from kedro.pipeline import Pipeline, node, pipeline
from .nodes import generate_model_dashboard

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=generate_model_dashboard,
            inputs="params:reporting_config",
            outputs=None,
            name="generate_visual_dashboard"
        )
    ])