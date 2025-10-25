from kedro.pipeline import Pipeline, node
from .nodes import train_linear_regression, train_xgb_regressor, train_ridge_regressor, train_rf_regressor, train_lasso_regressor

def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=train_linear_regression,
            inputs=["Features_training_v2", "params:linear_model_path"],
            outputs="linear_results",
            name="train_linear_regression_node"
        ),
        node(
            func=train_xgb_regressor,
            inputs=["Features_training_v2", "params:xgb_regressor_model_path"],
            outputs="xgb_regressor_results",
            name="train_xgb_regressor_node"
        ),
        node(
            func=train_ridge_regressor,
            inputs=["Features_training_v2", "params:ridge_model_path"],
            outputs="ridge_results",
            name="train_ridge_regressor_node"
        ),
        node(
            func=train_rf_regressor,
            inputs=["Features_training_v2", "params:rf_regressor_model_path"],
            outputs="rf_regressor_results",
            name="train_rf_regressor_node"
        ),
          node(
            func=train_lasso_regressor,
            inputs=["Features_training_v2", "params:lasso_model_path"],
            outputs="lasso_results",
            name="train_lasso_regressor_node"
        )
    ])