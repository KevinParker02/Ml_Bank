"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 1.0.0
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_and_validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Valida y retorna el dataset cargado.

    Args:
        df: DataFrame raw

    Returns:
        DataFrame validado
    """
    assert not df.empty, "Dataset vacío"
    logger.info(f"Dataset cargado: {df.shape}")
    return df

