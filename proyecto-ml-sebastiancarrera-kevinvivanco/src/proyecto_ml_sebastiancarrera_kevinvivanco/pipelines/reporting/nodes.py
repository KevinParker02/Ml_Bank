"""
This is a boilerplate pipeline 'reporting'
generated using Kedro 1.0.0
"""
import pandas as pd
def generate_reporting(df: pd.DataFrame, dataset_name: str, stage: str) -> pd.DataFrame:
    report = pd.DataFrame([{
        "etapa": stage,
        "dataset": dataset_name,
        "filas": df.shape[0],
        "columnas": df.shape[1],
        "duplicados": df.duplicated().sum(),
        "nulos": df.isnull().sum().sum(),
        "nulos (%)": round((df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100, 4)
    }])
    return report