import os
import pandas as pd

# =========================================================
# 🗂️ 0️⃣ Verificación y creación de carpetas (excepto 01_raw)
# =========================================================
import os

def ensure_data_folders(base_path="data"):
    folders = {
        "02_intermediate": [],
        "03_primary": [],
        "04_feature": [],
        "05_model_input": [],
        "06_models": ["clasificacion", "regresion"],
        "07_model_output": ["clasificacion", "regresion"],
        "08_reporting": ["clasificacion", "regresion"],
    }

    for folder, subfolders in folders.items():
        folder_path = os.path.join(base_path, folder)
        os.makedirs(folder_path, exist_ok=True)
        for sub in subfolders:
            os.makedirs(os.path.join(folder_path, sub), exist_ok=True)

ensure_data_folders()


# =========================================================
# 1️⃣ Selección de variables para CLASIFICACIÓN (is_fraud)
# =========================================================
def select_classification_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea el dataset de features para CLASIFICACIÓN."""
    keep_cols = [
        'AmountZScoreByLocation',
        'IsAnomaly',
        'IsLateNight',
        'IsWeekend',
        'is_fraud'
    ]

    # Filtrar solo columnas existentes
    keep_cols_existing = [c for c in keep_cols if c in df.columns]
    df_reduced = df[keep_cols_existing].copy()

    print(f"✅ Dataset de clasificación creado ({df_reduced.shape[0]} filas, {df_reduced.shape[1]} columnas)")
    print("Columnas finales:", list(df_reduced.columns))
    return df_reduced


# =========================================================
# 2️⃣ Selección de variables para REGRESIÓN (RiskScore)
# =========================================================
def select_regression_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea el dataset de features para REGRESIÓN."""
    keep_cols = [              
        "Monetary",       
        "TimeSinceLastTxn",
        "IsLateNight",            
        "AmountZScoreByLocation",
        "IsWeekend"     
    ]

    keep_cols_existing = [c for c in keep_cols if c in df.columns]
    df_reduced = df[keep_cols_existing].copy()

    print(f"✅ Dataset de regresión creado ({df_reduced.shape[0]} filas, {df_reduced.shape[1]} columnas)")
    print("Columnas finales:", list(df_reduced.columns))
    return df_reduced