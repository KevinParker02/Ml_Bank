import pandas as pd

# =========================================================
# 1️⃣ Selección de variables para CLASIFICACIÓN (is_fraud)
# =========================================================
def select_classification_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea el dataset de features para CLASIFICACIÓN."""
    keep_cols = [
        'AmountZScoreByLocation',
        'RiskScore',
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
        "TxnCountInLast24Hours",          
        "Recency",         
        "Monetary",       
        "TimeSinceLastTxn",            
        "AmountZScoreByLocation",  
        "TransactionAmount (INR)"         
    ]

    keep_cols_existing = [c for c in keep_cols if c in df.columns]
    df_reduced = df[keep_cols_existing].copy()

    print(f"✅ Dataset de regresión creado ({df_reduced.shape[0]} filas, {df_reduced.shape[1]} columnas)")
    print("Columnas finales:", list(df_reduced.columns))
    return df_reduced