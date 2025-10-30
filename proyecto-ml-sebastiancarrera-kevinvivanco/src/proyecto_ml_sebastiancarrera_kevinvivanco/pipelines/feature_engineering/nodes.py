import os
import pandas as pd
import numpy as np

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
# 1️⃣ Cargar y comparar datasets
# =========================================================
def compare_datasets(fraud_df: pd.DataFrame, pri_full_df: pd.DataFrame) -> pd.DataFrame:
    print("Dataset FRAUD:", fraud_df.shape)
    print("Dataset PRI_FULL_V2:", pri_full_df.shape)

    common_cols = list(set(fraud_df.columns).intersection(pri_full_df.columns))
    print("✅ Columnas comunes:", common_cols)
    return pri_full_df


# =========================================================
# 2️⃣ Crear columna is_fraud
# =========================================================
def create_is_fraud(pri_full_df: pd.DataFrame, fraud_df: pd.DataFrame) -> pd.DataFrame:
    if "TransactionID" in pri_full_df.columns and "TransactionID" in fraud_df.columns:
        pri_full_df["is_fraud"] = pri_full_df["TransactionID"].isin(
            fraud_df["TransactionID"]
        ).astype(int)
        print("📊 Etiquetas de fraude creadas:")
        print(pri_full_df["is_fraud"].value_counts())
    else:
        print("⚠️ No existe TransactionID en alguno de los datasets.")
        pri_full_df["is_fraud"] = 0
    return pri_full_df


# =========================================================
# 3️⃣ Feature engineering (temporales, comportamiento, riesgo)
# =========================================================
def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], errors="coerce")
    df["TransactionTime"] = pd.to_datetime(df["TransactionTime"], format="%H:%M:%S", errors="coerce")

    # Variables temporales
    df["DayOfWeek"] = df["TransactionDate"].dt.day_name()
    df["IsWeekend"] = df["DayOfWeek"].isin(["Saturday", "Sunday"]).astype(int)
    df["hour"] = df["TransactionTime"].dt.hour
    df["IsLateNight"] = ((df["hour"] >= 23) | (df["hour"] <= 5)).astype(int)
    df["TimeOfDay"] = pd.cut(
        df["hour"],
        bins=[0, 6, 12, 18, 24],
        labels=["Night", "Morning", "Afternoon", "Evening"],
        right=False,
        include_lowest=True,
    )

    # Variables de comportamiento
    df["AmountZScoreByLocation"] = (
        df.groupby("CustLocation")["TransactionAmount (INR)"]
        .transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
    )

    df = df.sort_values(["CustomerID", "TransactionDate"])
    df["TimeSinceLastTxn"] = (
        df.groupby("CustomerID")["TransactionDate"].diff().dt.total_seconds() / (60 * 60 * 24)
    ).fillna(0)

    df["TxnCountInLast24Hours"] = (
        (df.groupby("CustomerID")["TransactionDate"].diff().dt.total_seconds() / 3600 <= 24)
        .astype(int)
        .fillna(0)
    )

    # Variables de riesgo
    df["RiskScore"] = np.where(
        df["TransactionAmount (INR)"]
        > (df["TransactionAmount (INR)"].mean() + 2 * df["TransactionAmount (INR)"].std()),
        1,
        0,
    )
    df["IsAnomaly"] = ((df["RiskScore"] == 1) & (df["IsLateNight"] == 1)).astype(int)

    return df


# =========================================================
# 4️⃣ Limpieza final y validación
# =========================================================
def clean_and_save(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna({
        "AmountZScoreByLocation": 0,
        "TimeSinceLastTxn": 0,
        "TxnCountInLast24Hours": 0,
        "RiskScore": 0,
        "IsAnomaly": 0,
        "is_fraud": 0,
    })

    print("✅ Limpieza final aplicada.")
    print("Shape final:", df.shape)
    print("Nuevas columnas añadidas:", [
        "DayOfWeek", "IsWeekend", "IsLateNight", "TimeOfDay",
        "AmountZScoreByLocation", "TimeSinceLastTxn", "TxnCountInLast24Hours",
        "RiskScore", "IsAnomaly", "is_fraud"
    ])
    return df