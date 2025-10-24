import pandas as pd
import numpy as np

# =========================================================
# 1️⃣ Limpieza inicial
# =========================================================
def clean_datasets(cleaned_dataset, customer_agg, RFM):
    if "CustomerDOB" in cleaned_dataset.columns:
        cleaned_dataset = cleaned_dataset.drop(columns=["CustomerDOB","CustGender","CustAccountBalance","Age"])

    if "Unnamed: 0" in customer_agg.columns:
        customer_agg = customer_agg.drop(columns=["Unnamed: 0","gender","location","avg_spent_pct_balance"])

    drop_cols = [c for c in ["Unnamed: 0", "Segment","R","M","F"] if c in RFM.columns]
    RFM = RFM.drop(columns=drop_cols)

    return cleaned_dataset, customer_agg, RFM


# =========================================================
# 2️⃣ Transformación
# =========================================================
def transform_datasets(cleaned_dataset: pd.DataFrame,
                       customer_agg: pd.DataFrame,
                       rfm: pd.DataFrame):
    cleaned_dataset = cleaned_dataset.dropna()

    if "TransactionDate" in cleaned_dataset.columns:
        cleaned_dataset["TransactionDate"] = pd.to_datetime(cleaned_dataset["TransactionDate"], errors="coerce")

    if "TransactionTime" in cleaned_dataset.columns:
        cleaned_dataset["TransactionTime"] = pd.to_datetime(
            cleaned_dataset["TransactionTime"], format="%H:%M:%S", errors="coerce"
        ).dt.time

    if "CustLocation" in cleaned_dataset.columns:
        cleaned_dataset["CustLocation"] = cleaned_dataset["CustLocation"].astype("category")

    for col in ["TransactionID", "CustomerID"]:
        if col in cleaned_dataset.columns:
            cleaned_dataset[col] = cleaned_dataset[col].astype("string")

    customer_agg = customer_agg.dropna()

    if "CustomerID" in customer_agg.columns:
        customer_agg["CustomerID"] = customer_agg["CustomerID"].astype("string")

    for col in ["first_txn_date", "last_txn_date"]:
        if col in customer_agg.columns:
            customer_agg[col] = pd.to_datetime(customer_agg[col], errors="coerce")

    if "CustomerID" in rfm.columns:
        rfm["CustomerID"] = rfm["CustomerID"].astype("string")

    if "Segment_Final" in rfm.columns:
        rfm["Segment_Final"] = rfm["Segment_Final"].astype("category")

    return cleaned_dataset, customer_agg, rfm


# =========================================================
# 3️⃣ Tratamiento de Outliers (nuevo)
# =========================================================
def treat_outliers(cleaned: pd.DataFrame,
                   agg: pd.DataFrame,
                   rfm: pd.DataFrame):
    """
    Aplica capping IQR en columnas numéricas de los tres datasets.
    """

    def cap_iqr(df, col, k=1.5):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - k * IQR
        upper = Q3 + k * IQR
        df[col] = np.clip(df[col], lower, upper)
        return df

    # --- RFM ---
    if all(col in rfm.columns for col in ["Recency","Frequency","Monetary"]):
        rfm = cap_iqr(rfm, "Recency", k=2.0)
        rfm = cap_iqr(rfm, "Frequency", k=1.5)
        rfm = cap_iqr(rfm, "Monetary", k=1.0)

    # --- cleaned ---
    if "TransactionAmount (INR)" in cleaned.columns:
        cleaned = cap_iqr(cleaned, "TransactionAmount (INR)", k=1.5)

    # --- agg ---
    for col in ["txn_count", "total_spent", "avg_spent", "max_spent", "avg_balance", "recency_days"]:
        if col in agg.columns:
            agg = cap_iqr(agg, col, k=1.5)

    return cleaned, agg, rfm


# =========================================================
# 4️⃣ Integración final
# =========================================================
def integrate_datasets_v2(cleaned: pd.DataFrame,
                          customer: pd.DataFrame,
                          rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Une los tres datasets tratados en un único dataset maestro PRI_FULL_V2.
    """
    df_full_v2 = (
        cleaned
        .merge(customer, on="CustomerID", how="left")
        .merge(rfm, on="CustomerID", how="left")
    )
    return df_full_v2