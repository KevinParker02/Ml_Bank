"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 1.0.0
"""

import pandas as pd

def clean_datasets(cleaned_dataset, customer_agg, RFM):
    # --- cleaned_dataset ---
    if "CustomerDOB" in cleaned_dataset.columns:
        cleaned_dataset = cleaned_dataset.drop(columns=["CustomerDOB","CustGender","CustAccountBalance","Age"])

    #--- customer_agg ---,
    if "Unnamed: 0" in customer_agg.columns:
        customer_agg = customer_agg.drop(columns=["Unnamed: 0","gender","location","avg_spent_pct_balance"])

    #--- RFM ---,
    drop_cols = [c for c in ["Unnamed: 0", "Segment","R","M","F"] if c in RFM.columns]
    RFM = RFM.drop(columns=drop_cols)

    return cleaned_dataset, customer_agg, RFM

def transform_datasets(cleaned_dataset: pd.DataFrame,
                       customer_agg: pd.DataFrame,
                       rfm: pd.DataFrame):
    """
    Aplica transformaciones de tipos y limpieza ligera
    sobre los tres datasets.
    """
    # =========================
    # --- cleaned_dataset ---
    # =========================
    cleaned_dataset = cleaned_dataset.dropna()

    if "TransactionDate" in cleaned_dataset.columns:
        cleaned_dataset["TransactionDate"] = pd.to_datetime(
            cleaned_dataset["TransactionDate"], errors="coerce"
        )

    if "TransactionTime" in cleaned_dataset.columns:
        cleaned_dataset["TransactionTime"] = pd.to_datetime(
            cleaned_dataset["TransactionTime"], format="%H:%M:%S", errors="coerce"
        ).dt.time

    if "CustLocation" in cleaned_dataset.columns:
        cleaned_dataset["CustLocation"] = cleaned_dataset["CustLocation"].astype("category")

    if "TransactionID" in cleaned_dataset.columns:
        cleaned_dataset["TransactionID"] = cleaned_dataset["TransactionID"].astype("string")
    if "CustomerID" in cleaned_dataset.columns:
        cleaned_dataset["CustomerID"] = cleaned_dataset["CustomerID"].astype("string")

    # =========================
    # --- customer_agg ---
    # =========================
    customer_agg = customer_agg.dropna()

    if "CustomerID" in customer_agg.columns:
        customer_agg["CustomerID"] = customer_agg["CustomerID"].astype("string")

    if "first_txn_date" in customer_agg.columns:
        customer_agg["first_txn_date"] = pd.to_datetime(customer_agg["first_txn_date"], errors="coerce")

    if "last_txn_date" in customer_agg.columns:
        customer_agg["last_txn_date"] = pd.to_datetime(customer_agg["last_txn_date"], errors="coerce")

    # =========================
    # --- rfm ---
    # =========================
    if "CustomerID" in rfm.columns:
        rfm["CustomerID"] = rfm["CustomerID"].astype("string")

    if "Segment_Final" in rfm.columns:
        rfm["Segment_Final"] = rfm["Segment_Final"].astype("category")

    return cleaned_dataset, customer_agg, rfm


def integrate_datasets(cleaned_dataset: pd.DataFrame,
                       customer_agg: pd.DataFrame,
                       rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Une los 3 datasets por CustomerID y devuelve un dataset maestro.
    """
    df_full = (
        cleaned_dataset
        .merge(customer_agg, on="CustomerID", how="left")
        .merge(rfm, on="CustomerID", how="left")
    )
    return df_full
