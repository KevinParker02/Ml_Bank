"""
Feature engineering nodes: RFM, customer value, fraud features, temporal encoding.
Ajustado para recibir params = bloque `feature_engineering` (params:feature_engineering).
"""

from typing import Dict
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ----------------------------- helpers -----------------------------
def _colmap(params: Dict, section: str) -> Dict[str, str]:
    """
    Lee los mapeos de columnas desde parameters.yml para la sección dada.
    Espera algo como:
      params[section]["columns"] = {
          customer_id: ..., transaction_id: ..., transaction_date: ..., amount: ...
      }
    """
    cols = params.get(section, {}).get("columns", {})
    if not cols:
        raise ValueError(f"No se encontraron columnas en params para '{section}.columns'")

    return {
        "cid": cols["customer_id"],
        "tid": cols.get("transaction_id", None),
        "tdate": cols["transaction_date"],
        "amount": cols["amount"],
    }

# ------------------------------- RFM --------------------------------
def create_rfm_features(transactions_df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """
    Calcula R,F,M + puntajes y score combinado por cliente.
    `params` es el bloque YA recortado (viene de params:feature_engineering).
    """
    logger.info("Iniciando creación de features RFM")

    cm = _colmap(params, "rfm")

    # Validaciones mínimas
    for c in [cm["cid"], cm["tdate"], cm["amount"]]:
        if c not in transactions_df.columns:
            raise ValueError(f"Columna requerida no encontrada: {c}")

    df = transactions_df[[cm["cid"], cm["tdate"], cm["amount"]]].copy()
    df[cm["tdate"]] = pd.to_datetime(df[cm["tdate"]], errors="coerce")
    if df[cm["tdate"]].isna().all():
        raise ValueError("Todas las fechas son NaT; revisa parsing de fechas")

    # reference_date
    reference_date = params["rfm"].get("reference_date")
    if reference_date is None:
        reference_date = df[cm["tdate"]].max() + pd.Timedelta(days=1)
    else:
        reference_date = pd.to_datetime(reference_date)

    # groupby customer
    rfm = (
        df.groupby(cm["cid"])
          .agg(
              recency_days=(cm["tdate"], lambda x: (reference_date - x.max()).days),
              frequency=(cm["tdate"], "count"),
              monetary_value=(cm["amount"], "sum"),
          )
          .reset_index()
          .rename(columns={cm["cid"]: "CustomerID"})
    )

    # checks
    assert (rfm["recency_days"] >= 0).all(), "Recency negativa detectada"
    assert (rfm["frequency"] > 0).all(), "Frequency debe ser > 0"
    assert (rfm["monetary_value"] >= 0).all(), "Monetary negativo detectado"

    # ======= Scoring robusto a empates (evita error de qcut) =======
    def _qscore(series: pd.Series, bins: int, high_is_good: bool) -> pd.Series:
        ranks = series.rank(method="first", pct=True)             # maneja empates
        q = pd.qcut(ranks, q=bins, labels=False, duplicates="drop")  # 0..k-1
        k = int(q.max() + 1)                                      # k compartimentos reales
        if high_is_good:
            return (q + 1).astype(int)                            # 1..k
        else:
            return (k - q).astype(int)                            # k..1 (menor mejor)

    bins = int(params["rfm"].get("rfm_bins", 5))

    # Recency: menor días = mejor (puntaje alto)
    rfm["R_score"] = _qscore(rfm["recency_days"], bins=bins, high_is_good=False)
    # Frequency y Monetary: mayor = mejor
    rfm["F_score"] = _qscore(rfm["frequency"], bins=bins, high_is_good=True)
    rfm["M_score"] = _qscore(rfm["monetary_value"], bins=bins, high_is_good=True)

    rfm["RFM_score"] = (
        rfm["R_score"].astype(str)
        + rfm["F_score"].astype(str)
        + rfm["M_score"].astype(str)
    )
    rfm["RFM_total_score"] = rfm["R_score"] + rfm["F_score"] + rfm["M_score"]

    logger.info(
        "RFM creado para %s clientes | recency mean=%.1f días",
        len(rfm),
        rfm["recency_days"].mean(),
    )
    return rfm


def create_rfm_segments(rfm_df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """Asigna segmento de marketing según scores R,F,M."""
    rfm = rfm_df.copy()

    def segment(row):
        r, f, m = row["R_score"], row["F_score"], row["M_score"]
        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        if f >= 4 and (r >= 3 or m >= 3):
            return "Loyal Customers"
        if r >= 3 and f >= 3 and m >= 3:
            return "Potential Loyalists"
        if m >= 4 and f < 4:
            return "Big Spenders"
        if r <= 2 and f >= 3 and m >= 3:
            return "At Risk"
        if r <= 2 and f <= 2:
            return "Hibernating"
        if r == 1 and f <= 2 and m <= 2:
            return "Lost"
        if r >= 4 and f == 1:
            return "New Customers"
        if r >= 3 and f <= 2 and m <= 2:
            return "Promising"
        return "Other"

    rfm["segment"] = rfm.apply(segment, axis=1)
    logger.info("Distribución segmentos RFM:\n%s", rfm["segment"].value_counts().to_string())
    return rfm

# -------------------------- Customer value --------------------------
def create_customer_value_features(
    transactions_df: pd.DataFrame, rfm_df: pd.DataFrame, params: Dict
) -> pd.DataFrame:
    """Agrega métricas por cliente (promedios, std, lifetime, CLV simple) y une con RFM."""
    logger.info("Creando features de valor de cliente")

    cm = _colmap(params, "rfm")  # reusar mapeo de transacciones
    df = transactions_df[[cm["cid"], cm["tdate"], cm["amount"]]].copy()
    df[cm["tdate"]] = pd.to_datetime(df[cm["tdate"]], errors="coerce")

    agg = (
        df.groupby(cm["cid"])
          .agg(
              avg_transaction_amount=(cm["amount"], "mean"),
              median_transaction_amount=(cm["amount"], "median"),
              std_transaction_amount=(cm["amount"], "std"),
              min_transaction_amount=(cm["amount"], "min"),
              max_transaction_amount=(cm["amount"], "max"),
              total_transactions=(cm["tdate"], "count"),
              first_transaction_date=(cm["tdate"], "min"),
              last_transaction_date=(cm["tdate"], "max"),
          )
          .reset_index()
          .rename(columns={cm["cid"]: "CustomerID"})
    )

    agg["transaction_amount_range"] = agg["max_transaction_amount"] - agg["min_transaction_amount"]
    agg["transaction_cv"] = agg["std_transaction_amount"] / (agg["avg_transaction_amount"] + 1e-9)
    agg["customer_lifetime_days"] = (
        agg["last_transaction_date"] - agg["first_transaction_date"]
    ).dt.days.fillna(0)
    agg["transactions_per_day"] = agg["total_transactions"] / (agg["customer_lifetime_days"] + 1)

    out = rfm_df.merge(agg, on="CustomerID", how="left")
    annual_factor = params["customer_value"].get("annualization_factor", 12)
    out["estimated_clv"] = (
        out["avg_transaction_amount"].fillna(0)
        * out["frequency"].fillna(0)
        * annual_factor
        / 12.0
    )

    logger.info("Features de valor creadas: %s filas", len(out))
    return out


# --------------------------- Fraud features -------------------------
def create_fraud_detection_features(transactions_df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """
    Crea features transaccionales para detección de fraude:
    - hour, day_of_week, is_weekend, is_night
    - time_since_last_transaction (horas)
    - amount_zscore por cliente
    - flags: is_rapid_transaction, is_anomalous_amount, is_high_value
    """
    logger.info("Creando features de fraude")

    fd = params["fraud_detection"]
    cm = _colmap(params, "fraud_detection")

    df = transactions_df.copy()
    for c in [cm["cid"], cm["tdate"], cm["amount"]]:
        if c not in df.columns:
            raise ValueError(f"Columna requerida no encontrada: {c}")

    df[cm["tdate"]] = pd.to_datetime(df[cm["tdate"]], errors="coerce")
    df = df.sort_values([cm["cid"], cm["tdate"]]).reset_index(drop=True)

    # temporales
    df["hour"] = df[cm["tdate"]].dt.hour
    df["day_of_week"] = df[cm["tdate"]].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 6)).astype(int)

    # delta entre transacciones del mismo cliente (horas)
    df["time_since_last_transaction"] = (
        df.groupby(cm["cid"])[cm["tdate"]].diff().dt.total_seconds() / 3600.0
    )

    rapid_thr = float(fd.get("rapid_transaction_threshold_hours", 0.5))
    df["is_rapid_transaction"] = (df["time_since_last_transaction"] < rapid_thr).fillna(False).astype(int)

    # z-score por cliente
    stats = (
        df.groupby(cm["cid"])[cm["amount"]]
        .agg(["mean", "std"])
        .rename(columns={"mean": "mu", "std": "sigma"})
    )
    df = df.merge(stats, left_on=cm["cid"], right_index=True, how="left")
    df["amount_zscore"] = (df[cm["amount"]] - df["mu"]) / (df["sigma"].replace(0, np.nan))
    df["amount_zscore"] = df["amount_zscore"].fillna(0)

    zthr = float(fd.get("anomaly_zscore_threshold", 3))
    df["is_anomalous_amount"] = (np.abs(df["amount_zscore"]) > zthr).astype(int)

    hv = fd.get("high_value_threshold", None)
    if hv is None:
        hv = df[cm["amount"]].quantile(0.95)
    df["is_high_value"] = (df[cm["amount"]] > hv).astype(int)

    logger.info(
        "Fraud features listas | rapid=%d, anomalous=%d, high_value=%d",
        df["is_rapid_transaction"].sum(),
        df["is_anomalous_amount"].sum(),
        df["is_high_value"].sum(),
    )
    return df


# ---------------------- Temporal cyclic encoding --------------------
def encode_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Codifica hora y día de semana como componentes sinusoidales (cíclicos)."""
    out = df.copy()
    if "hour" in out.columns:
        out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
        out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    if "day_of_week" in out.columns:
        out["day_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7.0)
        out["day_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7.0)
    logger.info("Codificación temporal completada")
    return out
