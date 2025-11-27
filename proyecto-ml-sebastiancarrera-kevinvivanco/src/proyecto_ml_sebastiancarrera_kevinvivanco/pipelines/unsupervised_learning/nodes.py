# 📦 Manipulación de datos
import pandas as pd
import numpy as np
# 🎨 Visualización estática
import seaborn as sns
import matplotlib.pyplot as plt
# 🎨 Visualización interactiva
import plotly.express as px
import plotly.graph_objects as go
# 🔢 Preprocesamiento y escalado
from sklearn.preprocessing import StandardScaler
# 🤖 Clustering
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from typing import Dict
# 📊 Métricas de clustering
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
# 🔻 Reducción de dimensionalidad
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# 🌳 Jerárquico (para dendrogramas)
from scipy.cluster.hierarchy import linkage, dendrogram
# 📏 Distancias (para heatmaps de centroides)
from scipy.spatial.distance import cdist


#Nodo de dbscan
def pca_dbscan_clustering(
    df: pd.DataFrame,
    params: Dict,
) -> Dict:
    """
    PCA (2D) + DBSCAN clustering con métricas y visualización.
    """

    # --------------------------------------------------
    # 1) SAMPLE
    # --------------------------------------------------
    n_sample = params.get("n_sample", 50_000)
    df_sample = (
        df.sample(n=n_sample, random_state=42)
        if len(df) > n_sample
        else df.copy()
    )

    # --------------------------------------------------
    # 2) FEATURES
    # --------------------------------------------------
    features = params["features"]
    target = params["target"]

    X = df_sample[features]
    y = df_sample[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --------------------------------------------------
    # 3) PCA
    # --------------------------------------------------
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # --------------------------------------------------
    # 4) DBSCAN
    # --------------------------------------------------
    dbscan = DBSCAN(
        eps=params["dbscan"]["eps"],
        min_samples=params["dbscan"]["min_samples"],
    )

    labels = dbscan.fit_predict(X_pca)

    # --------------------------------------------------
    # 5) MÉTRICAS (sin ruido)
    # --------------------------------------------------
    mask = labels != -1
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))

    if n_clusters > 1 and mask.sum() > 0:
        metrics = {
            "silhouette": float(silhouette_score(X_pca[mask], labels[mask])),
            "davies_bouldin": float(davies_bouldin_score(X_pca[mask], labels[mask])),
            "calinski_harabasz": float(calinski_harabasz_score(X_pca[mask], labels[mask])),
        }
    else:
        metrics = {
            "silhouette": np.nan,
            "davies_bouldin": np.nan,
            "calinski_harabasz": np.nan,
        }

    metrics.update({
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "n_samples": len(df_sample),
    })

    # --------------------------------------------------
    # 6) FRAUDE POR CLUSTER
    # --------------------------------------------------
    df_fraude = pd.DataFrame({
        "cluster": labels,
        "is_fraud": y.values,
    })

    fraude_by_cluster = (
        df_fraude
        .groupby("cluster")["is_fraud"]
        .agg(
            n_transacciones="count",
            n_fraude="sum",
            pct_fraude=lambda s: 100 * s.mean()
        )
        .reset_index()
        .sort_values("pct_fraude", ascending=False)
    )

    # --------------------------------------------------
    # 7) DATAFRAME PCA
    # --------------------------------------------------
    df_pca = pd.DataFrame(
        X_pca,
        columns=["PC1", "PC2"],
        index=df_sample.index
    )
    df_pca["cluster"] = labels

    # --------------------------------------------------
    # 8) PLOT
    # --------------------------------------------------
    fig_path = params["outputs"]["figure_path"]

    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=df_pca[df_pca["cluster"] != -1],
        x="PC1", y="PC2",
        hue="cluster",
        palette="tab10",
        s=40,
        alpha=0.7,
    )

    plt.scatter(
        df_pca[df_pca["cluster"] == -1]["PC1"],
        df_pca[df_pca["cluster"] == -1]["PC2"],
        color="black",
        s=40,
        label="Ruido (-1)",
    )

    plt.title("DBSCAN - Clusters en espacio PCA")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    # --------------------------------------------------
    # OUTPUTS
    # --------------------------------------------------
    return {
        "pca_clusters": df_pca,
        "fraud_by_cluster": fraude_by_cluster,
        "metrics": metrics,
    }



def pca_optics_clustering(
    df: pd.DataFrame,
    params: Dict,
) -> Dict:
    """
    PCA (2D) + OPTICS clustering con métricas y visualización.
    """

    # --------------------------------------------------
    # 1) SAMPLE
    # --------------------------------------------------
    n_sample = params.get("n_sample", 50_000)
    df_sample = (
        df.sample(n=n_sample, random_state=42)
        if len(df) > n_sample
        else df.copy()
    )

    # --------------------------------------------------
    # 2) FEATURES
    # --------------------------------------------------
    features = params["features"]
    target = params["target"]

    X = df_sample[features]
    y = df_sample[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --------------------------------------------------
    # 3) PCA
    # --------------------------------------------------
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # --------------------------------------------------
    # 4) OPTICS
    # --------------------------------------------------
    optics = OPTICS(
        min_samples=params["optics"]["min_samples"],
        xi=params["optics"]["xi"],
        min_cluster_size=params["optics"]["min_cluster_size"],
    )

    labels = optics.fit_predict(X_pca)

    # --------------------------------------------------
    # 5) MÉTRICAS (sin ruido)
    # --------------------------------------------------
    mask = labels != -1
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))

    if n_clusters > 1 and mask.sum() > 0:
        metrics = {
            "silhouette": float(silhouette_score(X_pca[mask], labels[mask])),
            "davies_bouldin": float(davies_bouldin_score(X_pca[mask], labels[mask])),
            "calinski_harabasz": float(calinski_harabasz_score(X_pca[mask], labels[mask])),
        }
    else:
        metrics = {
            "silhouette": np.nan,
            "davies_bouldin": np.nan,
            "calinski_harabasz": np.nan,
        }

    metrics.update({
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "n_samples": len(df_sample),
    })

    # --------------------------------------------------
    # 6) FRAUDE POR CLUSTER
    # --------------------------------------------------
    df_fraude = pd.DataFrame({
        "cluster": labels,
        "is_fraud": y.values,
    })

    fraude_by_cluster = (
        df_fraude
        .groupby("cluster")["is_fraud"]
        .agg(
            n_transacciones="count",
            n_fraude="sum",
            pct_fraude=lambda s: 100 * s.mean()
        )
        .reset_index()
        .sort_values("pct_fraude", ascending=False)
    )

    # --------------------------------------------------
    # 7) DATAFRAME PCA
    # --------------------------------------------------
    df_pca = pd.DataFrame(
        X_pca,
        columns=["PC1", "PC2"],
        index=df_sample.index
    )
    df_pca["cluster"] = labels

    # --------------------------------------------------
    # 8) PLOT
    # --------------------------------------------------
    fig_path = params["outputs"]["figure_path"]

    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=df_pca[df_pca["cluster"] != -1],
        x="PC1", y="PC2",
        hue="cluster",
        palette="tab10",
        s=40,
        alpha=0.7,
    )

    plt.scatter(
        df_pca[df_pca["cluster"] == -1]["PC1"],
        df_pca[df_pca["cluster"] == -1]["PC2"],
        color="black",
        s=40,
        label="Ruido (-1)",
    )

    plt.title("OPTICS - Clusters en espacio PCA")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    return {
        "pca_clusters": df_pca,
        "fraud_by_cluster": fraude_by_cluster,
        "metrics": metrics,
    }

#clustering pca kmeans
def pca_kmeans_clustering(
    df: pd.DataFrame,
    params: Dict,
) -> Dict:
    """
    K-Means entrenado sobre X escalado.
    PCA solo para visualización 2D.
    """

    # --------------------------------------------------
    # 1) SAMPLE
    # --------------------------------------------------
    n_sample = params.get("n_sample", 50_000)
    df_sample = (
        df.sample(n=n_sample, random_state=42)
        if len(df) > n_sample
        else df.copy()
    )

    # --------------------------------------------------
    # 2) FEATURES
    # --------------------------------------------------
    features = params["features"]
    target = params["target"]

    X = df_sample[features]
    y = df_sample[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --------------------------------------------------
    # 3) ELBOW + SILHOUETTE (diagnóstico)
    # --------------------------------------------------
    inertias = {}
    silhouettes = {}

    k_range = params["kmeans"]["k_range"]

    for k in k_range:
        km = KMeans(
            n_clusters=k,
            random_state=42,
            n_init="auto",
        )
        labels_k = km.fit_predict(X_scaled)
        inertias[k] = float(km.inertia_)
        silhouettes[k] = float(silhouette_score(X_scaled, labels_k))

    # --------------------------------------------------
    # 4) K-MEANS FINAL
    # --------------------------------------------------
    k_final = params["kmeans"]["k_final"]

    kmeans = KMeans(
        n_clusters=k_final,
        random_state=42,
        n_init="auto",
    )

    labels = kmeans.fit_predict(X_scaled)

    metrics_df = pd.DataFrame([{
        "algoritmo": "KMeans",
        "k_final": k_final,
        "n_samples": len(df_sample),
        "silhouette": float(silhouette_score(X_scaled, labels)),
        "davies_bouldin": float(davies_bouldin_score(X_scaled, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X_scaled, labels)),
    }])

    # --------------------------------------------------
    # 5) FRAUDE POR CLUSTER
    # --------------------------------------------------
    df_fraude = pd.DataFrame({
        "cluster": labels,
        "is_fraud": y.values,
    })

    fraude_by_cluster = (
        df_fraude
        .groupby("cluster")["is_fraud"]
        .agg(
            n_transacciones="count",
            n_fraude="sum",
            pct_fraude=lambda s: 100 * s.mean()
        )
        .reset_index()
        .sort_values("pct_fraude", ascending=False)
    )

    # --------------------------------------------------
    # 6) PCA PARA VISUALIZACIÓN
    # --------------------------------------------------
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    df_pca = pd.DataFrame(
        X_pca,
        columns=["PC1", "PC2"],
        index=df_sample.index,
    )
    df_pca["cluster"] = labels

    # --------------------------------------------------
    # 7) PLOT PCA
    # --------------------------------------------------
    fig_path = params["outputs"]["figure_path"]

    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=df_pca,
        x="PC1",
        y="PC2",
        hue="cluster",
        palette="tab10",
        s=40,
        alpha=0.7,
    )
    plt.title("K-Means - Clusters en espacio PCA (2D)")
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    # --------------------------------------------------
    # OUTPUTS
    # --------------------------------------------------
    return {
        "pca_clusters": df_pca,
        "fraud_by_cluster": fraude_by_cluster,
        "kmeans_metrics": metrics_df,
        "kmeans_diagnostics": pd.DataFrame({
            "k": list(inertias.keys()),
            "inertia": list(inertias.values()),
            "silhouette": list(silhouettes.values()),
        }),
    }

#tsne 
def tsne_kmeans_visualization(
    df_sample: pd.DataFrame,
    X_scaled: pd.DataFrame,
    kmeans_labels: pd.Series,
    params: Dict,
) -> Dict:
    """
    t-SNE solo para visualización del clustering K-Means
    + análisis de fraude por cluster.
    """

    # ---------------------------
    # t-SNE
    # ---------------------------
    tsne = TSNE(
        n_components=2,
        perplexity=params["tsne"]["perplexity"],
        learning_rate=params["tsne"]["learning_rate"],
        init="pca",
        method="barnes_hut",
        angle=0.5,
        random_state=42,
        max_iter=params["tsne"]["n_iter"],
    )

    X_tsne = tsne.fit_transform(X_scaled)

    df_tsne = pd.DataFrame(
        X_tsne,
        columns=["TSNE1", "TSNE2"],
        index=df_sample.index,
    )

    df_tsne["cluster"] = kmeans_labels.values
    df_tsne["is_fraud"] = df_sample[params["target"]].values

    # ---------------------------
    # Fraude por cluster (t-SNE)
    # ---------------------------
    fraude_by_cluster = (
        df_tsne
        .groupby("cluster")["is_fraud"]
        .agg(
            n_transacciones="count",
            n_fraude="sum",
            pct_fraude=lambda x: 100 * x.mean()
        )
        .reset_index()
        .sort_values("pct_fraude", ascending=False)
    )

    return {
        "tsne_embedding": df_tsne,
        "fraud_by_cluster_tsne": fraude_by_cluster,
    }

def assign_kmeans_cluster_to_full_dataset(
    df_full: pd.DataFrame,
    scaler,
    kmeans_model,
    params: Dict,
) -> pd.DataFrame:
    """
    Genera cluster_id para TODO el dataset usando
    el mismo scaler y modelo K-Means entrenado.
    """

    features = params["features"]

    X_full = df_full[features]
    X_full_scaled = scaler.transform(X_full)

    df_out = df_full.copy()
    df_out["cluster_id"] = kmeans_model.predict(X_full_scaled)

    return df_out