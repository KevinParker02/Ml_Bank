# =========================================================
# 🧩 Pipeline de Clustering No Supervisado (DBSCAN, OPTICS, K-Means)
# =========================================================
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, OPTICS, KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

# =========================================================
# 🗂️ Creación de carpetas necesarias
# =========================================================
def ensure_data_folders(base_path="data"):
    folders = {
        "07_model_output": ["clustering"],
        "08_reporting": ["clustering"],
        "05_model_input": []
    }
    for folder, subfolders in folders.items():
        path = os.path.join(base_path, folder)
        os.makedirs(path, exist_ok=True)
        for sub in subfolders:
            os.makedirs(os.path.join(path, sub), exist_ok=True)

ensure_data_folders()


# =========================================================
# 🧠 Función principal (siguiendo el notebook 05)
# =========================================================
def run_unsupervised_learning(df: pd.DataFrame, params: dict) -> dict:
    """
    Ejecuta el pipeline completo de clustering:
    - DBSCAN
    - OPTICS
    - KMeans
    Reproduce el flujo del notebook 05, guardando resultados y gráficos.
    """

    # =========================================================
    # 1️⃣ Muestra del dataset
    # =========================================================
    n_muestra = 50_000
    df_sample = df.sample(n=n_muestra, random_state=42) if len(df) > n_muestra else df.copy()
    print(f"Filas totales: {len(df)} | Filas usadas en análisis: {len(df_sample)}")

    # =========================================================
    # 2️⃣ Features y escalado
    # =========================================================
    X = df_sample[["AmountZScoreByLocation", "IsAnomaly", "IsLateNight", "IsWeekend"]]
    y_sample = df_sample["is_fraud"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # =========================================================
    # 3️⃣ PCA
    # =========================================================
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    print("Shape X_pca:", X_pca.shape)

    # =========================================================
    # Helper: función para métricas
    # =========================================================
    def get_metrics(X, labels):
        mask = labels != -1
        if len(set(labels)) - (1 if -1 in labels else 0) > 1 and mask.sum() > 0:
            sil = silhouette_score(X[mask], labels[mask])
            db = davies_bouldin_score(X[mask], labels[mask])
            ch = calinski_harabasz_score(X[mask], labels[mask])
        else:
            sil = db = ch = np.nan
        return sil, db, ch

    # =========================================================
    # Helper: función de fraude por cluster
    # =========================================================
    def fraude_por_cluster(labels, y):
        df_temp = pd.DataFrame({"cluster": labels, "is_fraud": y.values})
        return df_temp.groupby("cluster")["is_fraud"].agg(
            n_transacciones="count",
            n_fraude="sum",
            pct_fraude=lambda s: 100 * s.mean()
        ).sort_values("pct_fraude", ascending=False)

    # =========================================================
    # 4️⃣ DBSCAN
    # =========================================================
    dbscan = DBSCAN(eps=params.get("dbscan_eps", 0.2), min_samples=params.get("dbscan_min_samples", 5))
    labels_dbscan = dbscan.fit_predict(X_pca)
    n_clusters = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
    n_ruido = np.sum(labels_dbscan == -1)

    sil, db, ch = get_metrics(X_pca, labels_dbscan)
    print("\n📊 DBSCAN:")
    print(f"Clusters: {n_clusters}, Ruido: {n_ruido}, Silhouette: {sil:.4f}")

    df_resumen_dbscan = pd.DataFrame({
        "algoritmo": ["DBSCAN"],
        "n_muestra": [len(df_sample)],
        "eps": [dbscan.eps],
        "min_samples": [dbscan.min_samples],
        "n_clusters": [n_clusters],
        "n_ruido": [n_ruido],
        "silhouette": [sil],
        "davies_bouldin": [db],
        "calinski_harabasz": [ch]
    })

    df_fraude_dbscan = fraude_por_cluster(labels_dbscan, y_sample)
    df_plot = pd.DataFrame({"PC1": X_pca[:, 0], "PC2": X_pca[:, 1], "cluster": labels_dbscan})

    # --- Gráfico DBSCAN
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=df_plot[df_plot["cluster"] != -1], x="PC1", y="PC2", hue="cluster", palette="tab10", s=40, alpha=0.7)
    plt.scatter(df_plot[df_plot["cluster"] == -1]["PC1"], df_plot[df_plot["cluster"] == -1]["PC2"], color="black", s=50, label="Ruido (-1)")
    plt.title("DBSCAN - Clusters en PCA")
    plt.legend()
    plt.tight_layout()
    plt.savefig("data/08_reporting/clustering/dbscan_clusters.png", dpi=300)
    plt.close()

    # Guardar resultados DBSCAN
    df_plot.to_csv("data/07_model_output/clustering/dbscan_pca_labels.csv", index=False)
    df_fraude_dbscan.to_csv("data/07_model_output/clustering/dbscan_fraud_by_cluster.csv")
    df_resumen_dbscan.to_json("data/07_model_output/clustering/dbscan_metrics.json", orient="records", indent=2)

    # =========================================================
    # 5️⃣ OPTICS
    # =========================================================
    optics = OPTICS(min_samples=50, xi=0.05, min_cluster_size=0.02)
    labels_optics = optics.fit_predict(X_pca)
    n_clusters_optics = len(set(labels_optics)) - (1 if -1 in labels_optics else 0)
    n_ruido_optics = np.sum(labels_optics == -1)

    sil_o, db_o, ch_o = get_metrics(X_pca, labels_optics)
    print("\n📊 OPTICS:")
    print(f"Clusters: {n_clusters_optics}, Ruido: {n_ruido_optics}, Silhouette: {sil_o:.4f}")

    df_resumen_optics = pd.DataFrame({
        "algoritmo": ["OPTICS"],
        "n_muestra": [len(df_sample)],
        "min_samples": [optics.min_samples],
        "xi": [optics.xi],
        "min_cluster_size": [optics.min_cluster_size],
        "n_clusters": [n_clusters_optics],
        "n_ruido": [n_ruido_optics],
        "silhouette": [sil_o],
        "davies_bouldin": [db_o],
        "calinski_harabasz": [ch_o]
    })

    df_fraude_optics = fraude_por_cluster(labels_optics, y_sample)
    df_optics_plot = pd.DataFrame({"PC1": X_pca[:, 0], "PC2": X_pca[:, 1], "cluster": labels_optics})

    # --- Gráfico OPTICS
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=df_optics_plot[df_optics_plot["cluster"] != -1], x="PC1", y="PC2", hue="cluster", palette="tab10", s=40, alpha=0.7)
    plt.scatter(df_optics_plot[df_optics_plot["cluster"] == -1]["PC1"], df_optics_plot[df_optics_plot["cluster"] == -1]["PC2"],
                color="black", s=50, label="Ruido (-1)")
    plt.title("OPTICS - Clusters en PCA")
    plt.legend()
    plt.tight_layout()
    plt.savefig("data/08_reporting/clustering/optics_clusters.png", dpi=300)
    plt.close()

    # --- Reachability Plot
    reachability = optics.reachability_
    ordering = optics.ordering_
    plt.figure(figsize=(12, 5))
    plt.plot(ordering, reachability[ordering], marker=".", alpha=0.6)
    plt.title("Reachability Plot (OPTICS)")
    plt.xlabel("Índice ordenado")
    plt.ylabel("Reachability Distance")
    plt.tight_layout()
    plt.savefig("data/08_reporting/clustering/optics_reachability.png", dpi=300)
    plt.close()

    # Guardar resultados OPTICS
    df_optics_plot.to_csv("data/07_model_output/clustering/optics_pca_labels.csv", index=False)
    df_fraude_optics.to_csv("data/07_model_output/clustering/optics_fraud_by_cluster.csv")
    df_resumen_optics.to_json("data/07_model_output/clustering/optics_metrics.json", orient="records", indent=2)

    # =========================================================
    # 6️⃣ K-Means
    # =========================================================
    X_kmeans = X_scaled
    k_range = range(2, 11)
    inertias, sil_scores = [], []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        km.fit(X_kmeans)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_kmeans, km.labels_))

    # --- Elbow y Silhouette plots
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertias, marker="o")
    plt.title("Método del Codo (KMeans)")
    plt.xlabel("Número de clusters (k)")
    plt.ylabel("Inercia (SSE)")
    plt.tight_layout()
    plt.savefig("data/08_reporting/clustering/kmeans_elbow.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(k_range, sil_scores, marker="o", color="green")
    plt.title("Silhouette Score por k (KMeans)")
    plt.xlabel("Número de clusters (k)")
    plt.ylabel("Silhouette")
    plt.tight_layout()
    plt.savefig("data/08_reporting/clustering/kmeans_silhouette_scores.png", dpi=300)
    plt.close()

    k_final = params.get("kmeans_k", 6)
    kmeans_final = KMeans(n_clusters=k_final, random_state=42, n_init="auto")
    labels_kmeans = kmeans_final.fit_predict(X_kmeans)

    sil_k, db_k, ch_k = get_metrics(X_kmeans, labels_kmeans)
    print("\n📊 KMEANS:")
    print(f"Clusters: {k_final}, Silhouette: {sil_k:.4f}")

    df_resumen_kmeans = pd.DataFrame({
        "algoritmo": ["KMeans"],
        "k": [k_final],
        "silhouette": [sil_k],
        "davies_bouldin": [db_k],
        "calinski_harabasz": [ch_k]
    })
    df_fraude_kmeans = fraude_por_cluster(labels_kmeans, y_sample)

    # --- PCA visualización
    pca2 = PCA(n_components=2)
    X_pca2 = pca2.fit_transform(X_kmeans)
    df_plot_kmeans = pd.DataFrame({"PC1": X_pca2[:, 0], "PC2": X_pca2[:, 1], "cluster": labels_kmeans})
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=df_plot_kmeans, x="PC1", y="PC2", hue="cluster", palette="tab10", s=40, alpha=0.7)
    plt.title("KMeans - Clusters PCA 2D")
    plt.tight_layout()
    plt.savefig("data/08_reporting/clustering/kmeans_pca.png", dpi=300)
    plt.close()

    # --- t-SNE visualización
    tsne = TSNE(n_components=2, perplexity=50, learning_rate="auto", init="pca",
                method="barnes_hut", angle=0.5, random_state=42, max_iter=1000)
    X_tsne_vis = tsne.fit_transform(X_kmeans)
    df_tsne_plot = pd.DataFrame({"TSNE1": X_tsne_vis[:, 0], "TSNE2": X_tsne_vis[:, 1], "cluster": labels_kmeans, "is_fraud": y_sample.values})
    df_tsne_plot.to_csv("data/07_model_output/clustering/kmeans_tsne_embedding.csv", index=False)
    df_fraude_kmeans.to_csv("data/07_model_output/clustering/kmeans_fraud_by_cluster_tsne.csv")
    df_resumen_kmeans.to_csv("data/07_model_output/clustering/kmeans_metrics.csv", index=False)

    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=df_tsne_plot, x="TSNE1", y="TSNE2", hue="cluster", palette="tab10", s=40, alpha=0.7)
    plt.title(f"KMeans (k={k_final}) - t-SNE visualización")
    plt.tight_layout()
    plt.savefig("data/08_reporting/clustering/kmeans_tsne.png", dpi=300)
    plt.close()

    # =========================================================
    # 7️⃣ Generar cluster_id en todo el dataset y guardar
    # =========================================================
    X_full = df[["AmountZScoreByLocation", "IsAnomaly", "IsLateNight", "IsWeekend"]]
    X_full_scaled = scaler.transform(X_full)
    df["cluster_id"] = kmeans_final.predict(X_full_scaled)
    df.to_parquet("data/05_model_input/Features_clustering_v1.parquet", index=False)
    print("✅ Dataset completo guardado como Features_clustering_v1.parquet")

    # 👇 Kedro solo necesita el DataFrame final
    return df


# =========================================================
# 🧠 TASK 2: Evaluar mejor modelo de clasificación con Features_clustering_v1
# =========================================================
from xgboost import XGBClassifier
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, train_test_split
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve, roc_auc_score
)

def run_best_xgb_model(df, params):
    """
    Entrena y evalúa el mejor modelo XGBoost + SMOTE
    usando Features_clustering_v1 generado previamente.
    Guarda métricas, matriz de confusión y curva ROC-AUC.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    import joblib

    os.makedirs("data/08_reporting/clasificacion", exist_ok=True)
    os.makedirs("data/07_model_output/clasificacion", exist_ok=True)

    # =========================================================
    # 1️⃣ Preparar datos
    # =========================================================
    X = df.drop("is_fraud", axis=1)
    y = df["is_fraud"]

    neg, pos = (y == 0).sum(), (y == 1).sum()
    ratio = neg / pos
    print(f"Ratio clases (neg/pos): {ratio:.3f}")

    # =========================================================
    # 2️⃣ Pipeline SMOTE + XGBoost
    # =========================================================
    smote = SMOTE(random_state=42)

    best_params = {
        "subsample": 0.8,
        "scale_pos_weight": 32.9976809915676,
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.1,
        "colsample_bytree": 1.0,
    }

    xgb = XGBClassifier(
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
        tree_method="approx",
        **best_params
    )

    pipe = Pipeline([
        ("smote", smote),
        ("model", xgb)
    ])

    # =========================================================
    # 3️⃣ Cross-validation F1 score
    # =========================================================
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1", n_jobs=-1)
    f1_mean, f1_std = scores.mean(), scores.std()
    print(f"F1 promedio CV (2 folds): {f1_mean:.6f} ± {f1_std:.6f}")

    # =========================================================
    # 4️⃣ Entrenamiento final + evaluación
    # =========================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("\nEntrenando modelo final...")
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    # --- Reporte
    report = classification_report(y_test, y_pred, digits=3, output_dict=True)
    print("\n📊 CLASSIFICATION REPORT (MEJOR MODELO)")
    print(classification_report(y_test, y_pred, digits=3))

    # --- Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Fraude", "Fraude"])
    disp.plot(cmap="Blues")
    plt.title("Matriz de Confusión – XGBoost + SMOTE")
    plt.tight_layout()
    plt.savefig("data/08_reporting/clasificacion/confusion_matrix_xgb.png", dpi=300)
    plt.close()

    # --- ROC–AUC
    y_pred_proba = pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", label="Azar (0.5)")
    plt.xlabel("Falsos Positivos (FPR)")
    plt.ylabel("Verdaderos Positivos (TPR)")
    plt.title("Curva ROC – XGBoost + SMOTE")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("data/08_reporting/clasificacion/roc_auc_xgb.png", dpi=300)
    plt.close()

    # =========================================================
    # 5️⃣ Guardar resultados
    # =========================================================
    results = {
        "f1_mean_cv": round(f1_mean, 6),
        "f1_std_cv": round(f1_std, 6),
        "roc_auc": round(auc_score, 6),
        "classification_report": report
    }

    import json
    with open("data/07_model_output/clasificacion/xgb_final_metrics.json", "w") as f:
        json.dump(results, f, indent=4)

    # Guardar modelo entrenado
    joblib.dump(pipe, "data/06_models/clasificacion/xgb_final_model.pkl")
    print("✅ Modelo final y métricas guardadas correctamente.")

    return df


