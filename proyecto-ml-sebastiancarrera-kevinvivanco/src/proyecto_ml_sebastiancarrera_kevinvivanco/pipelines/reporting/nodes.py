import os
import glob
import joblib
from IPython.display import display, HTML


def generate_model_dashboard(reporting_config: dict):
    """Genera un dashboard visual combinando resultados de clasificación, regresión y clustering."""

    class_results_path = reporting_config["classification_results_path"]
    reg_results_path = reporting_config["regression_results_path"]
    class_img_path = reporting_config["classification_images_path"]
    reg_img_path = reporting_config["regression_images_path"]
    clust_img_path = reporting_config.get("clustering_images_path")  # 👈 NUEVO
    output_path = reporting_config["dashboard_output_path"]

    os.makedirs(output_path, exist_ok=True)

    # --- Leer pickles de ambos tipos ---
    pkl_class = glob.glob(f"{class_results_path}/*.pkl")
    pkl_reg = glob.glob(f"{reg_results_path}/*.pkl")

    class_models = [joblib.load(f) for f in pkl_class]
    reg_models = [joblib.load(f) for f in pkl_reg]

    # --- HTML base ---
    html = """
    <style>
        body { font-family: 'Segoe UI', sans-serif; background-color: #fafafa; margin: 40px; }
        h1 { color: #1a73e8; border-bottom: 2px solid #1a73e8; }
        h2 { color: #333; }
        h3 { color: #444; margin-top: 25px; }
        table { width: 100%; border-collapse: collapse; margin-bottom: 40px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
        th { background-color: #1a73e8; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        img { border-radius: 8px; margin: 10px auto; max-width: 500px; display: block; }
        .section { margin-bottom: 50px; }
    </style>
    <h1>📊 Dashboard Visual – Modelos Clasificación, Regresión y Clustering</h1>
    """

    # ======================================================
    # 📘 CLASIFICACIÓN
    # ======================================================
    if class_models:
        html += "<div class='section'><h2>📘 Modelos de Clasificación</h2>"
        html += "<table><tr><th>Modelo</th><th>F1 (CV)</th><th>AUC</th><th>F1 Test</th><th>Precisión</th><th>Recall</th><th>Gráficos</th></tr>"
        for res in class_models:
            name = res["model_name"]
            f1_mean_std = f"{res['f1_mean_cv']:.3f} ± {res['f1_std_cv']:.3f}"
            auc = res.get("auc", "—")
            report = res.get("report", {}).get("1", {})
            f1_test = report.get("f1-score", "—")
            prec = report.get("precision", "—")
            rec = report.get("recall", "—")

            # Buscar imágenes
            imgs = []
            if os.path.exists(class_img_path):
                for img in os.listdir(class_img_path):
                    if img.lower().startswith(name.lower()):
                        imgs.append(f"<img src='../clasificacion/{img}' alt='{img}'>")
            img_html = "".join(imgs) if imgs else "<em style='color:gray'>⚠️ No se encontraron gráficos</em>"

            html += f"""
            <tr>
                <td>{name}</td>
                <td>{f1_mean_std}</td>
                <td>{auc if isinstance(auc, str) else f"{auc:.3f}"}</td>
                <td>{f1_test}</td>
                <td>{prec}</td>
                <td>{rec}</td>
                <td>{img_html}</td>
            </tr>
            """
        html += "</table></div>"

    # ======================================================
    # 📗 REGRESIÓN
    # ======================================================
    if reg_models:
        html += "<div class='section'><h2>📗 Modelos de Regresión</h2>"
        html += "<table><tr><th>Modelo</th><th>R² (CV)</th><th>R² Test</th><th>RMSE</th><th>MAE</th><th>Gráficos</th></tr>"
        for res in reg_models:
            name = res["model_name"]
            r2_mean_std = f"{res['r2_mean_cv']:.3f} ± {res['r2_std_cv']:.3f}"
            r2_test = res["r2_test"]
            rmse = res["rmse"]
            mae = res["mae"]

            imgs = []
            if os.path.exists(reg_img_path):
                key = name.lower().replace(" ", "").replace("_", "")

                alias_map = {
                    "linearregression": ["linear_regression"],
                    "ridgeregression": ["ridge_regressor"],
                    "lassoregression": ["lasso_regressor"],
                    "randomforestregressor": ["rf_regressor", "random_forest_regressor"],
                    "xgbregressor": ["xgb_regressor", "xgboost_regressor"]
                }

                variants = [key]
                if key in alias_map:
                    variants.extend(alias_map[key])

                for img in os.listdir(reg_img_path):
                    img_lower = img.lower()
                    if any(img_lower.startswith(v) for v in variants):
                        imgs.append(f"<img src='../regresion/{img}' alt='{img}'>")

            img_html = "".join(imgs) if imgs else "<em style='color:gray'>⚠️ No se encontraron gráficos</em>"

            html += f"""
            <tr>
                <td>{name}</td>
                <td>{r2_mean_std}</td>
                <td>{r2_test:.3f}</td>
                <td>{rmse:.3f}</td>
                <td>{mae:.3f}</td>
                <td>{img_html}</td>
            </tr>
            """
        html += "</table></div>"

    # ======================================================
    # 🧩 CLUSTERING (DBSCAN / OPTICS / K-MEANS)
    # ======================================================
    if clust_img_path and os.path.exists(clust_img_path):
        html += "<div class='section'><h2>🧩 Análisis de Clustering (No Supervisado)</h2>"

        title_map = {
            "dbscan_clusters": "Clusters encontrados con DBSCAN (PCA)",
            "optics_reachability": "Reachability Plot – OPTICS",
            "optics_clusters": "Clusters encontrados con OPTICS",
            "kmeans_elbow": "Método del Codo – K-Means",
            "kmeans_pca": "K-Means sobre PCA",
            "kmeans_silhouette_scores": "Scores de Silhouette (K-Means)",
            "kmeans_tsne": "K-Means sobre t-SNE (50K muestras)"
        }

        for img in sorted(os.listdir(clust_img_path)):
            if not img.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            base = os.path.splitext(img)[0]
            title = title_map.get(base, base)

            html += f"""
            <div class='section'>
                <h3>{title}</h3>
                <img src='../clustering/{img}' alt='{img}'>
            </div>
            """

        html += "</div>"

    # ======================================================
    # 🟦 SECCIÓN FINAL – XGBOOST MEJORADO
    # ======================================================
    xgb_special_images = [
        "confusion_matrix_xgb.png",
        "roc_auc_xgb.png"
    ]

    html += "<div class='section'><h2>🟦 Resultados Finales – XGBoost Mejorado</h2>"

    for img in xgb_special_images:
        full = os.path.join(class_img_path, img)
        if os.path.exists(full):
            title = (
                "Matriz de Confusión – XGBoost Mejorado"
                if "confusion" in img
                else "Curva ROC–AUC – XGBoost Mejorado"
            )

            html += f"""
            <div class='section'>
                <h3>{title}</h3>
                <img src='../clasificacion/{img}' alt='{img}'>
            </div>
            """
        else:
            html += f"<p style='color:red'>⚠️ No se encontró la imagen: {img}</p>"

    html += "</div>"

    # ======================================================
    # 💾 GUARDAR
    # ======================================================
    output_file = os.path.join(output_path, "model_dashboard.html")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✅ Dashboard generado en: {output_file}")
    display(HTML(html))
