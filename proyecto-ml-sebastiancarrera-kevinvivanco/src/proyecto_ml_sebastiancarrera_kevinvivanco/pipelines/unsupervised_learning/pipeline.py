"""
This is a boilerplate pipeline 'unsupervised_learning'
generated using Kedro 1.0.0
"""

from kedro.pipeline import node, Pipeline
from .nodes import pca_dbscan_clustering, pca_optics_clustering, pca_kmeans_clustering, tsne_kmeans_visualization, assign_kmeans_cluster_to_full_dataset

def create_clustering_pipeline() -> Pipeline:
    return Pipeline(
        [
            node(
                func=pca_dbscan_clustering,
                inputs=["Features_training_v1", "params:pca_dbscan"],
                outputs={
                    "pca_clusters": "dbscan_pca_labels",
                    "fraud_by_cluster": "dbscan_fraud_by_cluster",
                    "metrics": "dbscan_metrics",
                },
                name="pca_dbscan_clustering_node",
            ),
              node(
                func=pca_optics_clustering,
                inputs=["Features_training_v1", "params:pca_optics"],
                outputs={
                    "pca_clusters": "optics_pca_labels",
                    "fraud_by_cluster": "optics_fraud_by_cluster",
                    "metrics": "optics_metrics",
                },
                name="pca_optics_clustering_node",
            ),
            node(
                func=pca_kmeans_clustering,
                inputs=["Features_training_v1", "params:pca_kmeans"],
                outputs=dict(
                    pca_clusters="kmeans_pca_labels",
                    fraud_by_cluster="kmeans_fraud_by_cluster",
                    metrics="kmeans_metrics",
                    diagnostics="kmeans_diagnostics",
                    kmeans_model="kmeans_model",
                    kmeans_scaler="kmeans_scaler",
                    kmeans_labels="kmeans_labels",
                    kmeans_X_scaled="kmeans_X_scaled",
                ),
                name="pca_kmeans_clustering_node",
            ),
                node(
                func=tsne_kmeans_visualization,
                inputs=dict(
                    df_sample="Features_training_v1",
                    X_scaled="kmeans_X_scaled",
                    kmeans_labels="kmeans_labels",
                    params="params:clustering",
                ),
                outputs=dict(
                    tsne_embedding="kmeans_tsne_embedding",
                    fraud_by_cluster_tsne="kmeans_fraud_by_cluster_tsne",
                ),
                name="kmeans_tsne_node",
            ),
            node(
                func=assign_kmeans_cluster_to_full_dataset,
                inputs=dict(
                    df_full="Features_training_v1",
                    scaler="kmeans_scaler",
                    kmeans_model="kmeans_model",
                    params="params:clustering",
                ),
                outputs="Features_clustering_v1",
                name="assign_kmeans_cluster_node",
            ),
        ]
    )



