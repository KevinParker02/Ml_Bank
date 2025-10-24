import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# =========================================================
# 🧠 Nodo 1 – Entrenamiento del modelo KNN con SMOTE
# =========================================================
def train_knn_classifier(df: pd.DataFrame, knn_model_path: str):
    """Entrena y evalúa un modelo KNN con SMOTE + StandardScaler y guarda el modelo."""

    # === Variables independientes y dependiente ===
    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]

    print("Shape X:", X.shape)
    print("Shape y:", y.shape)

    # === Pipeline: escalado + SMOTE + KNN ===
    scaler = StandardScaler()
    smote = SMOTE(random_state=42)
    knn = KNeighborsClassifier(n_neighbors=5)

    pipe = ImbPipeline([
        ("scaler", scaler),
        ("smote", smote),
        ("model", knn)
    ])

    # === Validación cruzada ===
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1")

    print(f"F1 promedio CV: {scores.mean():.4f} ± {scores.std():.4f}")

    # === División train/test ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # === Entrenamiento final ===
    pipe.fit(X_train, y_train)
    print("✅ Modelo KNN entrenado correctamente")

    # === Guardar modelo entrenado ===
    joblib.dump(pipe, knn_model_path)
    print(f"💾 Modelo guardado en: {knn_model_path}")

    # === Predicciones y métricas ===
    y_pred = pipe.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    # Mostrar resultados básicos
    print(classification_report(y_test, y_pred, digits=3))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Fraude", "Fraude"])
    disp.plot(cmap="Blues")
    plt.title("Matriz de Confusión – KNN con SMOTE")
    plt.show()

    # Retornar métricas y matriz para reporting
    return {
        "model_name": "KNeighborsClassifier",
        "f1_mean_cv": scores.mean(),
        "f1_std_cv": scores.std(),
        "report": report,
        "confusion_matrix": cm.tolist(),
    }