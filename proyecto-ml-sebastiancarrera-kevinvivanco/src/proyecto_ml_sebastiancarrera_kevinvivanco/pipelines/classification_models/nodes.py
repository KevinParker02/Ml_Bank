import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve
)

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
# Función auxiliar para guardar curvas y matrices
# =========================================================
def save_classification_plots(model_name, pipe, X_test, y_test, y_pred):
    """Guarda la curva ROC y la matriz de confusión para un modelo dado."""
    os.makedirs("data/08_reporting/clasificacion", exist_ok=True)

    # --- Curva ROC ---
    if hasattr(pipe.named_steps["model"], "predict_proba"):
        y_prob = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)

        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {auc:.2f})")
        plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Curva ROC – {model_name}")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(f"data/08_reporting/clasificacion/{model_name.lower()}_roc_curve.png")
        plt.close()

    # --- Matriz de confusión ---
    cm = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Fraude", "Fraude"])
    cm_display.plot(cmap="Blues")
    plt.title(f"Matriz de Confusión – {model_name}")
    plt.tight_layout()
    plt.savefig(f"data/08_reporting/clasificacion/{model_name.lower()}_confusion_matrix.png")
    plt.close()


# =========================================================
# Nodo 1 – KNN
# =========================================================
def train_knn_classifier(df: pd.DataFrame, knn_model_path: str) -> dict:
    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]

    scaler = StandardScaler()
    smote = SMOTE(random_state=42)
    knn = KNeighborsClassifier(n_neighbors=5)

    pipe = Pipeline([
        ("scaler", scaler),
        ("smote", smote),
        ("model", knn)
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    pipe.fit(X_train, y_train)
    joblib.dump(pipe, knn_model_path)

    y_pred = pipe.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    # =========================================================
    # ✅ Cálculo de AUC (aunque el pipeline sea imblearn)
    # =========================================================
    auc = None
    try:
        # Si el modelo tiene predict_proba, calculamos AUC
        if hasattr(pipe.named_steps["model"], "predict_proba"):
            y_prob = pipe.named_steps["model"].predict_proba(
                pipe.named_steps["scaler"].transform(X_test)
            )[:, 1]
            auc = roc_auc_score(y_test, y_prob)
    except Exception as e:
        print(f"[WARN] No se pudo calcular AUC para KNN: {e}")

    save_classification_plots("KNeighborsClassifier", pipe, X_test, y_test, y_pred)

    return {
        "model_name": "KNeighborsClassifier",
        "f1_mean_cv": float(scores.mean()),
        "f1_std_cv": float(scores.std()),
        "auc": float(auc) if auc else None,
        "report": report,
        "confusion_matrix": cm
    }


# =========================================================
# Nodo 2 – DecisionTreeClassifier
# =========================================================
def train_decision_tree(df: pd.DataFrame, dt_model_path: str) -> dict:
    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]

    smote = SMOTE(random_state=42)
    tree = DecisionTreeClassifier(class_weight="balanced", random_state=42)
    pipe = Pipeline([("smote", smote), ("model", tree)])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1")

    param_grid = {
        "model__max_depth": [None, 6, 10, 16, 24],
        "model__min_samples_leaf": [1, 3, 5, 10],
        "model__criterion": ["gini", "entropy", "log_loss"]
    }

    grid = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=cv, scoring="f1", n_jobs=-1, verbose=1)
    grid.fit(X, y)
    best_pipe = grid.best_estimator_

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    final_model = best_pipe
    final_model.fit(X_train, y_train)

    y_pred = final_model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    auc = None
    if hasattr(final_model.named_steps["model"], "predict_proba"):
        y_prob = final_model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)

    joblib.dump(final_model, dt_model_path)

    save_classification_plots("DecisionTreeClassifier", final_model, X_test, y_test, y_pred)

    return {
        "model_name": "DecisionTreeClassifier",
        "f1_mean_cv": float(scores.mean()),
        "f1_std_cv": float(scores.std()),
        "best_params": grid.best_params_,
        "f1_best_cv": float(grid.best_score_),
        "auc": float(auc) if auc else None,
        "report": report,
        "confusion_matrix": cm
    }


# =========================================================
# Nodo 3 – RandomForestClassifier
# =========================================================
def train_random_forest(df: pd.DataFrame, rf_model_path: str) -> dict:
    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]

    smote = SMOTE(random_state=42)
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=None, min_samples_leaf=3,
        class_weight="balanced", random_state=42, n_jobs=-1
    )

    pipe = Pipeline([("smote", smote), ("model", rf)])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])

    joblib.dump(pipe, rf_model_path)

    save_classification_plots("RandomForestClassifier", pipe, X_test, y_test, y_pred)

    return {
        "model_name": "RandomForestClassifier",
        "f1_mean_cv": float(scores.mean()),
        "f1_std_cv": float(scores.std()),
        "auc": float(auc),
        "report": report,
        "confusion_matrix": cm
    }


# =========================================================
# Nodo 4 – XGBoostClassifier
# =========================================================
def train_xgboost_classifier(df: pd.DataFrame, xgb_model_path: str) -> dict:
    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]

    smote = SMOTE(random_state=42)
    xgb = XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=10,
        eval_metric="auc", random_state=42, n_jobs=-1
    )

    pipe = Pipeline([("smote", smote), ("model", xgb)])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()
    auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])

    joblib.dump(pipe, xgb_model_path)

    save_classification_plots("XGBClassifier", pipe, X_test, y_test, y_pred)

    return {
        "model_name": "XGBClassifier",
        "f1_mean_cv": float(scores.mean()),
        "f1_std_cv": float(scores.std()),
        "auc": float(auc),
        "report": report,
        "confusion_matrix": cm
    }


# =========================================================
# Nodo 5 – MLPClassifier
# =========================================================
def train_mlp_classifier(df: pd.DataFrame, mlp_model_path: str) -> dict:
    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("model", MLPClassifier(
            hidden_layer_sizes=(32, 16), activation="relu", solver="adam",
            alpha=1e-4, learning_rate_init=1e-3, max_iter=200,
            early_stopping=True, n_iter_no_change=10, random_state=42
        ))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()
    auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])

    joblib.dump(pipe, mlp_model_path)

    save_classification_plots("MLPClassifier", pipe, X_test, y_test, y_pred)

    return {
        "model_name": "MLPClassifier",
        "f1_mean_cv": float(scores.mean()),
        "f1_std_cv": float(scores.std()),
        "auc": float(auc),
        "report": report,
        "confusion_matrix": cm
    }
