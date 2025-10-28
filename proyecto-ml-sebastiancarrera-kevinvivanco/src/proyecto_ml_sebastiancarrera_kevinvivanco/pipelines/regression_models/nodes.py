# =========================================================
# Imports generales para todos los modelos de regresión
# =========================================================
import os
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Modelos
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor



def train_linear_regression(df: pd.DataFrame, linear_model_path: str) -> dict:
    """
    Entrena un modelo de Regresión Lineal Múltiple para predecir Monetary.
    Guarda el modelo, los resultados y genera gráficos de validación y residuos.
    """

    # =========================================================
    # 1️⃣ Variables dependiente e independientes
    # =========================================================
    X = df[["AmountZScoreByLocation", "TimeSinceLastTxn", "IsLateNight", "IsWeekend"]]
    y = df["Monetary"]

    # =========================================================
    # 2️⃣ Train/Test Split
    # =========================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # =========================================================
    # 3️⃣ Escalamiento
    # =========================================================
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # =========================================================
    # 4️⃣ Entrenamiento + Cross Validation
    # =========================================================
    model = LinearRegression()
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_r2 = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="r2")
    cv_rmse = np.sqrt(
        -cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="neg_mean_squared_error")
    )

    # Entrenar modelo final
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # =========================================================
    # 5️⃣ Evaluación final
    # =========================================================
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    results = {
        "model_name": "LinearRegression",
        "r2_mean_cv": round(cv_r2.mean(), 4),
        "r2_std_cv": round(cv_r2.std(), 4),
        "r2_test": round(r2, 4),
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
    }

    # =========================================================
    # 6️⃣ Guardar modelo entrenado
    # =========================================================
    joblib.dump(model, linear_model_path)

    # =========================================================
    # 7️⃣ Generar gráficos
    # =========================================================
    os.makedirs("data/08_reporting/regresion", exist_ok=True)

    # --- Scatterplot: reales vs predichos
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    plt.xlabel("Valores reales (Monetary)")
    plt.ylabel("Predicciones")
    plt.title("Regresión Lineal – Reales vs Predichos")
    plt.tight_layout()
    plt.savefig("data/08_reporting/regresion/linear_regression_scatter.png")
    plt.close()

    # --- Histograma de errores (residuos)
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, bins=50, kde=True, color="cornflowerblue")
    plt.axvline(0, color="red", linestyle="--", lw=2)
    plt.xlabel("Error (valor real - predicho)")
    plt.ylabel("Frecuencia")
    plt.title("Distribución de errores – Linear Regression")
    plt.tight_layout()
    plt.savefig("data/08_reporting/regresion/linear_regression_residuals.png")
    plt.close()

    return results

def train_xgb_regressor(df: pd.DataFrame, xgb_regressor_model_path: str) -> dict:
    """
    Entrena y evalúa un modelo XGBoost Regressor para predecir el gasto total (Monetary).
    Guarda el modelo, resultados y gráficos.
    """

    # =========================================================
    # 1️⃣ Variables dependiente e independientes
    # =========================================================
    X = df[["AmountZScoreByLocation", "TimeSinceLastTxn", "IsLateNight", "IsWeekend"]]
    y = df["Monetary"]

    # =========================================================
    # 2️⃣ División Train/Test
    # =========================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # =========================================================
    # 3️⃣ Modelo + Cross Validation
    # =========================================================
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2 = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")
    cv_rmse = np.sqrt(
        -cross_val_score(model, X_train, y_train, cv=cv, scoring="neg_mean_squared_error")
    )

    # =========================================================
    # 4️⃣ Entrenamiento y evaluación final
    # =========================================================
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    results = {
        "model_name": "XGBRegressor",
        "r2_mean_cv": round(cv_r2.mean(), 4),
        "r2_std_cv": round(cv_r2.std(), 4),
        "r2_test": round(r2, 4),
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
    }

    # =========================================================
    # 5️⃣ Importancia de variables
    # =========================================================
    importances = pd.DataFrame({
        "Variable": X.columns,
        "Importancia": model.feature_importances_.round(4)
    }).sort_values("Importancia", ascending=False)

    results["feature_importances"] = importances.to_dict(orient="records")

    # =========================================================
    # 6️⃣ Guardar modelo entrenado
    # =========================================================
    joblib.dump(model, xgb_regressor_model_path)

    # =========================================================
    # 7️⃣ Generar gráficos y guardarlos
    # =========================================================
    os.makedirs("data/08_reporting/regresion", exist_ok=True)

    # --- Scatterplot: reales vs predichos
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    plt.xlabel("Valores reales (Monetary)")
    plt.ylabel("Predicciones (XGBoost)")
    plt.title("Valores reales vs predichos – XGBoost Regressor")
    plt.tight_layout()
    plt.savefig("data/08_reporting/regresion/xgb_regressor_scatter.png")
    plt.close()

    # --- Histograma de errores
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, bins=50, kde=True, color="darkorange")
    plt.axvline(0, color="red", linestyle="--", lw=2)
    plt.xlabel("Error (valor real - predicho)")
    plt.ylabel("Frecuencia")
    plt.title("Distribución de errores – XGBoost Regressor")
    plt.tight_layout()
    plt.savefig("data/08_reporting/regresion/xgb_regressor_residuals.png")
    plt.close()

    # --- Importancia de variables
    importances.sort_values("Importancia", ascending=True).plot(
        kind="barh", x="Variable", y="Importancia", color="darkorange", figsize=(6, 3)
    )
    plt.title("Importancia de variables – XGBoost Regressor")
    plt.xlabel("Importancia relativa")
    plt.tight_layout()
    plt.savefig("data/08_reporting/regresion/xgb_regressor_importances.png")
    plt.close()

    return results

def train_ridge_regressor(df: pd.DataFrame, ridge_model_path: str) -> dict:
    """
    Entrena un modelo Ridge Regression para predecir el gasto total (Monetary).
    Guarda el modelo, resultados y gráficos.
    """

    # =========================================================
    # 1️⃣ Variables dependiente e independientes
    # =========================================================
    X = df[["AmountZScoreByLocation", "TimeSinceLastTxn", "IsLateNight", "IsWeekend"]]
    y = df["Monetary"]

    # =========================================================
    # 2️⃣ División Train/Test
    # =========================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # =========================================================
    # 3️⃣ Escalamiento
    # =========================================================
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # =========================================================
    # 4️⃣ Modelo Ridge + Cross Validation
    # =========================================================
    model = Ridge(alpha=1.0, random_state=42)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_r2 = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="r2")
    cv_rmse = np.sqrt(
        -cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="neg_mean_squared_error")
    )

    # Entrenamiento final
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # =========================================================
    # 5️⃣ Evaluación final
    # =========================================================
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    results = {
        "model_name": "RidgeRegression",
        "r2_mean_cv": round(cv_r2.mean(), 4),
        "r2_std_cv": round(cv_r2.std(), 4),
        "r2_test": round(r2, 4),
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
    }

    # =========================================================
    # 6️⃣ Coeficientes del modelo
    # =========================================================
    coef_df = pd.DataFrame({
        "Variable": X.columns,
        "Coeficiente": model.coef_.round(4)
    })
    results["coefficients"] = coef_df.to_dict(orient="records")

    # =========================================================
    # 7️⃣ Guardar modelo entrenado
    # =========================================================
    joblib.dump(model, ridge_model_path)

    # =========================================================
    # 8️⃣ Gráficos
    # =========================================================
    os.makedirs("data/08_reporting/regresion", exist_ok=True)

    # --- Scatterplot: Reales vs Predichos
    plt.figure(figsize=(6,6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Valores reales (Monetary)")
    plt.ylabel("Predicciones (Ridge)")
    plt.title("Valores reales vs predichos – Ridge Regression")
    plt.tight_layout()
    plt.savefig("data/08_reporting/regresion/ridge_regressor_scatter.png")
    plt.close()

    # --- Histograma de errores
    residuals = y_test - y_pred
    plt.figure(figsize=(6,4))
    sns.histplot(residuals, bins=50, kde=True, color="steelblue")
    plt.axvline(0, color="red", linestyle="--", lw=2)
    plt.title("Distribución de errores – Ridge Regression")
    plt.xlabel("Error (valor real - predicho)")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig("data/08_reporting/regresion/ridge_regressor_residuals.png")
    plt.close()

    return results


def train_rf_regressor(df: pd.DataFrame, rf_regressor_model_path: str) -> dict:
    """
    Entrena y evalúa un modelo Random Forest Regressor para predecir el gasto total (Monetary).
    Guarda el modelo, resultados y gráficos.
    """

    # =========================================================
    # 1️⃣ Variables dependiente e independientes
    # =========================================================
    X = df[["AmountZScoreByLocation", "TimeSinceLastTxn", "IsLateNight", "IsWeekend"]]
    y = df["Monetary"]

    # =========================================================
    # 2️⃣ División Train/Test
    # =========================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # =========================================================
    # 3️⃣ Modelo Random Forest + Cross Validation
    # =========================================================
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_r2 = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")
    cv_rmse = np.sqrt(
        -cross_val_score(model, X_train, y_train, cv=cv, scoring="neg_mean_squared_error")
    )

    # Entrenamiento final
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # =========================================================
    # 4️⃣ Evaluación del modelo
    # =========================================================
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    results = {
        "model_name": "RandomForestRegressor",
        "r2_mean_cv": round(cv_r2.mean(), 4),
        "r2_std_cv": round(cv_r2.std(), 4),
        "r2_test": round(r2, 4),
        "rmse": round(rmse, 2),
        "mae": round(mae, 2)
    }

    # =========================================================
    # 5️⃣ Importancia de variables
    # =========================================================
    importances = pd.DataFrame({
        "Variable": X.columns,
        "Importancia": model.feature_importances_.round(4)
    }).sort_values("Importancia", ascending=False)

    results["feature_importances"] = importances.to_dict(orient="records")

    # =========================================================
    # 6️⃣ Guardar modelo entrenado
    # =========================================================
    joblib.dump(model, rf_regressor_model_path)

    # =========================================================
    # 7️⃣ Generar gráficos
    # =========================================================
    os.makedirs("data/08_reporting/regresion", exist_ok=True)

    # --- Scatterplot: reales vs predichos
    plt.figure(figsize=(6,6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Valores reales (Monetary)")
    plt.ylabel("Predicciones (Random Forest)")
    plt.title("Valores reales vs. predichos – Random Forest Regressor")
    plt.tight_layout()
    plt.savefig("data/08_reporting/regresion/rf_regressor_scatter.png")
    plt.close()

    # --- Histograma de errores
    residuals = y_test - y_pred
    plt.figure(figsize=(8,5))
    sns.histplot(residuals, bins=50, kde=True, color="seagreen")
    plt.axvline(0, color="red", linestyle="--", lw=2)
    plt.xlabel("Error (valor real - predicho)")
    plt.ylabel("Frecuencia")
    plt.title("Distribución de errores – Random Forest Regressor")
    plt.tight_layout()
    plt.savefig("data/08_reporting/regresion/rf_regressor_residuals.png")
    plt.close()

    # --- Importancia de variables
    importances.sort_values("Importancia", ascending=True).plot(
        kind="barh", x="Variable", y="Importancia", color="seagreen", figsize=(6,3)
    )
    plt.title("Importancia de variables – Random Forest Regressor")
    plt.xlabel("Importancia relativa")
    plt.tight_layout()
    plt.savefig("data/08_reporting/regresion/rf_regressor_importances.png")
    plt.close()

    return results

def train_lasso_regressor(df: pd.DataFrame, lasso_model_path: str) -> dict:
    """
    Entrena y evalúa un modelo Lasso Regression para predecir el gasto total (Monetary).
    Guarda el modelo, los resultados y gráficos de rendimiento.
    """

    # =========================================================
    # 1️⃣ Variables dependiente e independientes
    # =========================================================
    X = df[["AmountZScoreByLocation", "TimeSinceLastTxn", "IsLateNight", "IsWeekend"]]
    y = df["Monetary"]

    # =========================================================
    # 2️⃣ División Train/Test
    # =========================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # =========================================================
    # 3️⃣ Escalamiento
    # =========================================================
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # =========================================================
    # 4️⃣ Modelo Lasso + Cross Validation
    # =========================================================
    model = Lasso(alpha=0.01, random_state=42, max_iter=10000)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_r2 = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="r2")
    cv_rmse = np.sqrt(
        -cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="neg_mean_squared_error")
    )

    # Entrenamiento final
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # =========================================================
    # 5️⃣ Evaluación final
    # =========================================================
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    results = {
        "model_name": "LassoRegression",
        "r2_mean_cv": round(cv_r2.mean(), 4),
        "r2_std_cv": round(cv_r2.std(), 4),
        "r2_test": round(r2, 4),
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
    }

    # =========================================================
    # 6️⃣ Coeficientes del modelo
    # =========================================================
    coef_df = pd.DataFrame({
        "Variable": X.columns,
        "Coeficiente": model.coef_.round(4)
    })
    results["coefficients"] = coef_df.to_dict(orient="records")

    # =========================================================
    # 7️⃣ Guardar modelo entrenado
    # =========================================================
    joblib.dump(model, lasso_model_path)

    # =========================================================
    # 8️⃣ Generar gráficos
    # =========================================================
    os.makedirs("data/08_reporting/regresion", exist_ok=True)

    # --- Scatterplot: reales vs predichos
    plt.figure(figsize=(6,6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Valores reales (Monetary)")
    plt.ylabel("Predicciones (Lasso)")
    plt.title("Comparación entre valores reales y predichos – Lasso Regression")
    plt.tight_layout()
    plt.savefig("data/08_reporting/regresion/lasso_regressor_scatter.png")
    plt.close()

    # --- Histograma de errores
    errores = y_test - y_pred
    plt.figure(figsize=(8,5))
    sns.histplot(errores, bins=50, kde=True)
    plt.axvline(0, color='red', linestyle='--', label="Error = 0")
    plt.xlabel("Error (valor real - predicho)")
    plt.ylabel("Frecuencia")
    plt.title("Distribución de errores – Lasso Regression")
    plt.legend()
    plt.tight_layout()
    plt.savefig("data/08_reporting/regresion/lasso_regressor_residuals.png")
    plt.close()

    return results