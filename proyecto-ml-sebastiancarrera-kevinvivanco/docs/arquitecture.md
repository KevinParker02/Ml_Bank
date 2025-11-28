# Arquitectura del Proyecto de Machine Learning  
**Proyecto:** Detección de fraude bancario y predicción de Monetary
**Autor/es:** Sebastián Carrera y Kevin Vivanco  
**Fecha:** 27/11/2025  

---

## 1. Descripción General de la Arquitectura

Este proyecto utiliza una arquitectura modular basada en **Kedro**, permitiendo un flujo de datos claro, reproducible y escalable. El pipeline integra:

- **Preparación de datos**
- **Ingeniería de características**
- **Modelos supervisados (Clasificación y Regresión)**
- **Modelos no supervisados (Clustering)**
- **Reducción dimensional**
- **Reentrenamiento del modelo con nuevas features**

El objetivo principal es **detectar fraude bancario**, identificando patrones anómalos en transacciones y evaluando cómo la variable `cluster_id`, generada mediante
técnicas de clustering, mejora el desempeño del modelo final.

El enfoque combina métodos de clustering (DBSCAN, OPTICS, K-Means) con reducción dimensional (PCA y t-SNE) para comprender mejor la estructura interna del dataset 
y crear características adicionales para los modelos supervisados.

---

## 2. Componentes Principales

### 2.1. Estructura de Carpetas (Kedro)
├── data/
│ ├── 01_raw/
│ ├── 02_intermediate/
│ ├── 03_primary/
│ ├── 04_feature/
│ ├── 05_model_input/
│ ├── 06_models/
│ ├── 07_model_output/
│ └── 08_reporting/
├── docs/
├── notebooks/
├── src/
│ ├── pipelines/
│ ├── utils/
│ └── project/
└── conf/

---

## 2.2. Flujo del Pipeline

### **1. Ingesta de Datos**
- Lectura de transacciones desde `data/01_raw/`.
- Limpieza, validación e identificación de outliers.

---

### **2. Feature Engineering**
Generación de características para:

#### ✔ Clasificación (`Features_training_v1`)
Incluye variables como:
- `AmountZScoreByLocation`
- `IsAnomaly`
- `IsLateNight`
- `IsWeekend`
- Indicadores temporales y de comportamiento.

#### ✔ Regresión (`Features_training_v2`)
Variables ajustadas para evaluación de montos o patrones continuos.

Estas características resumen comportamiento del cliente, anomalías y patrones temporales relevantes para el fraude.

---

### **3. Entrenamiento de Modelos Supervisados**

#### **Clasificación (detección de fraude)**
Modelos probados:
- DecisionTreeClassifier  
- KNeighborsClassifier  
- MLPClassifier  
- RandomForestClassifier  
- XGBClassifier (**modelo final**)  

#### **Regresión (Predicción de variable Monetary)**
Modelos utilizados:
- Lasso Regression  
- Linear Regression  
- RandomForestRegressor  
- Ridge Regression  
- XGBRegressor  

Cada modelo es evaluado mediante métricas correspondientes a su tipo (F1, AUC, MAE, RMSE, etc.).

---

### **4. Clustering No Supervisado y Reentrenamiento del Modelo**

Modelos aplicados:

- **DBSCAN** → detección de densidades homogéneas y ruido.
- **OPTICS** → alternativa más flexible ante densidades variables.
- **K-Means + t-SNE** → clustering sobre espacio no lineal de alta complejidad.
- **Clustering jerárquico** → dendrogramas en PCA/t-SNE como análisis estructural.

#### ✔ *Punto clave:*  
Se genera el dataset **`Features_clustering_v1`**, incorporando la variable nueva:

- `cluster_id`

Luego, se **reentrena XGBClassifier** con esta nueva característica para mejorar la capacidad de detectar patrones anómalos.

---

### **5. Evaluación y Reporte de Modelos**

Se evalúa el impacto de `cluster_id` en:
- Recall de fraude  
- AUC  
- F1  
- Interpretabilidad del modelo  


---

## 3. Diagrama de Flujo de Arquitectura
Raw Data
↓
Preprocesamiento
↓
Feature Engineering
↓
Entrenamiento de Modelos
↓
Reducción Dimensional (PCA / t-SNE) para clustering
↓
Modelos No Supervisados (DBSCAN / OPTICS / K-Means)
↓
Generación de cluster_id
↓
Reentrenamiento XGBClassifier
↓
Evaluación y Reportes

---

## 4. Herramientas y Librerías

- **Kedro**  
- **Scikit-Learn**  
- **Pandas / NumPy**  
- **Seaborn / Matplotlib**  
- **Plotly (visualización interactiva)**  
- **SciPy**

---

## 5. Consideraciones de Despliegue

- El pipeline está diseñado para ser **100% reproducible**.  
- Los modelos no supervisados permiten incorporar patrones nuevos sin necesidad de re-etiquetar manualmente.  
- El sistema puede integrarse a un mecanismo de scoring o alertas de fraude en tiempo real.  
- El uso de Kedro facilita mantener versiones del pipeline, control de datos y despliegues más seguros.

---

## 6. Conclusión

La arquitectura utilizada permite una comprensión profunda del comportamiento transaccional y facilita el entrenamiento de modelos robustos para la detección de fraude. Al integrar clustering con aprendizaje supervisado, el proyecto logra identificar grupos atípicos y mejorar la capacidad del modelo final para reconocer transacciones sospechosas.  
Esta estructura modular y escalable facilita su uso en entornos reales donde se requiere un análisis continuo y actualización periódica del modelo.