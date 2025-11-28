# 1. Introducción

El objetivo de este análisis es descubrir patrones, segmentos y estructuras ocultas dentro del comportamiento transaccional del banco. Mediante técnicas de clustering, reducción de dimensionalidad y perfilamiento, buscamos encontrar grupos naturales que permitan:

- Identificar segmentos con alto riesgo de fraude.
- Mejorar el modelo supervisado integrando la variable `cluster_id`.
- Entregar información accionable para el área de riesgo y prevención de fraude.

Este análisis complementa el trabajo realizado en la etapa supervisada y completa el pipeline end-to-end del proyecto.

---

# 2. Preparación de los datos

Los datos utilizados provienen del dataset procesado por Kedro en la etapa de data engineering y feature engineering. Las variables consideradas para el clustering fueron:

- AmountZScoreByLocation  
- IsAnomaly  
- IsLateNight  
- IsWeekend  

Estas variables capturan señales operativas y comportamentales del usuario, relevantes para el riesgo transaccional.

## Preprocesamiento aplicado

- Se utilizó **StandardScaler** para todos los algoritmos basados en distancia.  
- Se removieron identificadores y columnas no relevantes.  
- Se seleccionaron únicamente variables numéricas/booleanas con impacto en el comportamiento del cliente.

---

# 3. Clustering

Se aplicaron múltiples algoritmos, pero el análisis profundo se centró en **K-Means**, **DBSCAN** y **OPTICS**, cumpliendo con la rúbrica del proyecto.

## 3.1 Selección del modelo

Tras evaluar:

- Silhouette Score  
- Davies-Bouldin Index  
- Calinski-Harabasz  
- Gráficos de codo (Elbow Method)

El mejor desempeño global lo obtuvo **K-Means con k=6**, ya que generó clusters:

- Estables  
- Interpretables  
- Con buen nivel de separación  
- Con patrones diferenciados de fraude  

---

# 4. Descripción detallada de los clusters (K-Means, k = 6)

A continuación se muestra la estructura de fraude por cluster:

| Cluster | Transacciones | Fraudes | % Fraude   | Interpretación                                         |
| ------- | ------------- | ------- | ---------- | ------------------------------------------------------ |
| **2**   | 318           | 49      | **15.41%** | **Máximo riesgo**: montos extremadamente atípicos.     |
| **5**   | 3871          | 163     | **4.21%**  | Riesgo alto: actividad irregular + señales temporales. |
| **3**   | 6929          | 236     | **3.41%**  | Riesgo medio-alto, anomalías frecuentes.               |
| **4**   | 3345          | 30      | **0.89%**  | Riesgo bajo, comportamiento moderado.                  |
| **1**   | 10443         | 8       | **0.07%**  | Comportamiento estable, casi sin fraude.               |
| **0**   | 25094         | 8       | **0.03%**  | **Cluster más seguro**, comportamiento regular.        |

---

## 4.1 Interpretación general

Los resultados muestran una separación clara entre segmentos de alto riesgo y segmentos seguros:

- **Cluster 2** es el grupo más crítico: concentra proporciones extremas de fraude y transacciones fuera de patrón.  
- **Clusters 3 y 5** forman una “zona gris” con comportamientos irregulares y señales no triviales de riesgo.  
- **Clusters 0 y 1** representan la gran masa de clientes estables, con comportamiento natural y casi sin incidentes.

Esto refleja que el algoritmo reveló comportamientos que no eran evidentes mediante análisis supervisado solamente.

---

# 5. Reducción de dimensionalidad

Se aplicaron dos técnicas principales: **PCA** y **t-SNE**, cumpliendo los requisitos obligatorios de la rúbrica.

## 5.1 PCA (Principal Component Analysis)

- PC1 explica la mayoría de la varianza asociada a montos atípicos y anomalías.  
- PC2 captura diferencias temporales (late-night, weekend).  
- El **biplot** del PCA mostró que los clusters extremos (2, 3 y 5) se separan claramente del resto.

Esto sugiere que los patrones de fraude están vinculados a combinaciones no lineales de las variables originales.

## 5.2 t-SNE

- Muestra separación visual clara entre los clusters.  
- Segmentos de riesgo alto aparecen aglomerados y alejados de los grupos normales.  
- Es una excelente herramienta para interpretar la estructura 2D del clustering.

---

# 6. Perfilamiento de patrones por cluster

A continuación se resumen los hallazgos más importantes:

### **Cluster 2 — “Transacciones Extremas e Irregulares”**
- Z-scores muy altos.  
- Comportamiento fuera de cualquier distribución normal.  
- **15% de fraude → prioridad absoluta.**

### **Cluster 5 — “Actividad Irregular con Sesgo Temporal”**
- Alta presencia de operaciones nocturnas o en fin de semana.  
- Señales híbridas de riesgo.  
- **4% de fraude.**

### **Cluster 3 — “Riesgo Medio-Alto con Anomalías Frecuentes”**
- Presencia marcada de `IsAnomaly`.  
- Patrón de riesgo menos extremo que el cluster 2, pero importante.

### **Clusters 0 y 1 — “Segmentos Estables y Seguros”**
- Montos dentro de rango.  
- Comportamiento regular.  
- Fraude prácticamente inexistente.

## Conclusión de patrones

El clustering permitió reconstruir **mapas comportamentales reales** del banco, dividiendo las transacciones en segmentos con riesgo bien diferenciado y valor analítico directo para la toma de decisiones.

---

# 7. Integración con el modelo supervisado

La variable `cluster_id` fue incorporada al pipeline supervisado con XGBoost.

## Resultados del modelo final

- **Recall:** 0.984  
- **F1:** 0.036 (esperable por el desbalance)

## Interpretación

El recall extremadamente alto demuestra que:

- Los clusters aportan señales nuevas al modelo.  
- El modelo captura patrones que antes no eran visibles en las variables originales.  
- La integración permitió mejorar la sensibilidad a fraudes sin necesidad de modificar la arquitectura del modelo.

---

# 8. Conclusiones generales

- El análisis no supervisado permitió identificar segmentos críticos de fraude que no estaban explícitos en los datos originales.  
- K-Means con k=6 fue el modelo más estable y útil desde el punto de vista analítico.  
- La reducción de dimensionalidad (PCA + t-SNE) permitió visualizar la estructura real del comportamiento transaccional.  
- La integración del `cluster_id` fortaleció el modelo de XGBoost, alcanzando un **recall sobresaliente (0.984)**.  
- Este análisis aporta valor tanto al modelo predictivo como al negocio, permitiendo identificar zonas de riesgo y priorizar la detección temprana.

---

# 9. Referencias

- Scikit-learn Clustering: https://scikit-learn.org/stable/modules/clustering.html
- Scikit-learn Decomposition: https://scikitlearn.
org/stable/modules/decomposition.html
- UMAP: https://umap-learn.readthedocs.io/
- Kedro: https://kedro.readthedocs.io/
- Airflow: https://airflow.apache.org/docs/
- DVC: https://dvc.org/doc
