# Ml_Bank (Machine Learning - Bank Customer Transaction Analysis)

# 📊 Fase 1 – Comprensión del Negocio

Proyecto de Machine Learning con **Kedro** – Evaluación Parcial 1  
Asignatura: *Machine Learning (MLY0100)*  

---

## 🏦 Contexto del Proyecto

Una entidad bancaria busca **mejorar la toma de decisiones estratégicas** mediante el uso de sus datos históricos de clientes y transacciones.  
Actualmente, el banco cuenta con registros de **clientes, transacciones y fraudes**, pero carece de un sistema predictivo robusto para:

- Detectar **transacciones fraudulentas**.  
- Identificar **clientes de alto valor** para retención.  
- Estimar el **valor futuro de las transacciones** para planificación financiera.  

Esto genera **riesgos financieros** y pérdida de **oportunidades de negocio**.

---

## 🎯 Objetivos del Proyecto

- **General**: aprovechar datos históricos para construir un sistema de análisis predictivo y descriptivo con técnicas de Machine Learning.  

- **Específicos**:  
  - Diseñar un modelo de **clasificación** que anticipe si una transacción será fraudulenta.  
  - Construir un modelo de **regresión** para predecir el monto promedio futuro de transacciones por cliente.  
  - (Opcional) Aplicar segmentación basada en **RFM (Recency, Frequency, Monetary)** para agrupar clientes en perfiles de valor.  

---

## 📂 Datasets Seleccionados

| Archivo                          | Descripción |
|---------------------------------|-------------|
| **customers.csv**               | Información demográfica y de perfil de clientes. |
| **bank_customer_transactions.csv** | Historial detallado de transacciones realizadas. |
| **fraud_dataset.csv**           | Registros de operaciones con etiquetas de fraude. |

> Estos datasets se encuentran en la carpeta `data/01_raw/` y están configurados en el `catalog.yml` de Kedro.

---

## 📌 Situación Actual

- El banco posee **grandes volúmenes de datos** pero no los utiliza en un proceso automatizado.  
- El análisis actual es **manual y reactivo**, dificultando la detección temprana de fraude.  
- No existen modelos predictivos que permitan **anticipar riesgos** ni proyectar el **valor futuro de clientes**.  

---

## 🧠 Objetivos de Machine Learning

- **Clasificación** → Determinar si una transacción es **fraudulenta o legítima**.  
- **Regresión** → Estimar el **monto promedio futuro** de transacciones por cliente.  
- **Segmentación (opcional)** → Generar perfiles de clientes usando métricas **RFM** para estrategias de marketing.  

---

## 🗺️ Plan del Proyecto (CRISP-DM)

| Semana | Actividad Principal | Entregable |
|--------|---------------------|------------|
| **1**  | Comprensión del negocio y selección de datasets | Notebook `01_business_understanding.ipynb` |
| **2**  | Análisis exploratorio de datos (EDA) | Notebook `02_data_understanding.ipynb` |
| **3**  | Limpieza y feature engineering | Notebook `03_data_preparation.ipynb` |
| **4**  | Documentación y entrega final | Repositorio GitHub con README y pipelines Kedro |

---

## 👥 Equipo de Trabajo

- 🧑‍💻 **Sebatián Carrera** 
- 🧑‍💻 **Kevin Vivanco** 

---

## ✅ Resultado Esperado

Un proyecto reproducible en **Kedro**, con pipelines organizados según CRISP-DM, que permita:  
- Detectar transacciones fraudulentas.  
- Predecir el valor de clientes.  
- Segmentar perfiles para estrategias de negocio.  

---

