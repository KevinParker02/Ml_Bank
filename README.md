# Ml_Bank (Machine Learning - Bank Customer Transaction Analysis)

## 🚀 Iniciar el Proyecto en Visual Studio Code  

### 🔹 1. Clonar el repositorio
```bash
git clone https://github.com/tu-usuario/Ml_Bank.git
cd Ml_Bank/proyecto-ml-sebastiancarrera-kevinvivanco

# Crear entorno virtual
python -m venv venv

# Activar en Windows PowerShell
.\venv\Scripts\activate

# Activar en Linux / MacOS
source venv/bin/activate

# Instalamos las dependencias
pip install -r requirements.txt

# DEBEMOS ingresar a la carpeta del proyecto
cd proyecto-ml-sebastiancarrera-kevinvivanco

# Verificamos la instalación de Kedro
kedro info

# Ejecutar los pipelines
kedro run

# Abrir los notebooks en VS Code
jupyter notebook

# Visualizar los pipelines
kedro viz
```

---

# 📊 Fase 1 – Comprensión del Negocio

Proyecto de Machine Learning con **Kedro** – Evaluación Parcial 1  
Asignatura: *Machine Learning (MLY0100)* 
Dataset: https://www.kaggle.com/datasets/tanmayjune/bank-customer-transaction-analysis?select=bank_customer_transactions.csv  

---

## 🏦 Contexto del Proyecto

Una entidad bancaria, proveniente de la India, busca **mejorar la toma de decisiones estratégicas** mediante el uso de sus datos históricos de clientes y transacciones.  
Actualmente, el banco cuenta con registros de **clientes, transacciones y perfiles de valor**, pero carece de un sistema predictivo robusto para:

- Detectar **transacciones fraudulentas**.  
- Identificar **clientes de alto valor** para retención.  
- Segmentar clientes según su **recencia, frecuencia y valor monetario (RFM)**.  
- Estimar el **valor futuro de las transacciones** para planificación financiera.  

Esto genera **riesgos financieros** y pérdida de **oportunidades de negocio**.

---

## 🎯 Objetivos del Proyecto

- **General**: aprovechar datos históricos para construir un sistema de análisis predictivo y descriptivo con técnicas de Machine Learning.  

- **Específicos**:  
  - Diseñar un modelo de **clasificación** que anticipe si una transacción será fraudulenta.  
  - Construir un modelo de **regresión** para predecir el monto promedio futuro de transacciones por cliente.  
  - Aplicar segmentación basada en **RFM (Recency, Frequency, Monetary)** para agrupar clientes en perfiles de valor.  

---

## 📂 Datasets Seleccionados

| Archivo                 | Descripción |
|--------------------------|-------------|
| **cleaned_dataset.csv** | Transacciones detalladas de clientes, con variables demográficas, saldos y montos ya depurados. |
| **customer_agg.csv**    | Información agregada por cliente: gasto total, frecuencia, recencia y métricas financieras promedio. |
| **RFM.csv**             | Segmentación de clientes mediante Recency, Frequency, Monetary; incluye puntuaciones y categorías de cliente. |

> Estos datasets se encuentran en la carpeta `data/01_raw/` y están configurados en el `catalog.yml` de Kedro.

---

## 📌 Situación Actual

- El banco posee **grandes volúmenes de datos**, pero no los utiliza en un proceso automatizado.  
- El análisis actual es **manual y reactivo**, dificultando la detección temprana de fraude.  
- No existen modelos predictivos que permitan **anticipar riesgos** ni proyectar el **valor futuro de clientes**.  

---

## 🧠 Objetivos de Machine Learning

- **Clasificación** → Determinar si una transacción es **fraudulenta o legítima** (`cleaned_dataset`).  
- **Regresión** → Estimar el **monto promedio futuro** de transacciones por cliente (`customer_agg`).  
- **Segmentación (RFM)** → Generar perfiles de clientes (`RFM`) para estrategias de retención y marketing.  

---

## 🗺️ Plan del Proyecto (CRISP-DM)

| Semana | Actividad Principal | Entregable |
|--------|---------------------|------------|
| **1**  | Comprensión del negocio | Notebook `01_business_understanding.ipynb` |
| **2**  | Análisis exploratorio de datos (EDA) | Notebook `02_data_understanding.ipynb` |
| **3**  | Preparación y Limpieza de los datos | Notebook `03_data_preparation.ipynb` |
| **4**  | Modelado y entrega final | Repositorio GitHub con README y pipelines Kedro |
| **5**  | Documentación y entrega final | Repositorio GitHub con README y pipelines Kedro |

---

## 👥 Equipo de Trabajo

- 🧑‍💻 **Sebastián Carrera**  
- 🧑‍💻 **Kevin Vivanco**  

---

## ✅ Resultado Esperado

Un proyecto reproducible en **Kedro**, con pipelines organizados según CRISP-DM, que permita:  
- Detectar transacciones fraudulentas.  
- Predecir el valor futuro de clientes.  
- Segmentar perfiles estratégicos para decisiones de negocio.  
- Generar visualizaciones claras para la **toma de decisiones gerenciales**.  

---
