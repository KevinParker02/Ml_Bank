# ================================
# Dockerfile - ML_Bank Kedro + Airflow + DVC
# ================================

# 1️⃣ Imagen base
FROM python:3.11-slim

# 2️⃣ Variables de entorno
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV AIRFLOW_HOME=/usr/local/airflow

# 3️⃣ Directorio de trabajo
WORKDIR /app

# 4️⃣ Copiar todo el proyecto
COPY . /app

# 5️⃣ Instalar dependencias del sistema
RUN apt-get update && \
    apt-get install -y build-essential git curl && \
    apt-get clean

# 6️⃣ Instalar dependencias de Python
RUN pip install --upgrade pip setuptools wheel \
    && pip install kedro==0.19.4 apache-airflow==2.7.1 dvc[all]==3.53.2 pendulum==2.1.2 Flask-Session==0.4.0 connexion==2.14.2 \
    && pip install scikit-learn==1.3.2 xgboost==2.1.0 pandas==2.1.4 numpy==1.26.4 matplotlib seaborn joblib

# 7️⃣ Inicializar Airflow (DB y usuario admin)
RUN airflow db init && \
    airflow users create \
        --username admin \
        --firstname Kevin \
        --lastname Vivanco \
        --role Admin \
        --email kevin@example.com \
        --password 1234

# 8️⃣ Exponer puertos de Airflow
EXPOSE 8080

# 9️⃣ Copiar DAGs al directorio de Airflow
RUN mkdir -p /usr/local/airflow/dags
COPY dags/* /usr/local/airflow/dags/


# 1️⃣0️⃣ Comando por defecto
CMD ["bash", "-c", "airflow webserver -p 8080 & airflow scheduler"]
