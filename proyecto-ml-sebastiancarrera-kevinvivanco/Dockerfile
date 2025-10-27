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

# 5️⃣ Instalar herramientas base y actualizar pip
RUN apt-get update && apt-get install -y build-essential libffi-dev libssl-dev && \
    pip install --upgrade pip setuptools wheel && \
    rm -rf /var/lib/apt/lists/*

# 6️⃣ Instalar dependencias del entorno Docker (versión ligera y estable)
RUN pip install --no-cache-dir -r requirements_docker.txt

# 7️⃣ Inicializar Airflow (DB y usuario admin)
RUN airflow db init && \
    airflow users create \
        --username admin \
        --firstname Kevin \
        --lastname Vivanco \
        --role Admin \
        --email kevin@example.com \
        --password 1234

# 8️⃣ Crear carpeta para DAGs y copiar los archivos
RUN mkdir -p /usr/local/airflow/dags
COPY dags/* /usr/local/airflow/dags/

# 9️⃣ Exponer puertos (Airflow Web UI)
EXPOSE 8080

# 🔟 Comando por defecto: iniciar Airflow webserver + scheduler
CMD ["bash", "-c", "airflow webserver -p 8080 & airflow scheduler"]
