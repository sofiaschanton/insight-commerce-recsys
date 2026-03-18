<div align="center">

# Insight Commerce · RecSys

</div>

> **ML Consulting Services:** Solución integral de *Next-Basket Recommendation* con arquitectura MLOps desplegada en AWS.

<p align="center">
  <img src="reports/figures/logo.png" width="300" alt="Insight Commerce Logo">
  <br>
</p>


<p align="center">
  <img src="https://img.shields.io/badge/AWS-ECS%20Fargate-orange?style=flat-square&logo=amazonaws&logoColor=white" alt="AWS">
  <img src="https://img.shields.io/badge/FastAPI-Serving-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Model-LightGBM+Optuna-FFB300?style=flat-square" alt="LightGBM">
  <img src="https://img.shields.io/badge/MLFlow-Tracking-0194E2?style=flat-square&logo=mlflow&logoColor=white" alt="MLFlow">
  <img src="https://img.shields.io/badge/GitHub%20Actions-MLOps-2088FF?style=flat-square&logo=githubactions&logoColor=white" alt="GitHub Actions">
  <a href="https://streamlit.io/"><img src="https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit"></a>
</p>


---

## Executive Summary
Insight Commerce presenta un **Recommender System (RecSys)** de nivel empresarial diseñado para retail digital. Esta solución no es solo un modelo de predicción; es un ecosistema de **MLOps productivo** que mitiga la degradación del modelo mediante monitoreo estadístico, automatiza el reentrenamiento con **Optuna** y garantiza despliegues continuos sobre **AWS**.

--- 

## Business Impact & Strategy
Insight Commerce no entrega modelos estáticos; entregamos **activos digitales resilientes**. Nuestra arquitectura está diseñada para maximizar el ROI del retail mediante tres pilares estratégicos:

* **Mitigación del Riesgo Operativo:** Gracias a la detección activa de **Data Drift (PSI/KS)**, el sistema identifica cambios en los hábitos de consumo antes de que afecten la conversión. Esto elimina la "degradación silenciosa" del modelo, asegurando que las recomendaciones sigan siendo relevantes mes a mes.
* **Eficiencia de Costos y Escalabilidad:** El uso de una arquitectura **Serverless (AWS Fargate)** permite que el motor de recomendación escale automáticamente con los picos de tráfico (ej. Black Friday) sin necesidad de mantener servidores costosos subutilizados durante periodos valle.
* **Hiper-Personalización Basada en Evidencia:** El análisis de bimodalidad temporal (ciclos de 7 y 30 días) identificado en nuestro [Informe de EDA](./docs/Informe_EDA.md) permite al modelo actuar en el momento exacto de necesidad del cliente, aumentando el *Ticket Promedio* y la tasa de retención.
* **Continuidad de Servicio (Zero Downtime):** La lógica de **Inferencia Dual** garantiza que incluso los usuarios nuevos (Cold-Start) reciban una experiencia personalizada desde el segundo cero, eliminando la pérdida de clientes por falta de datos históricos.

--- 

##  Arquitectura y Stack Tecnológico
<div align="center">

| Categoría | Herramientas / Tecnologías |
| :--- | :--- |
| **Lenguaje** | Python 3.10 |
| **API** | FastAPI + Uvicorn |
| **Training** | LightGBM + Optuna |
| **Data** | pandas, NumPy, Parquet |
| **Cloud** | AWS S3, AWS RDS, ECS Fargate, IAM |
| **CI/CD** | GitHub Actions |
| **Tracking** | MLflow |
</div>

Diseñamos infraestructuras desacopladas para garantizar que la inteligencia del negocio nunca se detenga:

* **Serving:** API en **FastAPI** desplegada en **AWS ECS Fargate** carga modelos dinámicamente desde **Amazon S3** al iniciar el contenedor.
* **Data Layer:** **AWS RDS (PostgreSQL)** para almacenamiento transaccional.(Ver [Diccionario de Datos](./docs/data_dictionary.md)).
* **Artifacts:** **Amazon S3** para modelos y matrices de entrenamiento.
* **Experiment Tracking:** Gestión de experimentos con **MLFlow** para total trazabilidad. ([Guía MLFlow](./docs/MLFLOW_SETUP.md)).
* **Monitoreo:** Detección de *Drift* mediante **PSI** y **KS-test** ([Ver monitoreo](./src/model_monitoring.py)).




---


## Índice
1. [Executive Summary](#executive-summary)
2. [Business Impact](#bussines-impact)
3. [Arquitectura y Stack Tecnológico](#arquitectura-y-stack-tecnológico)
4. [Hallazgos del EDA](#hallazgos-críticos-del-eda)
5. [Lógica de Inferencia](#lógica-de-inferencia-modo-dual)
6. [Ciclo MLOps](#ciclo-mlops-y-gitflow)
7. [Dashboards de Visualización (Streamlit)](#dashboards-de-visualización-streamlit)
8. [Instalación](#instalación)
9. [Variables de entorno](#variables-de-entorno)
10. [Estructura del proyecto](#estructura-del-proyecto)
11. [Pipeline de ejecución](#pipeline-de-ejecucion)
12. [Base de datos](#base-de-datos)
13. [Modelo](#modelo)
14. [MLOps Lifecycle](#mlops-lifecycle)
15. [Architecture Overview](#architecture-overview)
16. [Key Features](#key-features)
17. [Artifact Strategy](#artifact-strategy)
18. [Cloud Deployment](#cloud-deployment)
19. [Serving Layer](#serving-layer)
20. [Git Workflow -- Ramas y Pull Requests](#git-workflow----ramas-y-pull-requests)

---

## Hallazgos Críticos del EDA
Basado en nuestro [Informe de EDA](./docs/Informe_EDA.md):
* **Frecuencia:** Picos de reposición detectados a los **7 y 30 días**.
* **Filtros:** Solo entrenamos usuarios con $\geq$ 5 órdenes para asegurar señales estables.

---

## Lógica de Inferencia (Modo Dual)
El sistema garantiza una disponibilidad del 100% mediante dos estrategias en `src/api/inference.py`:
1.  **Modelo de Propensión:** Clasificador LightGBM para usuarios con historial (Umbral: $\geq$ 5 órdenes).
2.  **Cold-Start Global:** Fallback automático de **Top 10 Popularidad** para usuarios nuevos, asegurando que el cliente siempre reciba una recomendación relevante.

---

## Ciclo MLOps y GitFlow

### Monitoreo de Drift y Reentrenamiento
El sistema evalúa semanalmente la estabilidad de las features (Detalles en [Guía CI/CD](./docs/CI_CD.md)):
- **PSI < 0.10**: Estable.
- **PSI > 0.25**: Alerta de Drift detectada -> Dispara pipeline de reentrenamiento y actualización en S3.

### Workflow Profesional
- **GitFlow:** Desarrollo basado en ramas `feature/*` e integración en `develop`.
- **Calidad de Código:** PRs con revisión obligatoria y tests de cobertura.

---

## Dashboards de Visualización (Streamlit)
Ofrecemos transparencia total para cada área de la empresa:
* **`01_Top_10.py`**: Interfaz de cara al cliente final.
* **`02_Impacto_de_Negocio.py`**: Dashboard de KPIs para el Product Owner.
* **`03_Metricas_del_Modelo.py`**: Diagnóstico técnico para el equipo de Data Science.

## Dashboards de Visualización (Streamlit)

Ofrecemos transparencia total para cada área de la empresa:

* **`01_Top_10.py`**: Interfaz de cara al cliente final.
* **`02_Impacto_de_Negocio.py`**: Dashboard de KPIs para el Product Owner.
* **`03_Metricas_del_Modelo.py`**: Diagnóstico técnico para el equipo de Data Science.

### Cómo levantar la app

**Requisitos previos:**
- Archivo `.env` configurado con `API_URL` apuntando a la API activa
- `models/model_log.json` presente en la carpeta `models/`

**Verificar que la API está activa:**
```
http://18.217.26.71:8000/docs
```

**Levantar Streamlit:**
```bash
streamlit run app/streamlit_app.py
```

Las páginas de **Impacto de Negocio** y **Métricas del Modelo** funcionan sin conexión a la API — solo requieren `model_log.json`.


## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/sofiaschanton/insight-commerce-recsys.git
cd insight-commerce-recsys
```

### 2. Crear y activar un entorno virtual

```bash
# Crear entorno virtual con Python 3.10
py -3.10 -m venv venv

# Activar en Linux/macOS
source venv/bin/activate

# Activar en Windows
venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar las variables de entorno

Copia el archivo de ejemplo y editalo con tus datos:

```bash
cp .env.example .env
```

Luego abre `.env` y completa los valores segun tu entorno (ver seccion [Variables de entorno](#variables-de-entorno)).

### 5. Ejecutar el ETL

```bash
python src/data/etl_dimensional.py
```

---

## Variables de entorno

El proyecto utiliza un archivo `.env` en la raiz del proyecto para gestionar la configuracion sensible. **Este archivo nunca debe subirse al repositorio.**

### Ejemplo de `.env`

```env
# Base de datos — AWS RDS PostgreSQL
DB_HOST=
DB_PORT=5432
DB_NAME=
DB_USER=
DB_PASSWORD=
DB_SSLMODE=require

# Artefactos del modelo — S3
S3_BUCKET=insight-commerce-artifacts
S3_PREFIX=models
USE_S3=false

# Streamlit
API_URL=http://localhost:8000

# Configuracion del proyecto
RANDOM_SEED=42

```

### Descripcion de variables

| Variable | Descripcion | Valor por defecto |
|---|---|---|
| `DB_HOST` | Host de AWS RDS PostgreSQL | — |
| `DB_PORT` | Puerto PostgreSQL | `5432` |
| `DB_NAME` | Nombre de la base de datos | — |
| `DB_USER` | Usuario PostgreSQL | — |
| `DB_PASSWORD` | Contraseña PostgreSQL | — |
| `DB_SSLMODE` | Modo SSL | `require` |
| `S3_BUCKET` | Bucket S3 para artefactos | `insight-commerce-artifacts` |
| `S3_PREFIX` | Prefijo dentro del bucket | `models` |
| `USE_S3` | Cargar artefactos desde S3 | `false` (local) / `true` (Fargate) |
| `API_URL` | URL de la API para Streamlit | `http://localhost:8000` |
| `RANDOM_SEED` | Semilla aleatoria global | `42` |

> **Nunca compartas ni subas tu archivo `.env` a control de versiones.** Asegurate de que `.env` este incluido en tu `.gitignore`.

---

## Estructura del proyecto

Solo se muestran los archivos versionados. Los directorios `data/raw/`, `data/processed/`, `models/` y `venv/` estan excluidos por `.gitignore`.

```
insight-commerce-recsys/
|
+-- .github/
|   +-- PULL_REQUEST_TEMPLATE.md
|
+-- .vscode/
|   +-- settings.json
|
+-- app/
|   +-- streamlit_app.py            # Entry point Streamlit
|   +-- pages/
|       +-- 01_Top_10.py            # Recomendaciones personalizadas
|       +-- 02_Impacto_de_Negocio.py # Dashboard para el PO
|       +-- 03_Metricas_del_Modelo.py # Métricas técnicas del modelo
|
+-- data/
|   +-- samples/                    # Muestras pequenas para desarrollo
|   +-- local_database/
|       +-- InstaCart_DataBase_Creation.sql
|       +-- InstaCart_DataBase_Creation_Dimensional.sql
|
+-- docs/
|   +-- figures/
|   |   +-- erd_dimensional.png
|   +-- DATABASE_SETUP.md
|   +-- FEATURES.md
|   +-- SUPABASE_SETUP.md
|   +-- data_dictionary.md
|   +-- estrategia_validacion.md    # Split por filas vs GroupShuffleSplit (Sprint 2)
|
+-- notebooks/
|   +-- 01_eda.ipynb                # EDA sobre CSVs completos (base local)
|   +-- 01_eda_neon.ipynb           # EDA y validaciones V-1 a V-8 sobre AWS RDS
|   +-- 02_calidad_datos.ipynb
|   +-- 03_feature_engineering.ipynb  # Feature matrix: calculo, validacion y guardado del parquet
|   +-- 04_modelado.ipynb           # Modelado: LightGBM + CatBoost + Optuna (Sprint 2)
|
+-- reports/
|   +-- figures/                    # Graficos generados por notebooks y EDA
|   +-- reporte_calidad_datos.csv
|   +-- reporte_calidad_datos.html
|
+-- src/
|   +-- __init__.py
|   +-- api/
|   |   +-- __init__.py
|   |   +-- inference.py           # RecommendationService: features online + prediccion
|   |   +-- main.py                # FastAPI app: rutas, startup, manejo de errores
|   |   +-- schemas.py             # Pydantic schemas: request/response de cada endpoint
|   +-- data/
|   |   +-- __init__.py
|   |   +-- data_ingestation.py
|   |   +-- data_loader.py          # load_data_from_aws()
|   |   +-- etl_dimensional.py
|   +-- evaluation/
|   |   +-- __init__.py
|   +-- features/
|   |   +-- __init__.py
|   |   +-- feature_engineering.py  # build_feature_matrix() -- Feature Schema v4
|   |   +-- pipeline.py             # Orquestador: feat_eng + preprocessing
|   |   +-- preprocessing.py        # preprocess() -- pipeline sklearn
|   +-- models/                     # Agregado en Sprint 2
|       +-- __init__.py
|       +-- train.py                # Entrenamiento LightGBM con Optuna
|
+-- tests/
|
+-- .gitignore
+-- LICENSE
+-- README.md
+-- requirements.txt
```

---

## Pipeline de Ejecución

El proyecto permite dos modalidades de ejecución: mediante scripts independientes (ideal para desarrollo y debugging) o a través de un orquestador centralizado (MLOps) para producción y monitoreo de drift.

---

### 1. Ejecución Manual (Pasos Independientes)

Para el desarrollo inicial o pruebas unitarias, se pueden ejecutar los componentes por separado desde la raíz del repositorio:

#### Paso 1 — Carga de datos
Carga datos desde AWS RDS y genera la data raw en parquet:

```bash
python -m src.data.data_loader
```
Output: `data/raw/data.parquet`

### Paso 2 — Feature Engineering

Lee el parquet de data, calcula features y genera la matriz de entrenamiento:

```bash
python -m src.features.feature_engineering
```

Output: `data/processed/feature_matrix.parquet`

### Paso 3 — Entrenamiento

Lee el parquet, entrena LightGBM con Optuna y serializa el modelo:

```bash
# Con Optuna (50 trials -- recomendado)
python -m src.models.train

# Sin Optuna -- hiperparametros por defecto, mas rapido
python -m src.models.train --no-optuna

# Con mas trials
python -m src.models.train --trials 100
```

Output: `models/model.pkl`, `models/cluster_models.pkl`, `models/model_log.json`

### 2. Orquestación Profesional (`pipeline.py`)

Para entornos de producción y automatización de MLOps, el proyecto utiliza un orquestador centralizado (`src/pipeline.py`) que integra todo el flujo en memoria. Esto optimiza el rendimiento al evitar lecturas/escrituras intermedias en disco y asegura la trazabilidad completa del experimento.

### A. Pipeline de Entrenamiento (Full Retrain)
Ejecuta los pasos de carga, ingeniería de atributos, validación y entrenamiento bajo un único run de **MLflow**.

```bash
# Ejecución estándar con optimización de hiperparámetros
python -m src.pipeline --trials 50

# Ejecución rápida (debugging) con muestra reducida y sin Optuna
python -m src.pipeline --n-users 5000 --no-optuna
```

- Validación de Calidad: Antes del entrenamiento, el pipeline ejecuta validate_feature_matrix. Si los datos no superan los checks de calidad, el proceso se detiene para proteger la integridad del modelo.

- MLflow Tracking: Registra automáticamente parámetros (n_users, trials), métricas de rendimiento (Precision, Recall, F1, AUC) y guarda los modelos (.pkl) como artefactos.

- Referencia para Drift: Si USE_S3=true, sube la matriz de entrenamiento a S3 como el baseline oficial para futuras comparaciones de monitoreo.

### B. Snapshot Semanal (Monitoreo de Drift)
Genera la matriz de características con los datos más recientes de la base de datos sin disparar un entrenamiento nuevo.

```bash
python -m src.pipeline --snapshot-only
```

- Propósito: Este modo es utilizado por jobs programados (ej. GitHub Actions) para obtener una "foto" actual de los datos.

- Almacenamiento en S3: El snapshot se guarda en s3://<BUCKET>/monitoring/actual/feature_matrix.parquet, quedando disponible para que el sistema de monitoreo detecte desviaciones estadísticas (Data Drift).

### Paso 4 — API de recomendaciones (FastAPI)

> **Requisito previo:** los Pasos 2 y 3 deben haberse ejecutado al menos una vez para que existan los artefactos `models/model.pkl`, `models/cluster_models.pkl` y `models/model_log.json`.

Levanta la API con los artefactos entrenados:

```bash
# Desarrollo (con recarga automatica al modificar archivos)
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload

# Produccion (sin --reload)
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Endpoints disponibles en Swagger: `http://localhost:8000/docs`

| Metodo | Endpoint | Descripcion |
| ------ | -------- | ----------- |
| `GET` | `/health` | Estado de la API y modelo cargado |
| `POST` | `/recommend/{user_id}` | Top-10 recomendaciones para un usuario |
| `POST` | `/recommend/batch` | Top-10 recomendaciones para hasta 100 usuarios |

#### Ejemplos de uso

```bash
# Individual
curl -X POST "http://localhost:8000/recommend/72136" \
     -H "accept: application/json"

# Batch
curl -X POST "http://localhost:8000/recommend/batch" \
     -H "Content-Type: application/json" \
     -d "{\"user_ids\":[72136, 66806, 132022]}"
```

#### Ejemplo de respuesta — `/recommend/{user_id}`

```json
{
  "user_id": 72136,
  "recommendations": [
    {"product_key": 13176, "product_name": "Bag of Organic Bananas", "probability": 0.678},
    {"product_key": 21137, "product_name": "Organic Strawberries", "probability": 0.645}
  ]
}
```

#### Comportamiento por cantidad de ordenes

| Ordenes prior del usuario | Comportamiento |
|---|---|
| Usuario no existe en BD | `200` — **cold-start global**: top-10 productos más populares |
| `0` (existe en BD sin historial) | `404` — usuario sin historial prior |
| `1 – 4` | `200` — **cold-start personal**: top-10 productos más comprados por el usuario |
| `>= 5` | `200` — recomendaciones del modelo LightGBM + flag `cold_start: false` |


El campo `probability` en cold-start refleja la frecuencia de compra normalizada por órdenes (cuántas de sus órdenes incluyeron ese producto), no una probabilidad del modelo.

#### Codigos de respuesta HTTP

| Codigo | Causa |
| ------ | ----- |
| `200` | Recomendaciones generadas correctamente (modelo ML o cold-start) |
| `404` | `user_id` no tiene ningún historial en la base de datos (0 órdenes prior) |
| `400` | Las features calculadas no coinciden con el contrato del modelo |
| `503` | No se pudo conectar a PostgreSQL (verificar `.env` y estado de Neon) |
| `500` | Error interno inesperado (ver CloudWatch Logs en AWS) |

---

## Base de datos

El proyecto usa dos bases de datos:

**Local (PostgreSQL):** modelo relacional normalizado con los CSVs originales de Instacart. Se usa como fuente para el ETL.

**AWS RDS (PostgreSQL cloud):** modelo dimensional star schema con los datos filtrados y listos para feature engineering.

### Esquema dimensional en AWS RDS

| Tabla                 | Filas     | Descripcion                                          |
| --------------------- | --------- | ---------------------------------------------------- |
| `dim_user`            | 10.000    | Usuarios aptos (>=5 ordenes prior + >=1 orden train) |
| `dim_product`         | 26.686    | Productos aptos (>=50 compras globales)              |
| `fact_order_products` | 1.999.645 | Hechos de compra (prior + train)                     |

### Filtros aplicados en el ETL

- `eval_set != 'test'` -- excluir ordenes de test
- Usuarios con >= 5 ordenes `prior` y >= 1 orden `train`
- Productos con >= 50 compras globales en `prior`
- `LIMIT 10.000` usuarios aptos · `RANDOM_SEED = 42`

---

## Modelo

### Feature Schema -- 26 columnas + label

| Grupo           | Features                                                                                                                                                                                                                                                                | Count |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| IDs             | `user_key`, `product_key`                                                                                                                                                                                                                                               | 2     |
| Usuario         | `user_total_orders`, `user_avg_basket_size`, `user_days_since_last_order`, `user_reorder_ratio`, `user_distinct_products`, `user_segment_code`                                                                                                                          | 6     |
| Producto        | `product_total_purchases`, `product_reorder_rate`, `product_avg_add_to_cart`, `product_unique_users`, `p_department_reorder_rate`, `p_aisle_reorder_rate`                                                                                                               | 6     |
| Interaccion u×p | `up_times_purchased`, `up_reorder_rate`, `up_orders_since_last_purchase`, `up_first_order_number`, `up_last_order_number`, `up_avg_add_to_cart_order`, `up_days_since_last`, `up_avg_days_between_orders`, `up_delta_days`, `u_favorite_department`, `u_favorite_aisle` | 11    |
| Label           | `label`                                                                                                                                                                                                                                                                 | 1     |

### Resultados del modelo (Sprint 2)

Split de evaluacion: **70/15/15 por usuarios** -- cada usuario aparece en un unico conjunto.

| Modelo               | Precision  | Recall     | F1         | AUC-ROC    | NDCG@10    |
| -------------------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| Baseline popularidad | 0.2370     | 0.1159     | 0.1557     | -          | -          |
| LightGBM baseline    | 0.0000     | 0.0000     | 0.0000     | 0.8257     | -          |
| LightGBM optimizado  | **0.4347** | 0.4106     | **0.4223** | 0.8197     | **0.5108** |
| CatBoost             | 0.2511     | **0.7386** | 0.3748     | **0.8253** | -          |

**Uplift vs baseline de popularidad global:** +296% — el modelo personalizado encuentra casi 3 veces más productos relevantes por usuario que recomendar los más vendidos globalmente (+2.1 productos/usuario en promedio).

Modelo en produccion: LightGBM optimizado con Optuna (50 trials · `scale_pos_weight=8.88`).

Top 5 features mas importantes: `up_reorder_rate`, `up_days_since_last`, `product_reorder_rate`, `user_reorder_ratio`, `up_delta_days`.

---

## MLOps Lifecycle

### Flujo semanal automatizado

Cada domingo a las **23:00 UTC**, GitHub Actions ejecuta un ciclo MLOps con cuatro etapas desacopladas:

1. **Generación de snapshot**
   Se extraen datos desde AWS RDS PostgreSQL, se construye la feature matrix en memoria y se sube a S3 como snapshot actual.

2. **Detección de drift**
   El job de monitoreo compara el snapshot actual contra la referencia del último entrenamiento exitoso usando **PSI** y **KS-test** por feature.

3. **Reentrenamiento condicionado**
   Si hay drift detectado, o si se fuerza manualmente, se ejecuta el pipeline completo de entrenamiento con **Optuna (50 trials)**.

4. **Despliegue en AWS ECS Fargate**
   Si el retrain finaliza correctamente, el workflow dispara un **rolling update** del servicio ECS para que los nuevos tasks carguen el modelo vigente desde S3.

### Regla de activación de drift

| Métrica | Umbral | Interpretación operativa |
|---|---:|---|
| `PSI` | `> 0.25` | Cambio significativo en la distribución |
| `KS` | `> 0.30` | Diferencia estadística relevante entre baseline y snapshot actual |

La decisión de reentrenar se toma cuando se cumple al menos una de estas condiciones:

```text
drift_detected = (PSI > 0.25) OR (KS > 0.30)
```

## Architecture Overview

### Infrastructure

| Componente | Implementación | Rol en la solución |
|---|---|---|
| Orquestación MLOps | GitHub Actions | Ejecuta snapshot, drift-check, retrain y deploy |
| Data source | AWS RDS PostgreSQL | Fuente de datos transaccionales para features |
| Artifact store | Amazon S3 | Versionado y distribución de artefactos del modelo |
| Model serving | FastAPI | Expone endpoints de inferencia y health check |
| Compute de serving | AWS ECS Fargate | Corre contenedores sin administrar servidores |
| Experiment tracking | MLflow | Registra parámetros, métricas y artefactos |
| Hyperparameter tuning | Optuna | Optimiza el entrenamiento en cada retrain |

### Tech Stack

| Capa | Stack |
|---|---|
| Lenguaje | Python 3.10 |
| API | FastAPI + Uvicorn |
| Training | LightGBM + Optuna |
| Data | pandas, NumPy, Parquet |
| Cloud | AWS S3, AWS RDS, ECS Fargate, IAM |
| CI/CD | GitHub Actions |
| Tracking | MLflow |

## Key Features

### 1. Drift monitoring basado en evidencia estadística

El monitoreo no depende de intuiciones ni de checks superficiales. El pipeline calcula **PSI** y **KS** sobre features numéricas críticas como:

- `user_total_orders`
- `user_avg_basket_size`
- `user_reorder_ratio`
- `product_total_purchases`
- `product_reorder_rate`
- `up_times_purchased`
- `up_reorder_rate`
- `up_days_since_last`

Además, genera un `drift_report.json` con:

- `timestamp`
- `triggered_by`
- `retrain_run_id`
- `psi`, `ks`
- `drift_detected`
- breakdown por feature (`psi_by_feature`, `ks_by_feature`)

### 2. Robustez ante Cold Start

Uno de los puntos más sólidos del diseño es el manejo explícito del **primer entrenamiento**:

- Si aún no existe `feature_matrix_reference.parquet`, el monitoreo **no falla de forma ciega**; se omite con logging claro.
- En el workflow, antes del retrain, se verifica si existe `s3://insight-commerce-artifacts/model_log.json`.
- Si no existe baseline, el pipeline considera el caso como **first training**, ejecuta el entrenamiento y establece la nueva referencia para futuras comparaciones.
- Tras el primer entrenamiento exitoso, GitHub Actions abre una notificación operativa indicando que el baseline quedó inicializado.

Esto evita uno de los errores más comunes en proyectos MLOps junior: asumir que siempre existe un modelo previo o una distribución de referencia válida.

### 3. Despliegue desacoplado del entrenamiento

El deploy no reconstruye la imagen en cada nuevo modelo. En cambio:

- los artefactos se actualizan en **S3**,
- ECS ejecuta `force-new-deployment`,
- los nuevos tasks levantan la misma imagen de API,
- la API carga el nuevo modelo desde S3 durante `startup()`.

Este patrón reduce tiempo de despliegue, costo de red y fricción operativa.

### 4. Inyección segura de secretos

El workflow consume secretos desde **GitHub Secrets** para:

- credenciales AWS (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`)
- conexión a RDS (`AWS_HOST`, `AWS_DATABASE`, `AWS_USER`, `AWS_PASSWORD`, `AWS_PORT`, `AWS_SSLMODE`)
- tracking de experimentos (`MLFLOW_TRACKING_URI`)

La operación del pipeline queda desacoplada del código fuente y alineada con buenas prácticas de seguridad.

## Artifact Strategy

### S3 como capa de versionado operativo

Amazon S3 actúa como repositorio central de artefactos y contratos del modelo.

| Artefacto | Ubicación lógica | Propósito |
|---|---|---|
| Snapshot actual | `monitoring/actual/feature_matrix.parquet` | Distribución semanal para drift detection |
| Baseline de entrenamiento | `feature_matrix_reference.parquet` | Referencia estadística post-retrain |
| Modelo principal | `model.pkl` | Modelo de inferencia en producción |
| Modelos auxiliares | `cluster_models.pkl` | Artefactos de clustering / transformación |
| Contrato del modelo | `model_log.json` | Metadata, features y versionado lógico |

`model_log.json` es especialmente importante porque funciona como **contrato de artefactos**, facilitando trazabilidad y validación de compatibilidad entre entrenamiento e inferencia.

## Cloud Deployment

### AWS ECS Fargate

La capa de serving corre sobre **AWS ECS Fargate** con una estrategia de **rolling update**.

| Recurso | Valor |
|---|---|
| Cluster | `insight-commerce-cluster` |
| Servicio | `insight-api-service` |
| Task Definition | `insight-commerce-task` |
| Contenedor | `api-container` |
| Puerto de serving | `8000` |

### IAM y acceso a artefactos

El task role **`ecsTaskRole-InsightCommerce`** dispone de permisos de lectura sobre S3 para que el contenedor recupere artefactos al iniciar, sin embutir credenciales estáticas en la imagen.

Esto habilita un patrón limpio:

```text
Nuevo task ECS inicia
-> FastAPI arranca
-> RecommendationService carga artefactos desde S3
-> /health confirma modelo activo y número de features
```

## Serving Layer

La API FastAPI expone una capa de inferencia productiva con foco operacional.

| Endpoint | Método | Propósito |
|---|---|---|
| `/health` | `GET` | Verifica estado del servicio y artefactos cargados |
| `/recommend/{user_id}` | `POST` | Devuelve top-N recomendaciones para un usuario |
| `/recommend/batch` | `POST` | Ejecuta inferencia por lote |

El endpoint `/health` reporta:

- estado del servicio
- modelo cargado
- número de features
- artefactos activos (`model.pkl`, `cluster_models.pkl`, `model_log.json`)

---

## Git Workflow -- Ramas y Pull Requests

### Estructura de ramas

```
main
+-- develop
        +-- feature/eda-exploratorio
        +-- feature/feature-engineering
        +-- feature/etl-neon-dimensional
        +-- feature/modelo-lightgbm
        +-- feature/api-fastapi
        +-- feature/demo-streamlit
        +-- feature/dashboard-metricas
        +-- hotfix/descripcion-del-fix
```

| Rama        | Proposito                                  | Desplegada en |
| ----------- | ------------------------------------------ | ------------- |
| `main`      | Codigo en produccion, siempre estable      | Produccion    |
| `develop`   | Integracion continua, base de trabajo      | Staging / QA  |
| `feature/*` | Desarrollo de funcionalidades individuales | Local / Dev   |

### Flujo de trabajo

#### 1. Crear una rama de feature

Siempre parte desde `develop`:

```bash
git checkout develop
git pull origin develop
git checkout -b feature/nombre-descriptivo
```

Convencion de commits:

```
tipo(scope): descripcion breve en imperativo

Ejemplos:
feat(eda): agregar analisis de distribucion de recompra por categoria
fix(etl): corregir filtro de usuarios en fact usando loaded_users desde Neon
docs(readme): actualizar instrucciones de instalacion
refactor(model): separar pipeline de features en modulo independiente
test(api): agregar test de endpoint /recommend
chore(deps): actualizar lightgbm a version 4.1

Tipos validos: feat, fix, docs, refactor, test, chore, style, perf
```

#### 2. Desarrollar y hacer commits

```bash
git add .
git commit -m "feat: descripcion clara del cambio"
git push origin feature/nombre-descriptivo
```

#### 3. Abrir un Pull Request hacia `develop`

- Ir al repositorio en GitHub
- Crear un PR desde `feature/*` hacia `develop`
- Completar la plantilla de PR
- Asignar al menos un revisor del equipo

#### 4. Revision de codigo

- El revisor analiza el codigo, deja comentarios y aprueba o solicita cambios
- El autor responde los comentarios y realiza las correcciones necesarias
- No se puede hacer merge sin al menos 1 aprobacion

#### 5. Merge a `develop`

Una vez aprobado, desde la interfaz de GitHub (squash merge recomendado).

#### 6. Release a `main`

Cuando `develop` esta estable y validado en QA:

```bash
git checkout main
git pull origin main
git merge --no-ff develop
git tag -a v1.x.x -m "Release v1.x.x"
git push origin main --tags
```

### Reglas de Pull Requests

Obligatorio para todo PR:

- Al menos 1 aprobacion de un miembro del equipo antes del merge
- Sin conflictos con la rama base
- Descripcion clara de los cambios realizados

Protecciones de ramas:

| Rama        | Merge directo | PR requerido | Aprobaciones minimas |
| ----------- | :-----------: | :----------: | :------------------: |
| `main`      |      No       |      Si      |          1           |
| `develop`   |      No       |      Si      |          1           |
| `feature/*` |      Si       |      -       |          -           |

### Plantilla de Pull Request

```markdown
## Descripcion

Breve resumen de los cambios y el contexto del problema que resuelven.

## Issue relacionado

Card #NRO

## Tipo de cambio

- [ ] Nueva funcionalidad
- [ ] Correccion de bug
- [ ] Refactor
- [ ] Documentacion
- [ ] Configuracion / chore

## Checklist

- [ ] El codigo sigue los estandares del proyecto
- [ ] He anadido/actualizado tests necesarios
- [ ] He actualizado la documentacion si aplica
- [ ] He probado los cambios localmente
- [ ] No hay conflictos con la rama base
```
