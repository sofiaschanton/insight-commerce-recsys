# insight-commerce-recsys

Sistema de recomendacion de proxima compra - Proyecto Final Data Science

---

## Instalacion

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
# Base de datos local (PostgreSQL)
LOCAL_HOST=localhost
LOCAL_DATABASE=InstaCart_DB
LOCAL_USER=postgres
LOCAL_PASSWORD=tu_password
LOCAL_PORT=5432

# AWS RDS PostgreSQL (nube)
AWS_HOST=
AWS_DATABASE=
AWS_USER=
AWS_PASSWORD=
AWS_PORT=
AWS_SSLMODE=

# Configuracion del proyecto
DATA_PATH=data/raw
RANDOM_SEED=42
N_USERS=100000

# API de recomendaciones (FastAPI)
API_MODEL_PATH=models/model.pkl
API_CLUSTER_MODEL_PATH=models/cluster_models.pkl
API_MODEL_LOG_PATH=models/model_log.json
API_HOST=0.0.0.0
API_PORT=8000
```

### Descripcion de variables

| Variable                 | Descripcion                                      | Valor por defecto           |
| ------------------------ | ------------------------------------------------ | --------------------------- |
| `LOCAL_HOST`             | Direccion del servidor PostgreSQL local          | `localhost`                 |
| `LOCAL_DATABASE`         | Nombre de la base de datos local                 | `InstaCart_DB`              |
| `LOCAL_USER`             | Usuario PostgreSQL local                         | `postgres`                  |
| `LOCAL_PASSWORD`         | Contrasena PostgreSQL local                      | -                           |
| `LOCAL_PORT`             | Puerto PostgreSQL local                          | `5432`                      |
| `AWS_HOST`               | Host de AWS RDS PostgreSQL                       | -                           |
| `AWS_DATABASE`           | Nombre de la base de datos en AWS RDS            | -                           |
| `AWS_USER`               | Usuario AWS RDS                                  | -                           |
| `AWS_PASSWORD`           | Contrasena AWS RDS                               | -                           |
| `AWS_PORT`               | Puerto AWS RDS                                   | `5432`                      |
| `AWS_SSLMODE`            | Modo SSL AWS RDS (`disable/prefer` si host local)| `require`                   |
| `DATA_PATH`              | Ruta a los CSVs originales                       | `data/raw`                  |
| `RANDOM_SEED`            | Semilla aleatoria global                         | `42`                        |
| `N_USERS`                | Usuarios a considerar en EDA local               | `100000`                    |
| `API_MODEL_PATH`         | Ruta al artefacto del modelo de inferencia       | `models/model.pkl`          |
| `API_CLUSTER_MODEL_PATH` | Ruta al artefacto de clusters (KMeans + scalers) | `models/cluster_models.pkl` |
| `API_MODEL_LOG_PATH`     | Ruta al contrato de features (`model_log.json`)  | `models/model_log.json`     |
| `API_HOST`               | Host para levantar FastAPI                       | `0.0.0.0`                   |
| `API_PORT`               | Puerto para levantar FastAPI                     | `8000`                      |

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
+-- app/                            # Streamlit (pendiente)
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

## Pipeline de ejecucion

El proyecto se ejecuta en dos pasos independientes desde la raiz del repositorio.

### Paso 1 — Feature pipeline

Carga datos desde AWS RDS, calcula features y genera el feature matrix:

```bash
python -m src.features.pipeline
```

Output: `data/processed/feature_matrix.parquet`

Solo es necesario volver a correr este paso si cambian las features o los datos en AWS RDS.

### Paso 2 — Entrenamiento

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

### Paso 3 — API de recomendaciones (FastAPI)

> **Requisito previo:** los Pasos 1 y 2 deben haberse ejecutado al menos una vez para que existan los artefactos `models/model.pkl`, `models/cluster_models.pkl` y `models/model_log.json`.

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
| ------------------------- | -------------- |
| `0` | `404` — usuario sin historial |
| `1 – 4` | `200` — **cold-start**: top-10 productos más comprados por el usuario (ranking por frecuencia, sin ML) |
| `>= 5` | `200` — recomendaciones generadas por el modelo LightGBM |

El campo `probability` en cold-start refleja la frecuencia de compra normalizada por órdenes (cuántas de sus órdenes incluyeron ese producto), no una probabilidad del modelo.

#### Codigos de respuesta HTTP

| Codigo | Causa |
| ------ | ----- |
| `200` | Recomendaciones generadas correctamente (modelo ML o cold-start) |
| `404` | `user_id` no tiene ningún historial en la base de datos (0 órdenes prior) |
| `400` | Las features calculadas no coinciden con el contrato del modelo |
| `503` | No se pudo conectar a PostgreSQL (verificar `.env` y estado de Neon) |
| `500` | Error interno inesperado (ver `src/api/reports/logs/api.log`) |

Logs API: `src/api/reports/logs/api.log` (UTF-8). La carpeta se crea automaticamente al iniciar.

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

### Feature Schema v4 -- 26 columnas + label

| Grupo           | Features                                                                                                                                                                                                                                                                | Count |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| IDs             | `user_key`, `product_key`                                                                                                                                                                                                                                               | 2     |
| Usuario         | `user_total_orders`, `user_avg_basket_size`, `user_days_since_last_order`, `user_reorder_ratio`, `user_distinct_products`, `user_segment_code`                                                                                                                          | 6     |
| Producto        | `product_total_purchases`, `product_reorder_rate`, `product_avg_add_to_cart`, `product_unique_users`, `p_department_reorder_rate`, `p_aisle_reorder_rate`                                                                                                               | 6     |
| Interaccion u×p | `up_times_purchased`, `up_reorder_rate`, `up_orders_since_last_purchase`, `up_first_order_number`, `up_last_order_number`, `up_avg_add_to_cart_order`, `up_days_since_last`, `up_avg_days_between_orders`, `up_delta_days`, `u_favorite_department`, `u_favorite_aisle` | 11    |
| Label           | `label`                                                                                                                                                                                                                                                                 | 1     |

### Resultados del modelo (Sprint 2)

Split de evaluacion: **70/15/15 por usuarios** -- cada usuario aparece en un unico conjunto.

| Modelo               | Precision  | Recall     | F1         | AUC-ROC    |
| -------------------- | ---------- | ---------- | ---------- | ---------- |
| Baseline popularidad | 0.2370     | 0.1159     | 0.1557     | -          |
| LightGBM baseline    | 0.0000     | 0.0000     | 0.0000     | 0.8257     |
| LightGBM optimizado  | **0.4347** | 0.4106     | **0.4223** | 0.8197     |
| CatBoost             | 0.2511     | **0.7386** | 0.3748     | **0.8253** |

Modelo en produccion: LightGBM optimizado con Optuna (50 trials · `scale_pos_weight=8.88`).

Top 5 features mas importantes: `up_reorder_rate`, `up_days_since_last`, `product_reorder_rate`, `user_reorder_ratio`, `up_delta_days`.

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
