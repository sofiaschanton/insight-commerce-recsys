# CI/CD — insight-commerce-recsys

Pipeline de integración y entrega continua basado en GitHub Actions, dividido en dos workflows independientes con responsabilidades separadas.

---

## Dos workflows, dos responsabilidades

| Workflow | Archivo | Trigger | Propósito |
|----------|---------|---------|-----------|
| **CI** | `ci.yml` | push / pull_request | Calidad de código: tests, tipos, cobertura |
| **CD** | `cd.yml` | `ci.yml` exitoso en main / manual | Deploy de código a ECS: rolling update + health check |
| **MLOps** | `mlops.yml` | cron semanal / manual | Pipeline MLOps: snapshot → drift → retrain → deploy |

---

## Estructura del pipeline

```
ci.yml — push (main, develop, feature/**, fix/**) / pull_request (main, develop)
        │
        └─── test
               ├─ pip install -r requirements.txt
               ├─ pytest --cov=src → coverage.xml
               ├─ mypy src/
               └─ SonarCloud scan
                       │
                       │ ci.yml exitoso en main
                       ▼
cd.yml — workflow_run (ci.yml completed) / workflow_dispatch (skip_deploy)
        │
        └─── deploy
               ├─ [skip si skip_deploy=true] aws ecs update-service --force-new-deployment
               ├─ [skip si skip_deploy=true] aws ecs wait services-stable
               ├─ health check GET /health → ALB (5 reintentos)
               ├─ smoke test POST /recommend
               ├─ rollback automático a task definition anterior si falla
               └─ GitHub Issue si deploy falla


mlops.yml — schedule (domingos 23:00 UTC) / workflow_dispatch
        │
        └─── build-snapshot
               ├─ RDS → build_feature_matrix() en memoria
               └─ upload → s3://insight-commerce-artifacts/monitoring/actual/feature_matrix.parquet
                       │
                       └─── drift-check (needs: build-snapshot)
                              ├─ descarga baseline (feature_matrix_reference.parquet)
                              ├─ descarga actual (monitoring/actual/feature_matrix.parquet)
                              ├─ PSI + KS por feature → drift_report.json
                              ├─ si drift → GitHub Issue label "data-drift"
                              └─ output: drift_detected (true/false)
                                      │
                              drift_detected == true
                                      │
                                      └─── retrain (needs: drift-check)
                                             ├─ pipeline.py --trials 50 (RDS → features → Optuna → train)
                                             ├─ rollback check (F1 nuevo vs anterior)
                                             ├─ si ok    → sube artefactos a S3 + GitHub Issue "primer train" (si aplica)
                                             ├─ si rollback → job falla + GitHub Issue "rollback"
                                             └─ result == 'success'
                                                     │
                                                     └─── deploy (needs: retrain)
                                                            ├─ aws ecs update-service --force-new-deployment
                                                            ├─ aws ecs wait services-stable
                                                            ├─ health check GET /health → ALB (5 reintentos)
                                                            ├─ smoke test POST /recommend
                                                            ├─ rollback automático si falla
                                                            └─ GitHub Issue si deploy falla
```

---

## Triggers

### `ci.yml`

| Evento | Job activado |
|--------|-------------|
| `push` a `main`, `develop`, `feature/**`, `fix/**` | `test` |
| `pull_request` hacia `main`, `develop` | `test` |
| `workflow_dispatch` | `test` |

### `cd.yml`

| Evento | Jobs activados | Condición |
|--------|----------------|-----------|
| `workflow_run` — `ci.yml` completado en `main` | `deploy` | Solo si `ci.yml` terminó con `success` |
| `workflow_dispatch` con `skip_deploy=false` (default) | `deploy` completo | ECS rolling update + health check |
| `workflow_dispatch` con `skip_deploy=true` | Solo health check + smoke test | ECS no se toca — útil para verificar prod sin deployar |

### `mlops.yml`

| Evento | Jobs activados | Condición |
|--------|----------------|-----------|
| `schedule` — domingos 23:00 UTC | `build-snapshot → drift-check → retrain → deploy` | retrain solo si drift detectado |
| `workflow_dispatch` con `skip_snapshot=true` | `drift-check → retrain → deploy` | usa datos ya existentes en S3 |
| `workflow_dispatch` con `force_retrain=true` | `build-snapshot → drift-check → retrain → deploy` | retrain aunque no haya drift |

---

## Jobs — `ci.yml`

### `test` — Tests + type check + SonarCloud

**Runner:** `ubuntu-latest` · **Python:** `3.10`

| Paso | Descripción |
|------|-------------|
| `actions/checkout@v4` | Clona el repositorio con historial completo (`fetch-depth: 0`) |
| `actions/setup-python@v5` | Python 3.10 con caché de pip |
| Install dependencies | `pip install -r requirements.txt` + `pytest pytest-cov mypy` |
| Run tests | `pytest --cov=src --cov-report=xml:coverage.xml` |
| Type check | `mypy src/ --ignore-missing-imports --no-strict-optional` |
| SonarCloud Scan | Análisis estático + cobertura. Consume `coverage.xml`. |

---

## Jobs — `cd.yml`

### `deploy` — Deploy de código a ECS

**Runner:** `ubuntu-latest`
**Trigger:** `ci.yml` exitoso en `main`, o `workflow_dispatch`

Deploya cambios de código a ECS sin reentrenar el modelo. El modelo en S3 no se toca — solo se reinicia el contenedor para que levante con el código nuevo.

| Paso | Descripción |
|------|-------------|
| Configure AWS credentials | IAM via `aws-actions/configure-aws-credentials@v4` |
| Force new ECS deployment | `aws ecs update-service --force-new-deployment` — omitido si `skip_deploy=true` |
| Wait for stability | `aws ecs wait services-stable` — omitido si `skip_deploy=true` |
| Health check post-deploy | `GET /health` contra ALB (5 reintentos, 10s entre intentos) |
| Smoke test del modelo | `POST /recommend` con `user_id: 1` — valida que el modelo responde |
| Rollback automático | Si falla: restaura task definition anterior via `aws ecs update-service --task-definition` |
| Notify deploy failure | GitHub Issue con labels `rollback`, `cd`, `deploy` |

> **`skip_deploy=true`:** permite correr solo el health check y smoke test sin tocar ECS. Útil para el primer merge a main cuando el código ya está en producción.

---

## Jobs — `mlops.yml`

### 1. `build-snapshot` — Snapshot semanal de features

**Runner:** `ubuntu-latest`
**Se omite si:** `workflow_dispatch` con `skip_snapshot=true`

Conecta a RDS, construye la feature matrix completa en memoria y la sube a S3. Este archivo es la distribución "actual" que el job `drift-check` comparará contra el baseline del último reentrenamiento.

| Paso | Descripción |
|------|-------------|
| Configure AWS credentials | IAM via `aws-actions/configure-aws-credentials@v4` |
| Build and upload snapshot | `python -m src.pipeline --snapshot-only` → `s3://.../monitoring/actual/feature_matrix.parquet` |

**S3 destino:** `s3://insight-commerce-artifacts/monitoring/actual/feature_matrix.parquet`

> **¿Por qué un snapshot separado del reentrenamiento?**
> El drift check debe ser una comparación estadística rápida y aislada de fallos de infraestructura. Si la conexión a RDS fallara dentro del mismo job de drift, el fallo sería indistinguible de un problema estadístico. Separar la carga de datos (job A) del análisis de drift (job B) garantiza trazabilidad: el parquet en S3 queda como evidencia de qué datos exactamente dispararon la alerta.

---

### 2. `drift-check` — Detección de drift PSI + KS

**Runner:** `ubuntu-latest`
**Depende de:** `build-snapshot` (o se ejecuta con datos ya en S3 si `skip_snapshot=true`)
**Output:** `drift_detected` (true/false)

Descarga dos parquets desde S3 y compara sus distribuciones para 8 features clave. No accede a RDS.

| Paso | Descripción |
|------|-------------|
| Configure AWS credentials | IAM via `aws-actions/configure-aws-credentials@v4` |
| Run drift monitoring | `python -m src.model_monitoring` → `drift_report.json` |
| Parse drift result | Lee `drift_report.json`, escribe `drift_detected` al output y al Step Summary |
| Upload drift report | Sube `drift_report.json` como artefacto (retención 90 días) |
| Create GitHub Issue | Solo si `drift_detected=true`. Label: `data-drift`. |

**Archivos S3 comparados:**

| Rol | S3 key | Escrito por |
|-----|--------|-------------|
| Actual (semanal) | `monitoring/actual/feature_matrix.parquet` | `build-snapshot` |
| Referencia (baseline) | `feature_matrix_reference.parquet` | `retrain` (tras cada reentrenamiento) |

**Schema de `drift_report.json`:**

```json
{
  "timestamp":      "2026-03-17T23:01:42.000000+00:00",
  "triggered_by":   "weekly_schedule",
  "retrain_run_id": "12345678",
  "psi":            0.31,
  "ks":             0.28,
  "drift_detected": true,
  "psi_by_feature": {
    "user_total_orders":     0.12,
    "user_avg_basket_size":  0.08,
    "user_reorder_ratio":    0.41,
    "product_total_purchases": 0.05,
    "product_reorder_rate":  0.09,
    "up_times_purchased":    0.33,
    "up_reorder_rate":       0.29,
    "up_days_since_last":    0.11
  },
  "ks_by_feature": { "...": "..." }
}
```

**Umbrales de alerta** (definidos en `src/model_monitoring.py`):

| Métrica | Umbral | Consecuencia |
|---------|--------|-------------|
| PSI | `>= 0.25` | `drift_detected = true` → dispara `retrain` |
| KS  | `>= 0.30` | `drift_detected = true` → dispara `retrain` |
| PSI | `0.10 – 0.24` | Cambio moderado — monitorear pero no retrain |
| PSI | `< 0.10` | Distribución estable |

> **¿Por qué chequeo semanal?** Los patrones de compra en e-commerce cambian lentamente — estacionalidad, promociones, cambios de catálogo. Un chequeo semanal es suficiente para detectar cambios relevantes sin incurrir en costos innecesarios de cómputo.

---

### 3. `retrain` — Reentrenamiento drift-gated

**Runner:** `ubuntu-latest`
**Trigger:** `drift_detected == true` o `force_retrain == true`
**Depende de:** `drift-check`

Ejecuta el pipeline completo: carga RDS → feature engineering → validación → Optuna 50 trials → LightGBM. Si el F1 nuevo degrada más del 5%, el mecanismo de rollback bloquea la sobreescritura de artefactos.

| Paso | Descripción |
|------|-------------|
| Configure AWS credentials | IAM via `aws-actions/configure-aws-credentials@v4` |
| Check if first training run | `aws s3 ls s3://.../model_log.json` para determinar si es el primer entrenamiento |
| Run full pipeline | `python -m src.pipeline --trials 50` con `USE_S3=true` |
| Upload model artifacts | `model.pkl`, `cluster_models.pkl`, `model_log.json` como artefactos de GitHub (retención 90 días) |
| Publish training summary | F1 y AUC al Step Summary de GitHub Actions |
| GitHub Issue — primer entrenamiento | Solo si es el primer modelo: notificación con métricas y artefactos. Label: `mlops` |
| GitHub Issue — rollback | Solo si el job falla: notificación con contexto y acciones recomendadas. Label: `rollback` |

**Artefactos subidos a S3 tras reentrenamiento exitoso:**

| Archivo S3 | Descripción |
|------------|-------------|
| `model.pkl` | Modelo LightGBM serializado |
| `cluster_models.pkl` | KMeans + StandardScaler (usuario y producto) |
| `model_log.json` | Métricas, feature_cols, parámetros, timestamp |
| `feature_matrix_reference.parquet` | Nuevo baseline para el próximo drift check |

**Rollback** (definido en `src/models/train.py`):

- Compara F1 nuevo vs F1 del `model_log.json` anterior (en S3 o en disco)
- Si `F1_nuevo < F1_anterior × 0.95` → `RuntimeError`, job falla, **artefactos en S3 no se modifican**
- Si no existe `model_log.json` previo → primer entrenamiento, rollback omitido

---

### 4. `deploy` — Force deploy a ECS Fargate

**Runner:** `ubuntu-latest`
**Trigger:** `retrain.result == 'success'`
**Depende de:** `retrain`

Fuerza un nuevo deployment en ECS Fargate sin reconstruir la imagen Docker. Los tasks nuevos descargan los artefactos actualizados desde S3 al arrancar (`USE_S3=true` en la task definition).

| Paso | Descripción |
|------|-------------|
| Configure AWS credentials | IAM via `aws-actions/configure-aws-credentials@v4` |
| Force new ECS deployment | `aws ecs update-service --force-new-deployment` |
| Wait for stability | `aws ecs wait services-stable` — bloquea hasta que el servicio esté estable |
| Health check post-deploy | `GET /health` contra ALB (5 reintentos, 10s entre intentos) |
| Smoke test del modelo | `POST /recommend` con `user_id: 1` — valida que el modelo responde correctamente |
| Rollback automático | Si health check o smoke test fallan: restaura task definition anterior |
| Notify deploy failure | GitHub Issue con label `rollback`, `mlops`, `deploy` si el deploy falla |

**Infraestructura ECS:**

| Campo | Valor |
|-------|-------|
| Cluster | `insight-commerce-cluster` |
| Servicio | `insight-api-service` |
| Task definition | `insight-commerce-task` |
| Contenedor | `api-container` |
| Región | `us-east-2` |

**Modelo de carga del modelo:**

```
ECS inicia task nuevo
        │
        └─ startup() en RecommendationService
                │
                └─ _load_artifacts() con USE_S3=true
                        │
                        ├─ s3://insight-commerce-artifacts/model.pkl
                        ├─ s3://insight-commerce-artifacts/cluster_models.pkl
                        └─ s3://insight-commerce-artifacts/model_log.json
```

El task role `ecsTaskRole-InsightCommerce` tiene política `AmazonS3ReadOnlyAccess`. No se requieren credenciales adicionales para la lectura desde S3.

---

## CD — Estrategia de deploy

### Rolling Update con ECS Fargate (implementado)

La estrategia de deploy es **rolling update nativo de ECS**: `--force-new-deployment` ordena a ECS reemplazar los tasks existentes con tasks nuevos que usan la misma imagen Docker ya registrada en ECR, pero descargan el modelo actualizado desde S3 al iniciar.

**¿Por qué no se reconstruye la imagen Docker en cada retrain?**

El código de la API no cambia entre retrains — solo cambian los artefactos del modelo en S3. Reconstruir y subir una imagen Docker (200-400 MB) para que el contenedor simplemente cargue un archivo diferente desde S3 es un overhead innecesario. El `force-new-deployment` logra el mismo resultado (modelo nuevo en producción) con menor latencia y menor consumo de red y almacenamiento en ECR.

```
retrain exitoso → model.pkl nuevo en S3
        │
        └─ aws ecs update-service --force-new-deployment
                │
                └─ ECS arranca tasks nuevos (misma imagen, modelo nuevo desde S3)
                        │
                        └─ aws ecs wait services-stable
                                │
                        ┌───────┴───────┐
                   tasks OK         tasks fallan
                        │                │
                  deploy completo    ECS revierte automáticamente
                                    (tasks previos siguen activos)
```

---

### Rolling Update vs Blue/Green

| Criterio | Rolling Update (implementado) | Blue/Green (mejora futura) |
|----------|-------------------------------|---------------------------|
| **ALB requerido** | No | Sí (obligatorio) |
| **Costo adicional** | Ninguno | Duplica tasks Fargate durante el deploy |
| **Downtime** | Mínimo (segundos) | Zero (switch instantáneo en el ALB) |
| **Rollback** | Automático si task falla al arrancar | Instantáneo via switch de tráfico en ALB |
| **Complejidad** | Baja — nativo en ECS | Alta — requiere CodeDeploy + target groups + `appspec.yml` |
| **Visibilidad** | Logs ECS + CloudWatch | CodeDeploy registra cada fase con gates |
| **Adecuado para** | Proyecto actual (sin ALB, presupuesto controlado) | Producción con SLA estricto y tráfico crítico |

**Blue/Green como mejora futura:** requiere ALB configurado con dos target groups (`blue` y `green`), AWS CodeDeploy application + deployment group, y `appspec.yml` en el repo. El job `deploy` en `mlops.yml` pasaría a usar `aws-actions/amazon-ecs-deploy-task-definition@v1` con configuración CodeDeploy. El resto del pipeline (snapshot → drift → retrain) no cambia.

---

## Secrets requeridos

Configurar en **Settings → Secrets and variables → Actions → New repository secret**.

### `ci.yml`

| Secret | Descripción | Estado |
|--------|-------------|--------|
| `SONAR_TOKEN` | Token de SonarCloud | Configurado |
| `GITHUB_TOKEN` | Token automático de GitHub | Automático |

### `cd.yml`

| Secret | Descripción | Estado |
|--------|-------------|--------|
| `AWS_ACCESS_KEY_ID` | Credencial IAM (ECS) | Configurado |
| `AWS_SECRET_ACCESS_KEY` | Credencial IAM | Configurado |
| `AWS_REGION` | Región AWS (`us-east-2`) | Configurado |
| `ALB_DNS` | DNS del ALB para health check y smoke test | Configurado |

### `mlops.yml`

| Secret | Descripción | Estado |
|--------|-------------|--------|
| `AWS_ACCESS_KEY_ID` | Credencial IAM (S3 + ECS) | Configurado |
| `AWS_SECRET_ACCESS_KEY` | Credencial IAM | Configurado |
| `AWS_REGION` | Región AWS (`us-east-2`) | Configurado |
| `AWS_HOST` | Host RDS PostgreSQL | Configurado |
| `AWS_DATABASE` | Nombre de la base de datos | Configurado |
| `AWS_USER` | Usuario RDS | Configurado |
| `AWS_PASSWORD` | Contraseña RDS | Configurado |
| `AWS_PORT` | Puerto RDS (`5432`) | Configurado |
| `AWS_SSLMODE` | Modo SSL (`require`) | Configurado |
| `MLFLOW_TRACKING_URI` | URI del servidor MLflow (vacío = archivo local) | Configurado |
| `ALB_DNS` | DNS del ALB para health check post-retrain | Configurado |

**Permisos IAM mínimos requeridos para el usuario/rol:**

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject", "s3:PutObject", "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::insight-commerce-artifacts",
        "arn:aws:s3:::insight-commerce-artifacts/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "ecs:UpdateService", "ecs:DescribeServices"
      ],
      "Resource": "arn:aws:ecs:us-east-2:*:service/insight-commerce-cluster/insight-api-service"
    }
  ]
}
```

---

## Labels de GitHub usados

| Label | Color | Creado por | Descripción |
|-------|-------|------------|-------------|
| `data-drift` | `#e4e669` | `drift-check` | Drift estadístico detectado en chequeo semanal |
| `mlops` | `#0075ca` | `retrain` | Eventos del pipeline MLOps (primer entrenamiento) |
| `rollback` | `#d93f0b` | `retrain`, `deploy` (cd + mlops) | Rollback de modelo o deploy activado |
| `cd` | `#1d76db` | `deploy` (cd.yml) | Eventos del pipeline de entrega continua |
| `deploy` | `#e99695` | `deploy` (cd.yml + mlops.yml) | Deploy fallido a ECS |

Los labels se crean automáticamente si no existen (via `gh label create ... 2>/dev/null || true`).

---

## Tests cubiertos por el pipeline

| Archivo | Qué valida |
|---------|------------|
| `tests/test_feature_engineering.py` | Leakage prior/train, NaN intencionales, segmentación |
| `tests/test_data_loader.py` | Schema 26 cols, nulos, label binario, pares únicos |
| `tests/test_inference.py` | Contrato de features: columnas faltantes, `n_features`, orden |
| `tests/test_train.py` | Split 70/15/15 por usuarios, ausencia de overlap, métricas [0,1] |
| `tests/test_model_monitoring.py` | PSI ≈ 0 con datos estables, `drift_detected=True` con drift real, schema del JSON |
| `tests/test_validate_data.py` | 26 columnas, sin nulos no permitidos, sin duplicados, label binario |
| `tests/test_integration.py` | Pipeline completo con datos sintéticos end-to-end |

---

## Artefactos generados por el pipeline

| Artefacto | Generado por | Consumido por | Retención |
|-----------|-------------|---------------|-----------|
| `coverage.xml` | `test` | SonarCloud (mismo job) | N/A |
| `drift-report-{run_id}` (`drift_report.json`) | `drift-check` | Auditoría / trazabilidad | 90 días |
| `model-artifacts-{run_id}` (`.pkl`, `.json`) | `retrain` | Auditoría / rollback manual | 90 días |
| `monitoring/actual/feature_matrix.parquet` | `build-snapshot` | `drift-check` | S3 (sobrescrito semanalmente) |
| `feature_matrix_reference.parquet` | `retrain` | `drift-check` (siguiente semana) | S3 (sobrescrito por retrain) |
| `model.pkl`, `cluster_models.pkl`, `model_log.json` | `retrain` | ECS tasks al arrancar | S3 (sobrescrito por retrain) |

---

## Ejecutar localmente

```bash
# Tests con cobertura
pytest --cov=src --cov-report=term-missing -v

# Type checking
mypy src/ --ignore-missing-imports --no-strict-optional

# Snapshot semanal (requiere USE_S3=true y AWS credentials)
python -m src.pipeline --snapshot-only

# Drift check (requiere los dos parquets en S3)
USE_S3=true python -m src.model_monitoring

# Pipeline completo de reentrenamiento
USE_S3=true python -m src.pipeline --trials 50
```

---

## Extensiones futuras

- **Blue/Green deploy:** cuando se requiera zero-downtime estricto, migrar a blue/green con AWS CodeDeploy — ver tabla comparativa en sección CD
- **Integration tests de API:** job post-deploy que valide endpoints adicionales más allá de `/health` y `/recommend`
- **ECR lifecycle policy:** limpiar imágenes viejas automáticamente en ECR para controlar costos de almacenamiento
- **Notificaciones Slack:** enviar alertas de drift, rollback y deploy fallido a un canal de equipo