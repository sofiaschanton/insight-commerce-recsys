# CI/CD — insight-commerce-recsys

Pipeline de integración y entrega continua basado en GitHub Actions.

---

## Estructura del pipeline

```
push (main, develop, feature/**, fix/**) / pull_request (main, develop)
        │
        └─── test ── instala deps → pytest --cov → mypy → SonarCloud scan

push a main / workflow_dispatch
        │
        ├─── data-validation ── validate_data.py → validation_report.json
        │           │ (bloquea retrain si falla)
        └─── retrain (drift-gated, needs: data-validation)
                │
                ├─ 1. model_monitoring.py → drift_report.json
                ├─ 2. lee drift_detected
                ├─ 3. si false → job termina sin entrenar
                ├─ 4. si true  → pipeline.py --trials 50 → rollback check → sube artefactos
                └─── deploy (needs: retrain, solo si retrain exitoso)
                        │
                        ├─ 1. build imagen Docker → push a ECR
                        ├─ 2. actualizar task definition de ECS
                        ├─ 3. ECS rolling deploy (wait-for-service-stability)
                        └─ 4. health check GET /health → rollback automático si falla

schedule (lunes 8am UTC)
        │
        └─── drift_check
                │
                ├─ 1. model_monitoring.py → drift_report.json
                ├─ 2. lee drift_detected
                ├─ 3. si true → dispara retrain via workflow_dispatch (API)
                └─ 4. si true → crea Issue en GitHub con label "data-drift"
```

El job `test` corre en push y pull_request (no en schedule). El job `data-validation` y `retrain` solo corren en push a `main` o ejecución manual — `retrain` requiere que `data-validation` pase. El job `drift_check` solo corre en el trigger semanal.

---

## Triggers

| Evento | Job activado | Branches / condición |
|---|---|---|
| `push` | `test` | `main`, `develop`, `feature/**`, `fix/**` |
| `pull_request` | `test` | `main`, `develop` |
| `push` a `main` | `data-validation` → `retrain` → `deploy` | rama `main` |
| `workflow_dispatch` | `data-validation` → `retrain` → `deploy` | manual desde GitHub UI |
| `schedule` (cron) | `drift_check` | lunes 8am UTC |

---

## Jobs

### 1. `test` — Tests + type check + cobertura

**Runner:** `ubuntu-latest` · **Python:** `3.10`

| Paso | Descripción |
|---|---|
| `actions/checkout@v4` | Clona el repositorio |
| `actions/setup-python@v5` | Instala Python 3.10 con caché de pip |
| Install dependencies | `pip install -r requirements.txt` + `pytest pytest-cov mypy` |
| Run tests | `pytest --cov=src --cov-report=xml` — cobertura sobre todo `src/` |
| Type check (mypy) | `mypy src/ --ignore-missing-imports --no-strict-optional` — último paso antes de SonarCloud |
| SonarCloud Scan | Análisis estático + cobertura. Consume `coverage.xml` generado en el paso anterior. |

El reporte de cobertura se genera en `coverage.xml` (formato Cobertura, compatible con SonarCloud). mypy corre como penúltimo paso dentro de este mismo job, sin ser un job separado.

---

### 2. `data-validation` — Validación del feature matrix

**Runner:** `ubuntu-latest`
**Trigger:** push a `main` o `workflow_dispatch` (mismo que `retrain`)

Corre `src/data/validate_data` sobre el feature matrix antes del entrenamiento. Si el parquet no existe (primer run del pipeline), el script termina con código 0 (skip) sin bloquear. Si existe y hay errores, el job falla y **bloquea `retrain`**.

| Paso | Descripción |
|---|---|
| Run data validation | `python -m src.data.validate_data` → escribe `reports/data/validation_report.json` |
| Upload validation report | Sube el reporte como artefacto (retención 7 días), siempre (`if: always()`) |

**Validaciones ejecutadas** (definidas en `src/data/validate_data.py`):
- Todas las 26 columnas del contrato presentes (`FEATURE_MATRIX_COLUMNS`)
- Sin nulos en columnas no-nulables (`NON_NULLABLE_COLUMNS`)
- Sin pares `(user_key, product_key)` duplicados
- `label` solo contiene 0 o 1
- `up_times_purchased > 0`
- `user_total_orders > 0`

---

### 3. `retrain` — Reentrenamiento condicional por drift

**Runner:** `ubuntu-latest`
**Trigger:** push a `main`, `workflow_dispatch` (manual o disparado por `drift_check`)
**Depende de:** `data-validation` (si falla, `retrain` no corre)

El entrenamiento **solo se ejecuta si hay drift detectado**. Al final del entrenamiento, el mecanismo de rollback compara el F1 nuevo con el anterior: si el F1 nuevo es más de 5% inferior, los artefactos no se sobreescriben y el job falla con un mensaje claro.

| Paso | Descripción |
|---|---|
| Run model monitoring | Corre `src/model_monitoring.py` → genera `drift_report.json` (PSI + KS). Si no hay datos de referencia, el script termina sin error y el JSON no se escribe. |
| Check drift | Lee `drift_report.json`. Si no existe o `drift_detected` es `false`, el job termina sin entrenar. |
| Run pipeline (Optuna 50 trials) | Solo si `drift_detected` es `true`. Corre `python -m src.pipeline --trials 50`. Incluye validación en memoria y rollback check. |
| Upload model artifacts | Sube `models/model.pkl`, `models/cluster_models.pkl` y `models/model_log.json` (retención 30 días). Solo si hubo reentrenamiento sin rollback. |

**Umbrales de drift** (definidos en `src/model_monitoring.py`):
- PSI ≥ 0.25 → drift significativo
- KS ≥ 0.30 → diferencia estadística significativa
- Basta con que uno supere el umbral para que `drift_detected = true`

**Rollback** (definido en `src/models/train.py`):
- Si `models/model_log.json` existe, se compara F1 nuevo vs F1 anterior
- Si `F1_nuevo < F1_anterior × 0.95` → `RuntimeError`, job falla, artefactos no sobreescritos

---

### 4. `drift_check` — Chequeo semanal de drift

**Runner:** `ubuntu-latest`
**Trigger:** `schedule` — cron `0 8 * * 1` (lunes 8am UTC)

Corre automáticamente cada lunes para detectar si la distribución de los datos de producción se alejó de los datos de referencia. Si se detecta drift, dispara el job `retrain` vía `workflow_dispatch` y crea un Issue en GitHub con label `data-drift`.

| Paso | Descripción |
|---|---|
| Run model monitoring | Corre `src/model_monitoring.py` → genera `drift_report.json` (PSI + KS) |
| Check drift | Lee `drift_report.json`. Si no existe o `drift_detected` es `false`, el job termina sin hacer nada. |
| Upload drift report | Sube `drift_report.json` como artefacto (retención 30 días). Solo si drift detectado. |
| Trigger retrain | Solo si `drift_detected` es `true`: llama a la API de GitHub para disparar `workflow_dispatch` sobre `main`. |
| Create GitHub Issue | Solo si `drift_detected` es `true`: crea Issue con título `[drift] Data drift detectado — YYYY-MM-DD`, body con PSI y KS, y label `data-drift` (se crea si no existe). |

> **¿Por qué chequeo semanal?** Los patrones de compra en supermercados online cambian lentamente — estacionalidad, promociones, cambios de catálogo — a diferencia de sistemas financieros o de fraude donde el drift puede ocurrir en horas. Un chequeo semanal es suficiente para detectar cambios de distribución relevantes sin incurrir en costos de cómputo innecesarios.

---

### 5. `deploy` — Deploy a ECS Fargate

**Runner:** `ubuntu-latest`
**Trigger:** post-retrain exitoso — solo corre si el job `retrain` terminó con `result == 'success'`
**Depende de:** `retrain`

El deploy **no se dispara en cada merge a `main`**. Solo ocurre cuando hay un modelo nuevo entrenado, es decir, cuando `retrain` completó exitosamente (drift detectado o `workflow_dispatch`). Esto evita redeploys innecesarios cuando no hay cambio de modelo.

| Paso | Descripción |
|---|---|
| Configure AWS credentials | Autentica con IAM usando `aws-actions/configure-aws-credentials@v4` y los secrets `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` |
| Login to Amazon ECR | Login al registry privado con `aws-actions/amazon-ecr-login@v2` |
| Build & push Docker image | Construye la imagen de la API desde el `Dockerfile` del repo y la sube a ECR con tag `${{ github.sha }}` |
| Update ECS task definition | Descarga la task definition actual, inyecta la nueva imagen y genera la versión actualizada con `aws-actions/amazon-ecs-render-task-definition@v1` |
| Deploy to ECS (rolling) | Registra la nueva task definition y actualiza el servicio ECS con `aws-actions/amazon-ecs-deploy-task-definition@v1`. Espera hasta que el servicio esté estable (`wait-for-service-stability: true`). Si las tasks nuevas no pasan el health check del ALB, ECS revierte automáticamente. |
| Health check | Llama a `GET /health` sobre el endpoint del ALB. Si responde HTTP distinto de 200, el step falla y queda registro del error en el log. |

El modelo vive en S3 (`insight-commerce-artifacts`) y la API lo descarga en el startup del contenedor — no se embebe en la imagen Docker.

**Placeholders a completar cuando Fargate esté configurado:**

| Variable | Dónde se usa | Descripción |
|---|---|---|
| `ECS_CLUSTER` | step deploy | Nombre del cluster ECS Fargate |
| `ECS_SERVICE` | step deploy y describe task-definition | Nombre del servicio ECS |
| `ECR_REPOSITORY` | step build & push | Nombre del repositorio en ECR |
| `CONTAINER_NAME` | step render task-definition | Nombre del contenedor dentro de la task definition |

---

## CD — Continuous Delivery

### Estrategia elegida: Rolling Update con ECS Fargate + ALB

La API corre en ECS Fargate dentro de una VPC privada, expuesta al exterior a través de un Application Load Balancer (ALB). La estrategia de deploy es **rolling update nativo de ECS**: ECS reemplaza gradualmente las tasks con la versión nueva, drena el tráfico de las tasks viejas a través del ALB y revierte automáticamente si las tasks nuevas no superan el health check configurado en el target group.

**Flujo del job `deploy`:**

```
retrain exitoso
       │
       └─ build imagen Docker (tag: ${{ github.sha }})
               │
               └─ push a ECR (insight-commerce-artifacts region)
                       │
                       └─ render nueva task definition (imagen actualizada)
                               │
                               └─ ECS rolling deploy (wait-for-service-stability)
                                       │
                               ┌───────┴───────┐
                          tasks OK          tasks fallan
                               │                │
                        health check        rollback automático
                        GET /health         (ECS revierte a versión anterior)
                               │
                        HTTP 200 → OK
```

**¿Por qué el deploy no corre en cada merge a `main`?**
El modelo de recomendación no cambia con cambios de código que no impliquen reentrenamiento. Un merge de código sin drift no produce un artefacto nuevo en S3 — forzar un redeploy en ese caso sería un deploy vacío. El trigger correcto es un modelo nuevo, no un commit nuevo.

---

### Rolling Update vs Blue/Green

| Criterio | Rolling Update (implementado) | Blue/Green (mejora futura) |
|---|---|---|
| **Costo** | Sin costo adicional — mismo número de tasks Fargate durante el deploy | Alto — duplica el costo de ECS Fargate durante ~24hs (dos entornos completos activos en paralelo) |
| **Complejidad** | Baja — nativo en ECS, sin infraestructura extra | Alta — requiere AWS CodeDeploy + reglas en el ALB + grupos de targets adicionales |
| **Overlap de instancias** | Momentáneo — ECS levanta tasks nuevas antes de bajar las viejas | Total — environment blue y green completos activos en simultáneo hasta confirmar el cutover |
| **Tiempo de rollback** | Automático y rápido — ECS revierte si el health check falla durante el deploy | Inmediato — switch de tráfico en el ALB (0 downtime), pero requiere aprobación manual o gate automático |
| **Visibilidad del estado** | Limitada — solo logs de ECS y métricas del ALB | Alta — CodeDeploy registra cada fase del deployment con aprobaciones y hooks |
| **Adecuado para** | Proyectos con SLA flexible, equipo pequeño, bajo presupuesto | Producción con SLA estricto (< 1s de downtime), tráfico crítico, requisitos de auditoría |

**Blue/Green como mejora futura recomendada:** para un entorno de producción con SLA estricto (e-commerce con tráfico real), blue/green es la estrategia ideal. Requiere integrar AWS CodeDeploy al pipeline, configurar dos target groups en el ALB y definir un gate de validación (health check + smoke tests) antes de redirigir el 100% del tráfico. El costo de duplicar Fargate durante las primeras 24hs post-deploy es el trade-off aceptable para garantizar rollback instantáneo sin impacto en usuarios.

---

### SonarCloud — Análisis estático

Corre dentro del job `test` como último paso (no es un job separado).

La configuración del proyecto está en [`sonar-project.properties`](../sonar-project.properties):

```properties
sonar.projectKey=sofiaschanton_insight-commerce-recsys
sonar.organization=sofiaschanton
sonar.sources=src
sonar.tests=tests
sonar.python.coverage.reportPaths=coverage.xml
sonar.exclusions=**/__pycache__/**,**/*.pyc,data/**,models/**,notebooks/**,reports/**
```

---

## Setup inicial (una sola vez)

### 1. Agregar `SONAR_TOKEN` al repositorio

1. Ir a [sonarcloud.io](https://sonarcloud.io) → tu organización → **Security** → generar token
2. En GitHub: **Settings → Secrets and variables → Actions → New repository secret**
3. Nombre: `SONAR_TOKEN` · Valor: el token generado

`GITHUB_TOKEN` es automático — GitHub lo provee en cada ejecución.

### 2. Habilitar el proyecto en SonarCloud

1. En SonarCloud: **+** → **Analyze new project** → seleccionar `insight-commerce-recsys`
2. Elegir **GitHub Actions** como método de CI
3. Desactivar el análisis automático de SonarCloud (para que no colisione con el workflow)

---

## Tests cubiertos por el pipeline

| Archivo | Qué valida |
|---|---|
| `tests/test_feature_engineering.py` | Leakage prior/train, separación `eval_set`, label solo desde train, NaN intencionales |
| `tests/test_data_loader.py` | Schema del parquet (26 cols), nulos, label binario, pares únicos, rangos de valores |
| `tests/test_inference.py` | Contrato de features: columnas faltantes, `n_features` mismatch, orden correcto |
| `tests/test_train.py` | Split 70/15/15 por usuarios, ausencia de overlap entre conjuntos, claves y rango [0,1] de métricas |
| `tests/test_model_monitoring.py` | PSI ≈ 0 con datos estables, `drift_detected=True` con distribución muy desplazada, estructura del reporte JSON |
| `tests/test_integration.py` | Pipeline completo con datos sintéticos: schema de 26 cols, ausencia de leakage user-level, entrenamiento end-to-end |

`test_data_loader.py` se **salta automáticamente** en CI si `data/processed/feature_matrix.parquet` no está commiteado (usa `pytest.skip`). Los demás tests usan datos sintéticos o mocks — no requieren DB ni archivos `.pkl`.

---

## Ejecutar localmente

```bash
# Tests con cobertura
pytest --cov=src --cov-report=term-missing -v

# Type checking
mypy src/ --ignore-missing-imports --no-strict-optional

# Validar feature matrix (si existe)
python -m src.data.validate_data
```

---

## Artefactos generados por el pipeline

| Artefacto | Generado por | Consumido por | Retención |
|---|---|---|---|
| `coverage.xml` | job `test` | SonarCloud (mismo job) | N/A |
| Reporte SonarCloud | job `test` (paso SonarCloud) | Dashboard en sonarcloud.io | N/A |
| `reports/data/validation_report.json` | job `data-validation` | Revisión manual / auditoría | 7 días |
| `drift_report.json` | job `drift_check` / job `retrain` | Trazabilidad de drift | 30 días |
| `models/model.pkl`, `cluster_models.pkl`, `model_log.json` | job `retrain` | Job `deploy` (vía S3 en startup del contenedor) | 30 días |
| Imagen Docker de la API | job `deploy` | ECS Fargate | ECR (sin expiración — gestionar lifecycle policy) |

---

## Extensiones futuras

- **Blue/Green deploy:** reemplazar el rolling update por blue/green con AWS CodeDeploy para entornos de producción con SLA estricto (ver tabla comparativa en la sección CD)
- **Integration tests de API:** job separado que valide los endpoints `/recommend` contra el entorno de staging antes del cutover a producción
- **ECR lifecycle policy:** configurar política de retención en ECR para limpiar imágenes viejas automáticamente y controlar costos de almacenamiento