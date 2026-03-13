# MLflow Setup & Workflow Guide
## Sistema de Recomendación de Próxima Compra (Next Basket Recommendation)

---

## 📋 Tabla de Contenidos

1. [Introducción](#introducción)
2. [Prerrequisitos](#prerrequisitos)
3. [Puesta en Marcha del Servidor](#puesta-en-marcha-del-servidor)
4. [Ejecución del Pipeline](#ejecución-del-pipeline)
5. [Interfaz Web de MLflow](#interfaz-web-de-mlflow)
6. [Troubleshooting](#troubleshooting)
7. [Estructura de Artefactos](#estructura-de-artefactos)

---

## 📌 Introducción

### ¿Qué es MLflow?

[MLflow](https://mlflow.org/) es una plataforma de código abierto diseñada para gestionar el ciclo de vida completo del aprendizaje automático. Proporciona herramientas para:

- **Tracking de Experimentos**: Registrar parámetros, métricas y artefactos de cada ejecución.
- **Model Registry**: Gestionar versiones de modelos en un registro centralizado.
- **Reproducibilidad**: Garantizar que cada experimento sea reproducible y auditable.

### Por qué lo usamos en este proyecto

En nuestro proyecto de **Next Basket Recommendation**, utilizamos MLflow para:

1. **Validar múltiples algoritmos**: Comparamos rendimiento entre **LightGBM**, **CatBoost** y un **Baseline de Recomendacion** de manera sistemática.
2. **Rastrear el preprocesamiento**: Registramos los parámetros de **K-Means** (clustering de usuarios y productos) que alimentan los modelos principales.
3. **Gestionar artefactos críticos**: Almacenamos no solo modelos entrenados, sino también diccionarios de hiperparámetros y el archivo `cluster_models.pkl`, esencial para la fase de inferencia.
4. **Facilitar la colaboración**: Todo miembro del equipo puede revisar experimentos previos, reproducirlos y construir sobre versiones anteriores.

---

## 📦 Prerrequisitos

### Instalación de MLflow

Asegúrate de que tu entorno virtual está activado:

```bash
source .venv/bin/activate
```

Luego, instala MLflow ejecutando:

```bash
pip install mlflow
```

Para verificar que la instalación fue exitosa:

```bash
mlflow --version
```

Deberías ver una salida similar a:
```
mlflow, version X.Y.Z
```

### Verificar dependencias adicionales

Si aún no has instalado todas las dependencias, ejecuta:

```bash
pip install -r requirements.txt
```

---

## 🚀 Puesta en Marcha del Servidor

### Levantar el servidor de MLflow

MLflow proporciona una interfaz web para visualizar experimentos. Para iniciar el servidor **en una terminal separada**, ejecuta:

```bash
mlflow ui --host 127.0.0.1 --port 5000
```

**Nota técnica importante:**
- `--host 127.0.0.1`: Vincula el servidor a localhost, previniendo errores de CORS y bloqueos de red.
- `--port 5000`: Define el puerto de escucha (por defecto también es 5000, pero es explícito para evitar conflictos).

### Verificar que el servidor está activo

Abre tu navegador web e ingresa a:

```
http://127.0.0.1:5000
```

Deberías ver la interfaz de MLflow mostrando un panel de inicio (probablemente vacío si es la primera ejecución).

### ¿Dónde se almacenan los registros?

Por defecto, MLflow crea un directorio `mlruns/` en el directorio desde donde ejecutaste el comando. En nuestro proyecto, los artefactos se encuentran en:

```
/home/asus_juan/Documents/GitHub/insight-commerce-recsys/mlartifacts/
```

Este directorio es gestionado por MLflow y contiene subdirectorios para cada experimento.

---

## 🔄 Ejecución del Pipeline

### Garantizar que MLflow está escuchando

Antes de ejecutar el script de entrenamiento, **asegúrate de que el servidor de MLflow está corriendo** en otra terminal. Verifica ejecutando:

```bash
curl http://127.0.0.1:5000
```

Deberías recibir una respuesta HTTP sin errores de conexión.

### Ejecutar el script principal de entrenamiento

En una **terminal diferente** (con el entorno virtual activado), ejecuta:

```bash
python -m src.models.train.py
```

### ¿Qué sucede durante la ejecución?

1. El script lee los datos preprocesados desde `data/processed/`.
2. Entrena modelos de **K-Means** para clustering de usuarios y productos.
3. Entrena el modelo principal (**LightGBM Optimizado**) utilizando los clusters como features.
4. Calcula métricas de negocio: **F1-Score**, **Recall**, **Precision**, **AUC-ROC**.
5. Registra automáticamente en MLflow:
   - **Parámetros**: Hiperparámetros de cada modelo.
   - **Métricas**: Todas las métricas de evaluación.
   - **Artefactos**: Modelos serializados, diccionarios de parámetros y `cluster_models.pkl`.

**Duración estimada**: 5-30 minutos (depende del tamaño del dataset y especificaciones de hardware).

Ademas debes dirigirte a `notebooks/04_modelado.ipynb` donde deberas dirigirte a el comando Run All

### ¿Qué sucede durante la ejecución?

1. El script lee los datos preprocesados desde `data/processed/`.
2. Entrena modelos de **K-Means** para clustering de usuarios y productos.
3. Entrena los modelos que se compararon (**LigthGBM**, **CatBoost**, **Baseline de Recomendacion**) utilizando los clusters como features.
4. Calcula métricas de negocio: **F1-Score**, **Recall**, **Precision**, **AUC-ROC**.
5. Registra automáticamente en MLflow:
   - **Parámetros**: Hiperparámetros de cada modelo.
   - **Métricas**: Todas las métricas de evaluación.
   - **Artefactos**: Modelos serializados, diccionarios de parámetros y `cluster_models.pkl`.

---

## 🎨 Interfaz Web de MLflow

### Navegación principal

Una vez ejecutado el pipeline, accede a `http://127.0.0.1:5000`. La interfaz mostrará:

#### 1. **Experiments**
Verás una lista de experimentos:
- `Next_Basket_Recommendation`
- `Recomendacion_Proxima_Compra`
- `CatBoost Baselines`

Haz clic en cualquier experimento para ver todos los runs (ejecuciones) asociados.

#### 2. **Runs y Métricas**
Para cada run, podrás visualizar:

| Métrica | Descripción |
|---------|-------------|
| **f1_score** | Balance entre precisión y recall; métrica principal de negocio. |
| **recall** | Proporción de compras reales correctamente predichas. |
| **precision** | Proporción de predicciones positivas que fueron correctas. |
| **auc_roc** | Área bajo la curva ROC; robustez ante desbalance de clases. |
| **training_time** | Tiempo de entrenamiento en segundos. |

#### 3. **Parámetros Registrados**
Cada run muestra los hiperparámetros utilizados:
- Para LightGBM: `learning_rate`, `num_leaves`, `max_depth`, etc.
- Para CatBoost: `depth`, `learning_rate`, `iterations`, etc.

#### 4. **Descarga de Artefactos**
En la pestaña "Artifacts", encontrarás:

```
runs/
├── <RUN_ID>/
│   ├── cluster_models/
│   │   ├── kmeans_users.pkl        # Modelo K-Means para usuarios
│   │   ├── kmeans_products.pkl     # Modelo K-Means para productos
│   │   └── cluster_models.pkl      # Referencia centralizada (CRÍTICO para inferencia)
│   ├── models/
│   │   ├── lightgbm_model.pkl      # Modelo LightGBM serializado
│   │   └── catboost_model.pkl      # Modelo CatBoost serializado
│   ├── hyperparams.json            # Diccionario de hiperparámetros
│   ├── metrics.json                # Métricas finales de evaluación
│   └── logs/
│       └── model_log.json          # Log detallado de la ejecución
```

**Descargar artefactos**: Haz clic en el botón "Download" en la sección de Artifacts para bajar los archivos.

### Comparación de Runs

MLflow permite comparar múltiples runs lado a lado:

1. Selecciona al menos 2 runs de la lista.
2. Haz clic en "Compare" en la barra superior.
3. Visualiza diferencias en parámetros y métricas.

---

## 🔧 Troubleshooting

### Problema 1: La interfaz muestra "INTERNAL_ERROR" o no visualiza los runs

**Síntomas:**
- La UI abre pero aparece un mensaje de error.
- Los runs no se muestran en la lista.

**Causa:**
Generalmente, existe un filtro por defecto en la barra de búsqueda (ej: `metrics.rmse < 1`) que excluye tus runs.

**Solución:**
1. Ubica la barra de búsqueda en la parte superior de la página.
2. **Borra cualquier texto en el campo de filtros**.
3. Presiona Enter o haz clic en "Clear Filters".
4. Actualiza la página (Ctrl+R o Cmd+R).

Si el problema persiste, accede directamente al directorio `mlartifacts/` para verificar que contiene directorios de runs.

---

### Problema 2: Error "Connection refused" en la consola de Python

**Síntomas:**
Al ejecutar `python -m src.models.train`, aparece un error como:
```
ConnectionRefusedError: [Errno 111] Connection refused
```
o
```
requests.exceptions.ConnectionError: HTTPConnectionPool(host='127.0.0.1', port=5000): Max retries exceeded
```

**Causa:**
El servidor de MLflow no está corriendo o no está escuchando en el puerto 5000.

**Solución:**
1. **Verifica si el servidor está activo**: En otra terminal, ejecuta:
   ```bash
   curl http://127.0.0.1:5000
   ```
   Si obtienes un error, el servidor no está corriendo.

2. **Inicia el servidor**: En una terminal **separada**, ejecuta:
   ```bash
   mlflow ui --host 127.0.0.1 --port 5000
   ```

3. **Espera a que esté listo**: Verás un mensaje como:
   ```
   [2026-03-13 10:30:45 +0000] [12345] [INFO] Started server process [12345]
   ```

4. **Luego, ejecuta el script de entrenamiento** en otra terminal.

---

### Problema 3: Conflicto de puertos (Puerto 5000 ya está en uso)

**Síntomas:**
```
Address already in use
```
cuando intentas iniciar MLflow.

**Solución:**

**Opción A: Usar un puerto diferente**
```bash
mlflow ui --host 127.0.0.1 --port 5001
```
Luego, accede a `http://127.0.0.1:5001`.

**Opción B: Matar el proceso que usa el puerto 5000**
```bash
lsof -i :5000         # Identifica el proceso
kill -9 <PID>         # Mata el proceso (reemplaza <PID>)
```

---

### Problema 4: Los artefactos no se guardan

**Síntomas:**
El script termina sin errores, pero en la UI no aparecen artefactos bajo "Artifacts".

**Causa:**
El código no está registrando los artefactos explícitamente con `mlflow.log_artifact()`.

**Verificación:**
Abre [src/models/train.py](../src/models/train.py) y busca líneas como:
```python
mlflow.log_artifact(model_path, artifact_path)
```

Si no existen, asegúrate de que el script incluya el registro de artefactos críticos como `cluster_models.pkl`.

---

## 📂 Estructura de Artefactos

### Árbol de directorios en MLflow

```
mlartifacts/
├── 1/                                    # Experiment ID: "LightGBM Baselines"
│   ├── 4c25f95078f340298f691a283c92e4e8/ # Run ID
│   │   └── artifacts/
│   │       ├── cluster_models/
│   │       │   ├── kmeans_users.pkl
│   │       │   ├── kmeans_products.pkl
│   │       │   └── cluster_models.pkl    # CRÍTICO para inferencia
│   │       └── logs/
│   │           └── model_log.json
│   └── aaf8edee91264b0e94eecbd479840e0d/ # Otro Run ID
│       └── artifacts/
│           └── ...
│
├── 4/                                    # Experiment ID: "CatBoost Baselines"
│   └── models/
│       └── m-c15208fd1ec444d288a6f32f99ec9fdb/
│           └── artifacts/
│               ├── conda.yaml            # Dependencias del modelo
│               ├── MLmodel               # Metadata de MLflow
│               ├── model.cb              # Modelo serializado
│               ├── python_env.yaml       # Validación de entorno
│               └── requirements.txt      # Dependencias exactas
```

### Archivos críticos para la inferencia

Antes de usar un modelo en producción, asegúrate de que tienes:

1. **`cluster_models.pkl`**: Contiene los modelos de K-Means necesarios para transformar datos nuevos.
2. **`lightgbm_model.pkl` o `catboost_model.pkl`**: El modelo predictivo principal.
3. **`hyperparams.json`**: Configuración utilizada durante el entrenamiento (útil para reproducibilidad).

---

## 🔗 Referencias Rápidas

### Comandos esenciales

| Acción | Comando |
|--------|---------|
| Activar entorno virtual | `source .venv/bin/activate` |
| Instalar MLflow | `pip install mlflow` |
| Iniciar servidor | `mlflow ui --host 127.0.0.1 --port 5000` |
| Ejecutar pipeline | `python -m src.models.train` |
| Verificar servidor | `curl http://127.0.0.1:5000` |

### URLs importantes

| Recurso | URL |
|---------|-----|
| Interfaz MLflow | `http://127.0.0.1:5000` |
| MLflow Docs | https://mlflow.org/docs/latest/index.html |
| MLflow GitHub | https://github.com/mlflow/mlflow |

### Métodos de MLflow en el código

```python
import mlflow

# Registrar parámetros
mlflow.log_param("learning_rate", 0.05)

# Registrar métricas
mlflow.log_metric("f1_score", 0.87)

# Registrar artefactos
mlflow.log_artifact("path/to/model.pkl", artifact_path="models")

# Iniciar un run
with mlflow.start_run():
    # Tu código de entrenamiento
    pass

# Finalizar un run
mlflow.end_run()
```

---

## Soporte

Si encuentras problemas adicionales no contemplados en este documento:

1. Revisa los logs del servidor MLflow (terminal donde ejecutaste `mlflow ui`).
2. Consulta la [documentación oficial de MLflow](https://mlflow.org/docs/latest/).
3. Verifica que todas las dependencias estén correctamente instaladas: `pip list | grep mlflow`.

---

**Última actualización**: Marzo 2026  
**Versión**: 1.0  
**Estado**: Documentación oficial de producción
