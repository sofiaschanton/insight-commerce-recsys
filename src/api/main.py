"""
main.py — Insight Commerce · Recsys API
Versión cloud-native para AWS Fargate + CloudWatch Logs.

Cambios en esta versión (production-ready):
  - /health fortalecido: valida que service.engine puede ejecutar SELECT 1 en RDS.
    Si la DB está caída, el health check retorna 503 — el ALB detiene el tráfico
    hacia esa tarea y ECS la reinicia automáticamente.
  - Logging exclusivo a stdout (StreamHandler) — sin FileHandler ni makedirs.
    CloudWatch captura stdout/stderr automáticamente vía el awslogs log driver.
  - Latencia logueada en /recommend/{user_id} con time.perf_counter().
  - Sin on_event("startup") deprecado — usa lifespan context manager (FastAPI 0.93+).
"""

import logging
import sys
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from src.api.inference import (
    DatabaseConnectionError,
    FeatureContractError,
    RecommendationService,
    UserNotFoundError,
)
from src.api.schemas import (
    BatchItemResult,
    BatchRequest,
    BatchResponse,
    HealthResponse,
    RecommendationItem,
    RecommendResponse,
)


# ---------------------------------------------------------------------------
# Logging → stdout (CloudWatch compatible)
#
# StreamHandler(sys.stdout) es el único handler registrado.
# CloudWatch Logs captura stdout del contenedor automáticamente
# gracias al awslogs log driver configurado en la Task Definition.
# Sin FileHandler, sin os.makedirs, sin rutas locales.
# ---------------------------------------------------------------------------

def _build_logger() -> logging.Logger:
    """Configura el logger de la API con salida exclusiva a stdout.

    El bloque 'if not logger.handlers' evita handlers duplicados si el módulo
    se importa múltiples veces (tests, hot-reload de uvicorn en desarrollo).
    """
    logger = logging.getLogger("api")
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Evita duplicados hacia el logger raíz de Python

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        )
        logger.addHandler(handler)

    return logger


# ---------------------------------------------------------------------------
# Instancias globales
# ---------------------------------------------------------------------------

logger  = _build_logger()
service = RecommendationService()  # Los artefactos se cargan en el lifespan


# ---------------------------------------------------------------------------
# Lifespan (reemplaza el @app.on_event("startup") deprecado desde FastAPI 0.93)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Descarga artefactos desde S3 y abre el pool de conexiones a RDS al arrancar."""
    service.startup()
    logger.info(
        "startup OK | model=%s | n_features=%d",
        service.model_name,
        service.n_features,
    )
    yield
    # Teardown: el pool de SQLAlchemy se cierra automáticamente al destruir el engine.
    logger.info("shutdown | cerrando pool de conexiones RDS")
    if service.engine:
        service.engine.dispose()


app = FastAPI(
    title="Insight Commerce Recsys API",
    version="1.0.0",
    description="API de recomendaciones Next Basket — AWS Fargate",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Manejadores de errores globales
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Captura cualquier excepción no controlada y devuelve HTTP 500.

    El stack trace completo queda en CloudWatch vía stdout.
    Al cliente solo se expone un mensaje genérico (sin detalles internos).
    """
    logger.exception("error interno | path=%s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Error interno del servidor."},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health check fortalecido: valida conectividad real a RDS con SELECT 1.

    Dos niveles de validación:
      1. Los artefactos están cargados en memoria (model_name != "LightGBM" vacío).
      2. El engine puede ejecutar SELECT 1 en RDS PostgreSQL.

    Si el engine no puede conectarse:
      - Se loguea el error en CloudWatch.
      - Se retorna HTTP 503 con detalle del problema.
      - El Target Group del ALB marca la tarea como unhealthy → ECS la reemplaza.

    El ALB debe configurarse con:
      HealthCheckPath: /health
      HealthyHttpCodes: 200
      UnhealthyThresholdCount: 2
    """
    logger.info("GET /health")

    # Validación 1: artefactos cargados
    if service._artifacts is None:
        logger.error("health FAIL | artefactos no cargados")
        raise HTTPException(
            status_code=503,
            detail="Los artefactos del modelo no están disponibles.",
        )

    # Validación 2: conectividad a RDS con SELECT 1
    if service.engine is None:
        logger.error("health FAIL | engine RDS no inicializado")
        raise HTTPException(
            status_code=503,
            detail="El engine de base de datos no está inicializado.",
        )

    try:
        with service.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except OperationalError as err:
        logger.error("health FAIL | RDS no responde | %s", err)
        raise HTTPException(
            status_code=503,
            detail=(
                f"No se pudo conectar a PostgreSQL "
                f"(host={service._db_host}, sslmode={service._db_sslmode}). "
                "Verificar que el RDS está disponible y el Security Group permite "
                "tráfico desde el Security Group de la tarea Fargate."
            ),
        ) from err

    logger.info("health OK | model=%s | rds=%s", service.model_name, service._db_host)
    return HealthResponse(
        status="ok",
        model=service.model_name,
        n_features=service.n_features,
        artefactos=["model.pkl", "cluster_models.pkl", "model_log.json"],
    )


# IMPORTANTE: /recommend/batch debe definirse ANTES de /recommend/{user_id}.
# FastAPI evalúa rutas en orden de definición. Si /{user_id} fuera primero,
# la cadena "batch" se interpretaría como un user_id entero y daría HTTP 422.
@app.post("/recommend/batch", response_model=BatchResponse)
def recommend_batch(payload: BatchRequest) -> BatchResponse:
    """Genera recomendaciones para hasta 100 usuarios en una sola llamada.

    Diseño fault-tolerant por usuario:
      - UserNotFoundError de un usuario individual se registra como campo 'error'
        en BatchItemResult sin interrumpir el procesamiento del resto.
      - FeatureContractError y DatabaseConnectionError son problemas sistémicos
        que cortan toda la solicitud con HTTP 400 / 503.

    El cliente identifica fallos individuales inspeccionando el campo 'error'
    de cada BatchItemResult (es None cuando la inferencia fue exitosa).
    """
    logger.info("POST /recommend/batch | n_users=%d", len(payload.user_ids))

    results = []
    for user_id in payload.user_ids:
        try:
            recs = service.recommend_user(user_id=user_id, top_k=10)
            results.append(
                BatchItemResult(
                    user_id=user_id,
                    recommendations=[RecommendationItem(**r) for r in recs],
                )
            )
        except UserNotFoundError as err:
            logger.warning("user_not_found | user_id=%d | %s", user_id, err)
            results.append(BatchItemResult(user_id=user_id, error=str(err)))
        except FeatureContractError as err:
            raise HTTPException(status_code=400, detail=str(err)) from err
        except DatabaseConnectionError as err:
            raise HTTPException(status_code=503, detail=str(err)) from err

    return BatchResponse(results=results)


@app.post("/recommend/{user_id}", response_model=RecommendResponse)
def recommend_user(user_id: int) -> RecommendResponse:
    """Genera las top-10 recomendaciones de next-basket para un único usuario.

    Códigos de respuesta:
      200 — Recomendaciones generadas correctamente.
      404 — El usuario no tiene historial prior en RDS.
      400 — El contrato de features del modelo fue violado.
      503 — No se pudo conectar a PostgreSQL en RDS.
      500 — Error interno no clasificado (ver CloudWatch Logs).
    """
    t0 = time.perf_counter()
    logger.info("POST /recommend/%d", user_id)

    try:
        recs = service.recommend_user(user_id=user_id, top_k=10)
        recommendations = [RecommendationItem(**r) for r in recs]
    except UserNotFoundError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err
    except FeatureContractError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    except DatabaseConnectionError as err:
        raise HTTPException(status_code=503, detail=str(err)) from err

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "recommend OK | user_id=%d | n_recs=%d | elapsed_ms=%.1f",
        user_id, len(recommendations), elapsed_ms,
    )
    return RecommendResponse(user_id=user_id, recommendations=recommendations)