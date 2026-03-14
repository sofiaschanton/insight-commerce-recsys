import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

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
    RecommendResponse,
)


def _build_logger() -> logging.Logger:
    """Configura y devuelve el logger de la API.

    Escribe los logs en un archivo dentro de src/api/reports/logs/api.log.
    El bloque 'if not logger.handlers' evita agregar handlers duplicados
    si la función se llama más de una vez (por ejemplo, en tests o reloads).
    """
    logs_dir = Path(__file__).resolve().parent / "reports" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)  # Crea la carpeta si no existe

    logger = logging.getLogger("api")
    logger.setLevel(logging.INFO)
    logger.propagate = False  # No reenvía logs al logger raíz de Python

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler = logging.FileHandler(logs_dir / "api.log", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Instancias globales: se crean una vez al importar el módulo
logger = _build_logger()
service = RecommendationService()  # El servicio se inicializa en el evento startup

app = FastAPI(
    title="Insight Commerce Recsys API",
    version="1.0.0",
    description="API de recomendaciones Next Basket",
)


@app.on_event("startup")
def startup_event() -> None:
    """Se ejecuta automáticamente cuando arranca el servidor FastAPI.
    Carga el modelo, los artefactos de clustering y abre la conexión a la DB.
    """
    service.startup()
    logger.info("startup completed | model=%s | n_features=%s", service.model_name, service.n_features)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Captura cualquier excepción no controlada y devuelve un 500 genérico.
    Evita exponer detalles internos al cliente; el detalle completo queda en el log.
    """
    logger.exception("internal error | endpoint=%s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Error interno del servidor."},
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Endpoint de salud: confirma que la API está activa y qué modelo tiene cargado.
    Útil para monitoreo y para verificar que el startup fue exitoso.
    """
    logger.info("GET /health")
    return HealthResponse(
        status="ok",
        model=service.model_name,
        n_features=service.n_features,
        artefactos=["model.pkl", "cluster_models.pkl", "model_log.json"],
    )


# IMPORTANTE: este endpoint debe ir ANTES de /recommend/{user_id}.
# FastAPI evalúa las rutas en orden de definición. Si /{user_id} fuera primero,
# la palabra "batch" se interpretaría como un user_id y daría error 422.
@app.post("/recommend/batch", response_model=BatchResponse)
def recommend_batch(payload: BatchRequest) -> BatchResponse:
    """Genera recomendaciones para múltiples usuarios en una sola llamada (hasta 100).

    Para cada usuario devuelve sus recomendaciones o un mensaje de error individual,
    sin que el fallo de un usuario interrumpa el procesamiento de los demás.
    Errores de DB o de contrato de features sí cortan toda la solicitud (503/400).
    """
    logger.info("POST /recommend/batch | n_users=%s", len(payload.user_ids))

    results = []
    for user_id in payload.user_ids:
        try:
            recommendations = service.recommend_user(user_id=user_id, top_k=10)
            results.append(BatchItemResult(user_id=user_id, recommendations=recommendations))
        except UserNotFoundError as err:
            # Usuario no encontrado → se registra en el resultado pero no falla el batch
            results.append(BatchItemResult(user_id=user_id, error=str(err)))
        except FeatureContractError as err:
            raise HTTPException(status_code=400, detail=str(err)) from err
        except DatabaseConnectionError as err:
            raise HTTPException(status_code=503, detail=str(err)) from err

    return BatchResponse(results=results)


@app.post("/recommend/{user_id}", response_model=RecommendResponse)
def recommend_user(user_id: int) -> RecommendResponse:
    """Genera las top-10 recomendaciones para un único usuario.

    Devuelve 404 si el usuario no tiene historial, 400 si hay un problema
    con las features, y 503 si no se puede conectar a la base de datos.
    """
    logger.info("POST /recommend/{user_id} | user_id=%s", user_id)
    try:
        recommendations = service.recommend_user(user_id=user_id, top_k=10)
    except UserNotFoundError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err
    except FeatureContractError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    except DatabaseConnectionError as err:
        raise HTTPException(status_code=503, detail=str(err)) from err

    return RecommendResponse(user_id=user_id, recommendations=recommendations)
