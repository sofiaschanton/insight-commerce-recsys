"""
schemas.py — Insight Commerce · Recsys API
Contratos de entrada/salida de la API definidos con Pydantic v2.

Pydantic valida tipos automáticamente: si el cliente envía un valor incorrecto,
FastAPI devuelve HTTP 422 antes de que el código de negocio sea invocado.

Sin cambios de estructura respecto a la versión local.
Compatible con el flujo cloud-native (S3 + RDS en Fargate).
"""

from typing import Annotated, List, Optional

from pydantic import BaseModel, Field


class RecommendationItem(BaseModel):
    """Un producto recomendado con su probabilidad de recompra predicha.

    Aparece dentro de las listas de recomendaciones en todas las respuestas de la API.
    """
    product_key: int
    # Puede ser None si el producto no aparece en dim_product (dato inconsistente en RDS)
    product_name: Optional[str] = None
    # Probabilidad de recompra predicha por LightGBM (escala 0–1)
    probability: float


class RecommendResponse(BaseModel):
    """Respuesta del endpoint POST /recommend/{user_id}.

    Devuelve el user_id solicitado junto con su lista de productos recomendados,
    ordenada de mayor a menor probabilidad de recompra.
    """
    user_id: int
    recommendations: List[RecommendationItem]


class BatchRequest(BaseModel):
    """Cuerpo del request para POST /recommend/batch.

    El cliente envía entre 1 y 100 user_ids y recibe recomendaciones para todos.
    Superar el límite de 100 devuelve HTTP 422 (validación Pydantic).
    """
    user_ids: Annotated[
        List[int],
        Field(min_length=1, max_length=100, description="Lista de user_id (1 a 100)"),
    ]


class BatchItemResult(BaseModel):
    """Resultado individual para un usuario dentro de una respuesta batch.

    Exactamente uno de los dos campos opcionales estará poblado:
      - recommendations: lista de productos si la inferencia fue exitosa.
      - error: mensaje descriptivo si hubo un problema para ese usuario.
    """
    user_id: int
    recommendations: Optional[List[RecommendationItem]] = None
    error: Optional[str] = None


class BatchResponse(BaseModel):
    """Respuesta del endpoint POST /recommend/batch.

    Un BatchItemResult por cada user_id enviado en el request,
    en el mismo orden que el input.
    """
    results: List[BatchItemResult]


class HealthResponse(BaseModel):
    """Respuesta del endpoint GET /health.

    Permite verificar que la API está activa, qué modelo tiene cargado
    y cuántas features está usando. Usado por el health check del ALB.
    """
    status: str            # Siempre "ok" si el endpoint responde
    model: str             # Nombre del modelo cargado desde model_log.json
    n_features: int        # Cantidad de features del modelo
    artefactos: List[str]  # Nombres de los artefactos descargados desde S3
