from typing import List, Optional

from pydantic import BaseModel, Field, conlist

# Este archivo define los esquemas de entrada y salida de la API usando Pydantic.
# Pydantic valida automáticamente los tipos de datos: si el cliente manda un valor
# incorrecto, FastAPI devuelve un error 422 antes de llegar al código de negocio.


class RecommendationItem(BaseModel):
    """Un único producto recomendado con su probabilidad de recompra.

    Aparece dentro de las listas de recomendaciones en las respuestas de la API.
    """
    product_key: int              # ID único del producto en la base de datos
    product_name: Optional[str] = None  # Nombre del producto (puede ser None si no está en dim_product)
    probability: float            # Probabilidad de recompra predicha por el modelo (entre 0 y 1)


class RecommendResponse(BaseModel):
    """Respuesta del endpoint POST /recommend/{user_id}.

    Devuelve el user_id solicitado junto con su lista de productos recomendados.
    """
    user_id: int                              # El usuario para el que se generaron las recomendaciones
    recommendations: List[RecommendationItem] # Lista ordenada de productos (de mayor a menor probabilidad)


class BatchRequest(BaseModel):
    """Cuerpo del request para el endpoint POST /recommend/batch.

    El cliente envía una lista de user_ids y recibe recomendaciones para todos a la vez.
    Se acepta un mínimo de 1 y un máximo de 100 usuarios por llamada.
    """
    user_ids: conlist(int, min_length=1, max_length=100) = Field(
        ..., description="Lista de user_id (1 a 100)"
    )


class BatchItemResult(BaseModel):
    """Resultado individual para un usuario dentro de una respuesta batch.

    Si el usuario tiene historial, 'recommendations' tendrá sus productos sugeridos
    y 'error' será None. Si hubo algún problema (ej: usuario no encontrado),
    'recommendations' será None y 'error' explicará qué pasó.
    """
    user_id: int
    recommendations: Optional[List[RecommendationItem]] = None  # None si hubo error para este usuario
    error: Optional[str] = None                                  # Mensaje de error si no se pudo recomendar


class BatchResponse(BaseModel):
    """Respuesta del endpoint POST /recommend/batch.

    Contiene una lista con un BatchItemResult por cada user_id enviado en el request.
    """
    results: List[BatchItemResult]  # Un resultado por cada usuario solicitado


class HealthResponse(BaseModel):
    """Respuesta del endpoint GET /health.

    Permite verificar que la API está activa y qué modelo tiene cargado en memoria.
    """
    status: str         # Estado general de la API (siempre "ok" si responde)
    model: str          # Nombre del modelo cargado (ej: "LightGBM optimizado")
    n_features: int     # Cantidad de features que usa el modelo
    artefactos: List[str]  # Nombres de los archivos de modelo que fueron cargados al iniciar
