"""
inference.py — Insight Commerce · Recsys API
Entorno destino: AWS Fargate (ECS) + S3 + RDS PostgreSQL
Autenticación  : IAM Task Role (ecsTaskRole.InsightCommerce) — sin hardcoded keys
Artefactos     : descargados desde S3 a /tmp/ al iniciar el contenedor

Cambios en esta versión (production-ready):
  - Try/except granular por archivo en _download_artifacts_from_s3():
    si falla la descarga de model.pkl se loguea exactamente qué archivo y por qué.
  - Variables de entorno de RDS estandarizadas: DB_HOST, DB_USER, DB_PASSWORD, DB_NAME.
  - Acceso explícito a cluster_models.pkl: cada clave del dict se asigna
    individualmente a atributos tipados para detectar KeyError en startup.
  - Sin Path(__file__), sin load_dotenv(), sin rutas locales de modelos.
"""

import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List

import boto3
import joblib
import pandas as pd
from botocore.exceptions import BotoCoreError, ClientError
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.exc import OperationalError

from src.features.feature_engineering import (
    _normalize_dim_product,
    get_user_aisle_feature,
    get_user_department_feature,
    get_user_features,
    get_user_product_features,
)

# Features base para K-Means de usuario
USER_CLUSTER_FEATURES = [
    "user_total_orders",
    "user_avg_basket_size",
    "user_days_since_last_order",
    "user_reorder_ratio",
    "user_distinct_products",
]

# Features base para K-Means de producto
PRODUCT_CLUSTER_FEATURES = [
    "product_total_purchases",
    "product_reorder_rate",
    "product_avg_add_to_cart",
    "product_unique_users",
    "p_department_reorder_rate",
]

logger = logging.getLogger("api")

# ---------------------------------------------------------------------------
# Configuración S3 — exclusivamente desde variables de entorno del sistema.
# Definidas en la ECS Task Definition; nunca en código ni en archivos locales.
# ---------------------------------------------------------------------------
S3_BUCKET: str = os.environ["S3_BUCKET"]          # Obligatorio — falla rápido si falta
S3_PREFIX: str = os.getenv("S3_PREFIX", "models") # Prefijo dentro del bucket (ej. "models/v2")

# /tmp/ es el ÚNICO directorio escribible en el sistema de archivos de Fargate.
# Todos los artefactos descargados desde S3 se almacenan aquí.
_TMP               = "/tmp"
_MODEL_LOCAL       = f"{_TMP}/model.pkl"
_CLUSTER_LOCAL     = f"{_TMP}/cluster_models.pkl"
_MODEL_LOG_LOCAL   = f"{_TMP}/model_log.json"

# Mínimo de órdenes prior para aplicar el modelo ML completo.
# Por debajo de este umbral se activa el cold-start (ranking por frecuencia personal).
MIN_ORDERS_FOR_MODEL = 5


# ---------------------------------------------------------------------------
# Excepciones de dominio
# ---------------------------------------------------------------------------

class UserNotFoundError(Exception):
    """El user_id no tiene historial prior en RDS (0 órdenes prior)."""


class FeatureContractError(Exception):
    """La matriz de features no coincide con el contrato del modelo entrenado."""


class DatabaseConnectionError(Exception):
    """No fue posible establecer conexión con PostgreSQL en RDS."""


# ---------------------------------------------------------------------------
# Contenedor de artefactos con tipado explícito de claves de cluster_models
# ---------------------------------------------------------------------------

@dataclass
class _ClusterArtifacts:
    """Desempaqueta las claves del dict cluster_models.pkl con tipado explícito.

    Al asignar cada clave en el constructor se obtiene un KeyError descriptivo
    en startup si el pkl no tiene la estructura esperada, en lugar de un
    AttributeError críptico durante la primera inferencia en producción.
    """
    scaler_user:      Any  # StandardScaler ajustado sobre USER_CLUSTER_FEATURES
    kmeans_user:      Any  # KMeans entrenado sobre perfiles de usuario
    user_profiles:    Any  # DataFrame / índice de referencia de usuarios conocidos
    scaler_product:   Any  # StandardScaler ajustado sobre PRODUCT_CLUSTER_FEATURES
    kmeans_product:   Any  # KMeans entrenado sobre perfiles de producto
    product_profiles: Any  # DataFrame / índice de referencia de productos conocidos

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "_ClusterArtifacts":
        """Construye la instancia desde el dict cargado de cluster_models.pkl.

        Lanza KeyError con el nombre exacto de la clave faltante si el pkl
        no tiene la estructura esperada — falla en startup, no en inferencia.
        """
        required_keys = [
            "scaler_user", "kmeans_user", "user_profiles",
            "scaler_product", "kmeans_product", "product_profiles",
        ]
        missing = [k for k in required_keys if k not in d]
        if missing:
            raise KeyError(
                f"cluster_models.pkl no contiene las claves requeridas: {missing}. "
                f"Claves presentes: {list(d.keys())}"
            )
        return cls(
            scaler_user      = d["scaler_user"],
            kmeans_user      = d["kmeans_user"],
            user_profiles    = d["user_profiles"],
            scaler_product   = d["scaler_product"],
            kmeans_product   = d["kmeans_product"],
            product_profiles = d["product_profiles"],
        )


@dataclass
class LoadedArtifacts:
    """Agrupa los tres artefactos cargados desde S3 al inicio del servicio."""
    model:    Any               # Modelo LightGBM serializado con joblib
    clusters: _ClusterArtifacts # Scalers + KMeans con tipado explícito
    model_log: Dict[str, Any]   # Metadatos: feature_cols, n_features, AUC, etc.


# ---------------------------------------------------------------------------
# Servicio principal
# ---------------------------------------------------------------------------

class RecommendationService:
    """Orquesta la descarga desde S3, la conexión a RDS y la inferencia."""

    def __init__(self) -> None:
        # Metadatos del modelo — se populan en startup() tras leer model_log.json
        self.model_name: str  = "LightGBM"
        self.n_features: int  = 0
        self.feature_cols: List[str] = []

        # Estado interno — se inicializan en startup()
        self.engine   = None    # Expuesto como atributo público para el health check de RDS
        self._db_host: str     = ""
        self._db_sslmode: str  = ""
        self._artifacts: LoadedArtifacts | None = None

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def startup(self) -> None:
        """Carga artefactos desde S3 y abre el pool de conexiones a RDS.

        Orden de operaciones:
          1. Descarga los tres artefactos de S3 a /tmp/ (con try/except granular)
          2. Los carga en memoria y desempaqueta cluster_models con tipado explícito
          3. Abre el pool de conexiones a RDS PostgreSQL
          4. Sincroniza metadatos desde model_log.json

        Si cualquier paso falla, el proceso arroja una excepción que Fargate/ECS
        registra en CloudWatch y reinicia la tarea automáticamente.
        """
        self._artifacts = self._download_and_load_artifacts()
        self.engine     = self._build_engine()

        self.model_name   = str(self._artifacts.model_log.get("model_name",  self.model_name))
        self.n_features   = int(self._artifacts.model_log.get("n_features",  0))
        self.feature_cols = list(self._artifacts.model_log.get("feature_cols", []))

        logger.info(
            "startup OK | model=%s | n_features=%d | bucket=%s | prefix=%s",
            self.model_name, self.n_features, S3_BUCKET, S3_PREFIX,
        )

    # ------------------------------------------------------------------
    # Descarga de artefactos desde S3
    # ------------------------------------------------------------------

    def _s3_key(self, filename: str) -> str:
        """Construye la clave S3: prefijo/filename o solo filename si el prefijo está vacío."""
        return f"{S3_PREFIX}/{filename}" if S3_PREFIX else filename

    def _download_and_load_artifacts(self) -> LoadedArtifacts:
        """Descarga los tres artefactos desde S3 a /tmp/ y los carga en memoria.

        Try/except granular por archivo:
          - Cada descarga está aislada en su propio bloque.
          - Si falla, el log en CloudWatch indica EXACTAMENTE qué archivo falló
            y la causa (permisos IAM, nombre incorrecto, bucket inexistente, etc.).
          - La excepción se re-lanza para que Fargate reinicie la tarea.

        boto3 obtiene credenciales automáticamente desde el IAM Task Role
        (ecsTaskRole.InsightCommerce) vía el metadata endpoint de ECS (169.254.170.2).
        No se pasa ninguna access_key ni secret_key en el código.
        """
        s3 = boto3.client("s3")  # Credenciales del Task Role — sin hardcoding

        artifacts_to_download = [
            ("model.pkl",          _MODEL_LOCAL),
            ("cluster_models.pkl", _CLUSTER_LOCAL),
            ("model_log.json",     _MODEL_LOG_LOCAL),
        ]

        for filename, local_path in artifacts_to_download:
            s3_key = self._s3_key(filename)
            try:
                logger.info(
                    "S3 download | bucket=%s | key=%s → %s",
                    S3_BUCKET, s3_key, local_path,
                )
                s3.download_file(S3_BUCKET, s3_key, local_path)
                logger.info("S3 download OK | %s", local_path)

            except ClientError as err:
                # ClientError cubre: NoSuchKey, NoSuchBucket, AccessDenied, etc.
                error_code = err.response.get("Error", {}).get("Code", "UNKNOWN")
                error_msg  = err.response.get("Error", {}).get("Message", str(err))
                logger.critical(
                    "S3 download FAILED | bucket=%s | key=%s | "
                    "error_code=%s | message=%s | "
                    "Verificar: (1) el archivo existe en S3, "
                    "(2) ecsTaskRole.InsightCommerce tiene s3:GetObject sobre el bucket.",
                    S3_BUCKET, s3_key, error_code, error_msg,
                )
                raise  # Re-lanza para que ECS registre el Exit Code y reinicie la tarea

            except BotoCoreError as err:
                # BotoCoreError: errores de red, timeout, credenciales no resueltas
                logger.critical(
                    "S3 download FAILED (network/credentials) | bucket=%s | key=%s | %s | "
                    "Verificar: (1) el Task Role tiene s3:GetObject, "
                    "(2) el contenedor puede alcanzar el endpoint de S3 (VPC endpoint o NAT).",
                    S3_BUCKET, s3_key, err,
                )
                raise

            except OSError as err:
                # OSError: no se puede escribir en local_path (no debería ocurrir con /tmp/)
                logger.critical(
                    "S3 download FAILED (write error) | local_path=%s | %s | "
                    "Asegúrate de que local_path está bajo /tmp/ (único dir escribible en Fargate).",
                    local_path, err,
                )
                raise

        # --- Carga en memoria ---
        logger.info("Cargando artefactos en memoria desde /tmp/...")

        model = joblib.load(_MODEL_LOCAL)
        logger.info("model.pkl cargado | tipo=%s", type(model).__name__)

        raw_cluster_dict = joblib.load(_CLUSTER_LOCAL)
        # Desempaqueta con tipado explícito — lanza KeyError si falta alguna clave
        clusters = _ClusterArtifacts.from_dict(raw_cluster_dict)
        logger.info(
            "cluster_models.pkl cargado | "
            "scaler_user=%s | kmeans_user=%s | scaler_product=%s | kmeans_product=%s",
            type(clusters.scaler_user).__name__,
            type(clusters.kmeans_user).__name__,
            type(clusters.scaler_product).__name__,
            type(clusters.kmeans_product).__name__,
        )

        with open(_MODEL_LOG_LOCAL, "r", encoding="utf-8") as fh:
            model_log = json.load(fh)
        logger.info(
            "model_log.json cargado | n_features=%s | feature_cols_count=%d",
            model_log.get("n_features", "N/A"),
            len(model_log.get("feature_cols", [])),
        )

        return LoadedArtifacts(model=model, clusters=clusters, model_log=model_log)

    # ------------------------------------------------------------------
    # Conexión a RDS PostgreSQL
    # ------------------------------------------------------------------

    def _build_engine(self):
        """Crea el engine de SQLAlchemy apuntando a RDS PostgreSQL.

        Variables de entorno estandarizadas (definidas en la Task Definition de ECS):
          DB_HOST     — endpoint del cluster RDS (obligatorio)
          DB_PORT     — puerto (default: 5432)
          DB_USER     — usuario de la base de datos (obligatorio)
          DB_PASSWORD — contraseña (obligatorio; usar Secrets Manager en producción)
          DB_NAME     — nombre de la base de datos (obligatorio)
          DB_SSLMODE  — modo SSL (default: "require"; Fargate → RDS siempre require)

        Ajuste automático de sslmode:
          Fargate → RDS: "require" es válido y recomendado siempre.
          localhost / 127.0.0.1: se baja a "prefer" para evitar errores de certificado
          en entornos de desarrollo local.
        """
        host = (os.environ.get("DB_HOST") or "").strip()
        if not host:
            raise DatabaseConnectionError(
                "DB_HOST no está configurado. "
                "Definir la variable en la ECS Task Definition antes de desplegar."
            )

        sslmode = (os.environ.get("DB_SSLMODE", "require") or "require").strip().lower()
        local_hosts = {"localhost", "127.0.0.1", "::1"}
        if host.lower() in local_hosts and sslmode in {"require", "verify-ca", "verify-full"}:
            sslmode = "prefer"
            logger.warning(
                "Host local detectado (%s); sslmode bajado automáticamente a 'prefer'.", host
            )

        valid_sslmodes = {"disable", "allow", "prefer", "require", "verify-ca", "verify-full"}
        if sslmode not in valid_sslmodes:
            logger.warning("sslmode='%s' no reconocido; usando 'require'.", sslmode)
            sslmode = "require"

        self._db_host    = host
        self._db_sslmode = sslmode

        # Claves obligatorias — os.environ[] lanza KeyError inmediato si faltan
        try:
            db_url = URL.create(
                drivername="postgresql+psycopg2",
                username=os.environ["DB_USER"],
                password=os.environ["DB_PASSWORD"],
                host=host,
                port=int(os.environ.get("DB_PORT", "5432")),
                database=os.environ["DB_NAME"],
            )
        except KeyError as err:
            raise DatabaseConnectionError(
                f"Variable de entorno de RDS faltante: {err}. "
                "Definir DB_USER, DB_PASSWORD y DB_NAME en la Task Definition."
            ) from err

        engine = create_engine(
            db_url,
            connect_args={"connect_timeout": 10, "sslmode": sslmode},
            pool_pre_ping=True,   # Verifica la conexión antes de cada uso
            pool_size=5,          # Conexiones permanentes en el pool
            max_overflow=10,      # Conexiones adicionales bajo picos de tráfico
        )
        logger.info(
            "Engine RDS creado | host=%s | port=%s | db=%s | sslmode=%s",
            host,
            os.environ.get("DB_PORT", "5432"),
            os.environ.get("DB_NAME", "?"),
            sslmode,
        )
        return engine

    # ------------------------------------------------------------------
    # Helpers SQL internos
    # ------------------------------------------------------------------

    def _read_sql(self, sql: str, params: dict | None = None) -> pd.DataFrame:
        """Ejecuta SQL parametrizado y devuelve DataFrame.

        Convierte OperationalError de SQLAlchemy en DatabaseConnectionError
        para que la capa HTTP retorne 503 en lugar de un 500 genérico.
        """
        assert self.engine is not None, "Engine no inicializado — llamar startup() primero."
        try:
            with self.engine.connect() as conn:
                return pd.read_sql(text(sql), conn, params=params)
        except OperationalError as err:
            hint = (
                " Para Postgres local usa DB_SSLMODE=disable en las variables de entorno."
                if self._db_host.lower() in {"localhost", "127.0.0.1", "::1"}
                else ""
            )
            raise DatabaseConnectionError(
                f"No se pudo conectar a PostgreSQL "
                f"(host={self._db_host}, sslmode={self._db_sslmode}).{hint}"
            ) from err

    def _query_user_prior(self, user_id: int) -> pd.DataFrame:
        sql = """
            SELECT user_key, product_key, order_key, order_number,
                   days_since_prior_order, reordered, add_to_cart_order, get_eval
            FROM fact_order_products
            WHERE user_key = :user_id AND get_eval = 'prior'
        """
        return self._read_sql(sql, {"user_id": user_id})

    def _query_user_dim_product(self, product_keys: List[int]) -> pd.DataFrame:
        sql = """
            SELECT product_key, product_name, department_name, aisle_name
            FROM dim_product
            WHERE product_key = ANY(:product_keys)
        """
        return self._read_sql(sql, {"product_keys": product_keys})

    def _query_product_features(self, product_keys: List[int]) -> pd.DataFrame:
        """Calcula features estadísticas de productos directamente en RDS con CTEs."""
        sql = """
            WITH selected_products AS (
                SELECT product_key, department_name, aisle_name
                FROM dim_product
                WHERE product_key = ANY(:product_keys)
            ),
            product_stats AS (
                SELECT
                    f.product_key,
                    COUNT(f.order_key)::int            AS product_total_purchases,
                    AVG(f.reordered::float)            AS product_reorder_rate,
                    AVG(f.add_to_cart_order::float)    AS product_avg_add_to_cart,
                    COUNT(DISTINCT f.user_key)::int    AS product_unique_users
                FROM fact_order_products f
                WHERE f.get_eval = 'prior' AND f.product_key = ANY(:product_keys)
                GROUP BY f.product_key
            ),
            department_stats AS (
                SELECT p.department_name,
                       AVG(f.reordered::float) AS p_department_reorder_rate
                FROM fact_order_products f
                JOIN dim_product p ON p.product_key = f.product_key
                WHERE f.get_eval = 'prior'
                GROUP BY p.department_name
            ),
            aisle_stats AS (
                SELECT p.aisle_name,
                       AVG(f.reordered::float) AS p_aisle_reorder_rate
                FROM fact_order_products f
                JOIN dim_product p ON p.product_key = f.product_key
                WHERE f.get_eval = 'prior'
                GROUP BY p.aisle_name
            )
            SELECT s.product_key,
                   ps.product_total_purchases,
                   ps.product_reorder_rate,
                   ps.product_avg_add_to_cart,
                   ps.product_unique_users,
                   ds.p_department_reorder_rate,
                   ais.p_aisle_reorder_rate
            FROM selected_products s
            LEFT JOIN product_stats    ps  ON ps.product_key     = s.product_key
            LEFT JOIN department_stats ds  ON ds.department_name = s.department_name
            LEFT JOIN aisle_stats      ais ON ais.aisle_name     = s.aisle_name
        """
        return self._read_sql(sql, {"product_keys": product_keys})

    def _query_user_order_count(self, user_id: int) -> int:
        sql = """
            SELECT COUNT(DISTINCT order_key) AS n_orders
            FROM fact_order_products
            WHERE user_key = :user_id AND get_eval = 'prior'
        """
        result = self._read_sql(sql, {"user_id": user_id})
        return int(result["n_orders"].iloc[0]) if not result.empty else 0

    # ------------------------------------------------------------------
    # Cold-start fallback
    # ------------------------------------------------------------------

    def _cold_start_top_products(self, user_id: int, top_k: int) -> List[dict]:
        """Ranking por frecuencia personal cuando el historial es insuficiente."""
        sql = """
            WITH user_stats AS (
                SELECT COUNT(DISTINCT order_key) AS n_orders
                FROM fact_order_products
                WHERE user_key = :user_id AND get_eval = 'prior'
            ),
            product_counts AS (
                SELECT product_key, COUNT(*) AS purchase_count
                FROM fact_order_products
                WHERE user_key = :user_id AND get_eval = 'prior'
                GROUP BY product_key
                ORDER BY purchase_count DESC
                LIMIT :top_k
            )
            SELECT pc.product_key,
                   pc.purchase_count,
                   ROUND(pc.purchase_count::numeric / us.n_orders, 4) AS probability
            FROM product_counts pc
            CROSS JOIN user_stats us
        """
        top = self._read_sql(sql, {"user_id": user_id, "top_k": top_k})
        if top.empty:
            return []

        names = (
            self._query_user_dim_product(top["product_key"].astype(int).tolist())
            [["product_key", "product_name"]]
            .drop_duplicates(subset=["product_key"])
        )
        top = top.merge(names, on="product_key", how="left")
        return [
            {
                "product_key":  int(row.product_key),
                "product_name": None if pd.isna(row.product_name) else str(row.product_name),
                "probability":  float(row.probability),
            }
            for row in top.itertuples(index=False)
        ]

    # ------------------------------------------------------------------
    # Asignación de clusters (usa _ClusterArtifacts con atributos tipados)
    # ------------------------------------------------------------------

    @staticmethod
    def _assign_user_cluster(
        matrix: pd.DataFrame,
        user_id: int,
        clusters: _ClusterArtifacts,
    ) -> pd.DataFrame:
        """Asigna el cluster de comportamiento del usuario vía KMeans.

        Usuarios no vistos durante el entrenamiento → cluster = -1.
        Accede a clusters.scaler_user / kmeans_user / user_profiles como
        atributos tipados (no como claves de dict) para detectar errores en startup.
        """
        matrix = matrix.copy()
        if user_id not in clusters.user_profiles.index:
            matrix["user_cluster"] = -1
            return matrix
        profile = matrix.groupby("user_key")[USER_CLUSTER_FEATURES].mean()
        scaled  = clusters.scaler_user.transform(profile)
        matrix["user_cluster"] = int(clusters.kmeans_user.predict(scaled)[0])
        return matrix

    @staticmethod
    def _assign_product_clusters(
        matrix: pd.DataFrame,
        clusters: _ClusterArtifacts,
    ) -> pd.DataFrame:
        """Asigna el cluster de popularidad a cada producto vía KMeans.

        Productos no vistos durante el entrenamiento → cluster = -1.
        """
        matrix   = matrix.copy()
        profiles = matrix.groupby("product_key")[PRODUCT_CLUSTER_FEATURES].mean()
        cluster_series = pd.Series(-1, index=profiles.index, name="product_cluster", dtype="int8")

        known = profiles.index.isin(clusters.product_profiles.index)
        if known.any():
            kp     = profiles[known].fillna(profiles[known].median())
            scaled = clusters.scaler_product.transform(kp)
            cluster_series.loc[kp.index] = clusters.kmeans_product.predict(scaled).astype("int8")

        return matrix.merge(cluster_series.reset_index(), on="product_key", how="left")

    # ------------------------------------------------------------------
    # Pipeline de features online
    # ------------------------------------------------------------------

    def _build_online_matrix(self, user_id: int) -> pd.DataFrame:
        """Construye la matriz de features usuario-producto para inferencia en tiempo real.

        Pasos:
          1. Historial prior desde RDS
          2. Features del usuario (frecuencia, recencia, segmento)
          3. Features usuario-producto (cuántas veces compró cada producto)
          4. Features estadísticas de productos calculadas en RDS (CTEs)
          5. Filtro: descarta productos con < 50 compras globales
          6. Join de todas las features en una sola matriz
          7. Features derivadas (up_reorder_rate, up_orders_since_last_purchase)
          8. Clusters de usuario y producto usando _ClusterArtifacts
        """
        prior = self._query_user_prior(user_id)
        if prior.empty:
            raise UserNotFoundError(
                f"user_id {user_id} sin historial prior en fact_order_products."
            )

        for col in ["user_key", "product_key", "order_key", "order_number",
                    "reordered", "add_to_cart_order"]:
            if col in prior.columns:
                prior[col] = pd.to_numeric(prior[col], errors="coerce")

        product_keys     = sorted(prior["product_key"].dropna().astype(int).unique().tolist())
        dim_product_user = self._query_user_dim_product(product_keys)
        dim_product_norm = _normalize_dim_product(dim_product_user)

        user_feat   = get_user_features(prior, min_user_orders=1)
        up_feat     = get_user_product_features(prior)
        user_dept   = get_user_department_feature(prior, dim_product_norm)
        user_aisle  = get_user_aisle_feature(prior, dim_product_norm)

        product_feat = self._query_product_features(product_keys)
        product_feat = product_feat.dropna(subset=["product_total_purchases"]).copy()
        product_feat = product_feat[product_feat["product_total_purchases"] >= 50].copy()
        if product_feat.empty:
            raise UserNotFoundError(
                f"user_id {user_id} sin productos con historial suficiente (≥ 50 compras globales)."
            )

        valid_products = set(product_feat["product_key"].astype(int).tolist())
        matrix = up_feat[up_feat["product_key"].isin(valid_products)].copy()
        if matrix.empty:
            raise UserNotFoundError(
                f"user_id {user_id} sin pares usuario-producto válidos tras filtros."
            )

        matrix = matrix.merge(user_feat,    on="user_key",    how="left")
        matrix = matrix.merge(product_feat, on="product_key", how="left")
        matrix = matrix.merge(user_dept,    on="user_key",    how="left")
        matrix = matrix.merge(user_aisle,   on="user_key",    how="left")

        matrix["up_reorder_rate"] = (
            matrix["up_times_purchased"] / matrix["user_total_orders"]
        ).astype("float32")
        matrix["up_orders_since_last_purchase"] = (
            matrix["user_total_orders"] - matrix["up_last_order_number"]
        ).clip(lower=0).astype("int16")

        # Usa el objeto _ClusterArtifacts tipado en lugar del dict raw
        clusters = self._artifacts.clusters
        matrix = self._assign_user_cluster(matrix, user_id, clusters)
        matrix = self._assign_product_clusters(matrix, clusters)

        return matrix

    # ------------------------------------------------------------------
    # Validación del contrato de features
    # ------------------------------------------------------------------

    def _align_and_validate(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """Garantiza que la matriz tenga exactamente las columnas del modelo entrenado."""
        missing = [c for c in self.feature_cols if c not in matrix.columns]
        if missing:
            raise FeatureContractError(f"Columnas faltantes para inferencia: {missing}")

        X = matrix[self.feature_cols].copy()
        if X.shape[1] != self.n_features:
            raise FeatureContractError(
                f"n_features inválido: esperado={self.n_features}, recibido={X.shape[1]}"
            )
        return X

    # ------------------------------------------------------------------
    # Punto de entrada público
    # ------------------------------------------------------------------

    def recommend_user(self, user_id: int, top_k: int = 10) -> List[dict]:
        """Genera las top_k recomendaciones de next-basket para un usuario.

        Lógica de decisión:
          n_orders == 0            → UserNotFoundError (HTTP 404)
          0 < n_orders < MIN_ORDERS → cold-start (ranking por frecuencia personal)
          n_orders >= MIN_ORDERS   → inferencia completa con LightGBM

        Retorna lista de dicts: [{product_key, product_name, probability}, ...]
        """
        n_orders = self._query_user_order_count(user_id)
        if n_orders == 0:
            raise UserNotFoundError(
                f"user_id {user_id} sin historial prior en fact_order_products."
            )
        if n_orders < MIN_ORDERS_FOR_MODEL:
            logger.info("cold-start | user_id=%d | n_orders=%d", user_id, n_orders)
            return self._cold_start_top_products(user_id, top_k)

        matrix = self._build_online_matrix(user_id)
        X      = self._align_and_validate(matrix)

        import typing
        model         = typing.cast(typing.Any, self._artifacts.model)
        probabilities = model.predict_proba(X)[:, 1]

        scored = matrix[["product_key"]].copy()
        scored["probability"] = probabilities
        scored = scored.sort_values("probability", ascending=False).head(top_k)

        names = (
            self._query_user_dim_product(scored["product_key"].astype(int).tolist())
            [["product_key", "product_name"]]
            .drop_duplicates(subset=["product_key"])
        )
        scored = scored.merge(names, on="product_key", how="left")

        return [
            {
                "product_key":  int(row.product_key),
                "product_name": None if pd.isna(row.product_name) else str(row.product_name),
                "probability":  float(row.probability),
            }
            for row in scored.itertuples(index=False)
        ]