"""
inference.py — Insight Commerce · Recsys API
Entorno destino: AWS Fargate (ECS) + S3 + RDS PostgreSQL
Autenticación  : IAM Task Role (ecsTaskRole.InsightCommerce) — sin hardcoded keys
Artefactos     : descargados desde S3 a /tmp/ al iniciar el contenedor
"""
import json
import logging
import os
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
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

USER_CLUSTER_FEATURES = [
    "user_total_orders",
    "user_avg_basket_size",
    "user_days_since_last_order",
    "user_reorder_ratio",
    "user_distinct_products",
]
PRODUCT_CLUSTER_FEATURES = [
    "product_total_purchases",
    "product_reorder_rate",
    "product_avg_add_to_cart",
    "product_unique_users",
    "p_department_reorder_rate",
]

logger = logging.getLogger("api")

S3_BUCKET: str = os.getenv("S3_BUCKET", "")
S3_PREFIX: str = os.getenv("S3_PREFIX", "models")

_TMP             = "/tmp"
_MODEL_LOCAL     = f"{_TMP}/model.pkl"
_CLUSTER_LOCAL   = f"{_TMP}/cluster_models.pkl"
_MODEL_LOG_LOCAL = f"{_TMP}/model_log.json"

MIN_ORDERS_FOR_MODEL = 5

class UserNotFoundError(Exception):
    """El user_id no tiene historial prior en RDS (0 órdenes prior)."""

class FeatureContractError(Exception):
    """La matriz de features no coincide con el contrato del modelo entrenado."""

class DatabaseConnectionError(Exception):
    """No fue posible establecer conexión con PostgreSQL en RDS."""

@dataclass
class _ClusterArtifacts:
    """Desempaqueta las claves del dict cluster_models.pkl con tipado explícito."""
    scaler_user:      Any
    kmeans_user:      Any
    user_profiles:    Any
    scaler_product:   Any
    kmeans_product:   Any
    product_profiles: Any

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "_ClusterArtifacts":
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
    model:     Any
    clusters:  _ClusterArtifacts
    model_log: Dict[str, Any]

class RecommendationService:
    """Orquesta la descarga desde S3, la conexión a RDS y la inferencia."""

    def __init__(self) -> None:
        self.model_name:   str       = "LightGBM"
        self.n_features:   int       = 0
        self.feature_cols: List[str] = []
        self.engine                  = None
        self._db_host:    str        = ""
        self._db_sslmode: str        = ""
        self._artifacts: LoadedArtifacts | None = None

    def startup(self) -> None:
        use_s3 = os.getenv("USE_S3", "false").lower() == "true"
        if use_s3 and not S3_BUCKET:
            raise RuntimeError(
                "S3_BUCKET no está configurado y USE_S3=true. "
                "Definir S3_BUCKET en la ECS Task Definition antes de desplegar."
            )
        self._artifacts   = self._download_and_load_artifacts()
        self.engine       = self._build_engine()
        self.model_name   = str(self._artifacts.model_log.get("model_name",   self.model_name))
        self.n_features   = int(self._artifacts.model_log.get("n_features",   0))
        self.feature_cols = list(self._artifacts.model_log.get("feature_cols", []))
        logger.info(
            "startup OK | model=%s | n_features=%d | bucket=%s | prefix=%s",
            self.model_name, self.n_features, S3_BUCKET, S3_PREFIX,
        )

    def _s3_key(self, filename: str) -> str:
        return f"{S3_PREFIX}/{filename}" if S3_PREFIX else filename

    def _download_and_load_artifacts(self) -> LoadedArtifacts:
        use_s3 = os.getenv("USE_S3", "false").lower() == "true"

        if not use_s3:
            # Modo local — lee desde disco sin credenciales AWS
            logger.info("USE_S3=false | cargando artefactos desde disco local")
            model_path   = Path(__file__).resolve().parents[2] / "models" / "model.pkl"
            cluster_path = Path(__file__).resolve().parents[2] / "models" / "cluster_models.pkl"
            log_path     = Path(__file__).resolve().parents[2] / "models" / "model_log.json"
            model            = joblib.load(model_path)
            raw_cluster_dict = joblib.load(cluster_path)
            clusters         = _ClusterArtifacts.from_dict(raw_cluster_dict)
            with open(log_path, "r", encoding="utf-8") as fh:
                model_log = json.load(fh)
            return LoadedArtifacts(model=model, clusters=clusters, model_log=model_log)

        # Modo S3 — descarga desde bucket
        s3 = boto3.client("s3")
        artifacts_to_download = [
            ("model.pkl",          _MODEL_LOCAL),
            ("cluster_models.pkl", _CLUSTER_LOCAL),
            ("model_log.json",     _MODEL_LOG_LOCAL),
        ]
        for filename, local_path in artifacts_to_download:
            s3_key = self._s3_key(filename)
            try:
                logger.info("S3 download | bucket=%s | key=%s → %s", S3_BUCKET, s3_key, local_path)
                s3.download_file(S3_BUCKET, s3_key, local_path,
                                 ExtraArgs={"ExpectedBucketOwner": os.getenv("AWS_ACCOUNT_ID", "")})
                logger.info("S3 download OK | %s", local_path)
            except ClientError as err:
                error_code = err.response.get("Error", {}).get("Code", "UNKNOWN")
                error_msg  = err.response.get("Error", {}).get("Message", str(err))
                logger.critical(
                    "S3 download FAILED | bucket=%s | key=%s | error_code=%s | message=%s",
                    S3_BUCKET, s3_key, error_code, error_msg,
                )
                raise
            except BotoCoreError as err:
                logger.critical(
                    "S3 download FAILED (network/credentials) | bucket=%s | key=%s | %s",
                    S3_BUCKET, s3_key, err,
                )
                raise
            except OSError as err:
                logger.critical(
                    "S3 download FAILED (write error) | local_path=%s | %s", local_path, err,
                )
                raise

        model            = joblib.load(_MODEL_LOCAL)
        raw_cluster_dict = joblib.load(_CLUSTER_LOCAL)
        clusters         = _ClusterArtifacts.from_dict(raw_cluster_dict)
        with open(_MODEL_LOG_LOCAL, "r", encoding="utf-8") as fh:
            model_log = json.load(fh)
        return LoadedArtifacts(model=model, clusters=clusters, model_log=model_log)
               

    def _build_engine(self):
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
            logger.warning("Host local detectado (%s); sslmode bajado a 'prefer'.", host)
        valid_sslmodes = {"disable", "allow", "prefer", "require", "verify-ca", "verify-full"}
        if sslmode not in valid_sslmodes:
            logger.warning("sslmode='%s' no reconocido; usando 'require'.", sslmode)
            sslmode = "require"
        self._db_host    = host
        self._db_sslmode = sslmode
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
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )
        logger.info(
            "Engine RDS creado | host=%s | port=%s | db=%s | sslmode=%s",
            host, os.environ.get("DB_PORT", "5432"),
            os.environ.get("DB_NAME", "?"), sslmode,
        )
        return engine

    def _read_sql(self, sql: str, params: dict | None = None) -> pd.DataFrame:
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

    def _get_global_top_products(self, top_k: int) -> List[dict]:
        """Devuelve los top_k productos más comprados globalmente.
        Se usa como fallback para usuarios que no existen en dim_user.
        La probability es la frecuencia normalizada sobre el total de compras globales.
        """
        sql = """
            WITH total AS (
                SELECT COUNT(*)::float AS total_purchases
                FROM fact_order_products
                WHERE get_eval = 'prior'
            ),
            top_products AS (
                SELECT product_key, COUNT(*) AS purchase_count
                FROM fact_order_products
                WHERE get_eval = 'prior'
                GROUP BY product_key
                ORDER BY purchase_count DESC
                LIMIT :top_k
            )
            SELECT tp.product_key,
                   tp.purchase_count,
                   ROUND((tp.purchase_count / t.total_purchases)::numeric, 6) AS probability
            FROM top_products tp
            CROSS JOIN total t
        """
        top = self._read_sql(sql, {"top_k": top_k})
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

    @staticmethod
    def _assign_user_cluster(
        matrix: pd.DataFrame,
        user_id: int,
        clusters: _ClusterArtifacts,
    ) -> pd.DataFrame:
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
        matrix   = matrix.copy()
        profiles = matrix.groupby("product_key")[PRODUCT_CLUSTER_FEATURES].mean()
        cluster_series = pd.Series(-1, index=profiles.index, name="product_cluster", dtype="int8")
        known = profiles.index.isin(clusters.product_profiles.index)
        if known.any():
            kp     = profiles[known].fillna(profiles[known].median())
            scaled = clusters.scaler_product.transform(kp)
            cluster_series.loc[kp.index] = clusters.kmeans_product.predict(scaled).astype("int8")
        return matrix.merge(cluster_series.reset_index(), on="product_key", how="left")

    def _build_online_matrix(self, user_id: int) -> pd.DataFrame:
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
        clusters = self._artifacts.clusters
        matrix = self._assign_user_cluster(matrix, user_id, clusters)
        matrix = self._assign_product_clusters(matrix, clusters)
        return matrix

    def _align_and_validate(self, matrix: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.feature_cols if c not in matrix.columns]
        if missing:
            raise FeatureContractError(f"Columnas faltantes para inferencia: {missing}")
        X = matrix[self.feature_cols].copy()
        if X.shape[1] != self.n_features:
            raise FeatureContractError(
                f"n_features inválido: esperado={self.n_features}, recibido={X.shape[1]}"
            )
        return X

    def recommend_user(self, user_id: int, top_k: int = 10) -> tuple[List[dict], bool]:
        """Genera las top_k recomendaciones de next-basket para un usuario.

        Retorna tupla (recommendations, cold_start) donde:
          cold_start=False → modelo LightGBM completo (≥ 5 órdenes prior)
          cold_start=True  → popularidad personal (1-4 órdenes) o global (usuario nuevo)

        Flujo:
          n_orders == 0 + NO existe en dim_user → popularidad global  (cold_start=True)
          n_orders == 0 + SÍ existe en dim_user → UserNotFoundError   (HTTP 404)
          n_orders 1-4                          → popularidad personal (cold_start=True)
          n_orders >= 5                         → modelo LightGBM      (cold_start=False)
        """
        n_orders = self._query_user_order_count(user_id)

        if n_orders == 0:
            exists = self._read_sql(
                "SELECT 1 FROM dim_user WHERE user_key = :user_id LIMIT 1",
                {"user_id": user_id},
            )
            if exists.empty:
                return self._get_global_top_products(top_k), True
            raise UserNotFoundError(
                f"user_id {user_id} sin historial prior en fact_order_products."
            )

        if n_orders < MIN_ORDERS_FOR_MODEL:
            logger.info("cold-start personal | user_id=%d | n_orders=%d", user_id, n_orders)
            return self._cold_start_top_products(user_id, top_k), True

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
        ], False