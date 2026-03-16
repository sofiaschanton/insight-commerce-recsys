import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.exc import OperationalError

from src.features.feature_engineering import (
    _normalize_dim_product,
    get_user_department_feature,
    get_user_features,
    get_user_product_features,
    get_user_aisle_feature,
)
from src.models.train import PRODUCT_CLUSTER_FEATURES, USER_CLUSTER_FEATURES


# Directorio raíz del proyecto (dos niveles arriba de este archivo)
ROOT_DIR = Path(__file__).resolve().parents[2]
# Carga las variables de entorno desde el archivo .env (credenciales de DB, rutas de modelos, etc.)
load_dotenv(ROOT_DIR / ".env")


# --- Excepciones personalizadas ---
# Se usan para diferenciar tipos de error y devolver el código HTTP correcto en la API

class UserNotFoundError(Exception):
    """Se lanza cuando el user_id no tiene ningún historial en la base de datos (0 órdenes prior)."""
    pass


class FeatureContractError(Exception):
    """Se lanza cuando la matriz de features no coincide con lo que espera el modelo entrenado."""
    pass


class DatabaseConnectionError(Exception):
    """Se lanza cuando no se puede conectar a PostgreSQL."""
    pass


# Mínimo de órdenes prior que un usuario necesita para que el modelo ML sea aplicable.
# Usuarios con menos órdenes reciben un ranking de popularidad personal (cold-start fallback).
MIN_ORDERS_FOR_MODEL = 5


@dataclass
class LoadedArtifacts:
    """Contenedor para los tres artefactos que se cargan al iniciar el servicio."""
    model: object           # Modelo LightGBM entrenado
    cluster_models: Dict[str, object]  # Scalers y KMeans para clusters de usuario y producto
    model_log: Dict[str, object]       # Metadatos del modelo: feature_cols, n_features, AUC, etc.


class RecommendationService:
    """Servicio principal que orquesta la inferencia de recomendaciones next-basket."""

    def __init__(self) -> None:
        # Rutas a los archivos de modelo (configurables por variables de entorno)
        self.model_path = Path(os.getenv("API_MODEL_PATH", "models/model.pkl"))
        self.cluster_model_path = Path(
            os.getenv("API_CLUSTER_MODEL_PATH", "models/cluster_models.pkl")
        )
        self.model_log_path = Path(os.getenv("API_MODEL_LOG_PATH", "models/model_log.json"))

        # Metadatos del modelo (se llenan en startup)
        self.model_name = "LightGBM optimizado"
        self.n_features = 0
        self.feature_cols: List[str] = []

        # Conexión a la base de datos y cache de artefactos (se inicializan en startup)
        self._engine = None
        self._db_host = ""      # Host de la DB (se guarda para mensajes de error)
        self._db_sslmode = ""   # Modo SSL efectivo que se usó al conectar
        self._artifacts: LoadedArtifacts | None = None

    def startup(self) -> None:
        """Se ejecuta una sola vez al iniciar la API. Carga modelos y abre la conexión a la DB."""
        self._artifacts = self._load_artifacts()
        self._engine = self._build_engine()
        # Sobreescribe los metadatos con los valores reales del model_log.json
        self.model_name = self._artifacts.model_log.get("model_name", self.model_name)
        self.n_features = int(self._artifacts.model_log.get("n_features", 0))
        self.feature_cols = list(self._artifacts.model_log.get("feature_cols", []))

    def _load_artifacts(self) -> LoadedArtifacts:
        """Carga desde disco el modelo, los modelos de clustering y el log del modelo."""
        model = joblib.load(self.model_path)
        cluster_models = joblib.load(self.cluster_model_path)
        with open(self.model_log_path, "r", encoding="utf-8") as f:
            model_log = json.load(f)
        return LoadedArtifacts(model=model, cluster_models=cluster_models, model_log=model_log)

    def _build_engine(self):
        """Crea el engine de SQLAlchemy para conectarse a PostgreSQL.

        Lee las credenciales desde variables de entorno (.env).
        Ajusta automáticamente el modo SSL: si el host es local (localhost/127.0.0.1)
        y se pide SSL estricto, lo baja a 'prefer' para evitar errores de conexión.
        """
        host = (os.getenv("NEON_HOST") or "").strip()
        sslmode = (os.getenv("NEON_SSLMODE", "require") or "require").strip().lower()

        if not host:
            raise DatabaseConnectionError(
                "NEON_HOST no está configurado. Definilo en .env antes de iniciar la API."
            )

        # Si el host es local, SSL estricto no es compatible → bajamos a 'prefer'
        local_hosts = {"localhost", "127.0.0.1", "::1"}
        is_local_host = host.lower() in local_hosts
        if is_local_host and sslmode in {"require", "verify-ca", "verify-full"}:
            sslmode = "prefer"

        # Validación: si el valor de sslmode no es conocido, usar 'require' como fallback seguro
        valid_sslmodes = {"disable", "allow", "prefer", "require", "verify-ca", "verify-full"}
        if sslmode not in valid_sslmodes:
            sslmode = "require"

        # Guardamos host y sslmode para poder incluirlos en mensajes de error posteriores
        self._db_host = host
        self._db_sslmode = sslmode

        db_url = URL.create(
            drivername="postgresql+psycopg2",
            username=os.getenv("NEON_USER"),
            password=os.getenv("NEON_PASSWORD"),
            host=host,
            port=os.getenv("NEON_PORT", "5432"),
            database=os.getenv("NEON_DATABASE"),
        )
        return create_engine(
            db_url,
            connect_args={"connect_timeout": 10, "sslmode": sslmode},
            pool_pre_ping=True,  # Verifica la conexión antes de cada uso (evita conexiones muertas)
        )

    def _read_sql(self, sql: str, params: dict | None = None) -> pd.DataFrame:
        """Ejecuta una query SQL parametrizada y devuelve el resultado como DataFrame.

        Captura errores de conexión de SQLAlchemy y los convierte en DatabaseConnectionError
        con un mensaje claro sobre el host y el modo SSL usado.
        """
        try:
            with self._engine.connect() as conn:
                return pd.read_sql(text(sql), conn, params=params)
        except OperationalError as err:
            # Mensaje adicional para usuarios con Postgres local
            hint = ""
            if self._db_host.lower() in {"localhost", "127.0.0.1", "::1"}:
                hint = " Para Postgres local, usá NEON_SSLMODE=disable o prefer en .env."

            raise DatabaseConnectionError(
                f"No se pudo conectar a PostgreSQL (host={self._db_host}, sslmode={self._db_sslmode}).{hint}"
            ) from err

    def _query_user_prior(self, user_id: int) -> pd.DataFrame:
        """Trae todo el historial de compras previas (get_eval='prior') del usuario.

        Estos datos son la base para calcular todas las features del usuario y
        de los pares usuario-producto.
        """
        sql = """
            SELECT user_key, product_key, order_key, order_number,
                   days_since_prior_order, reordered, add_to_cart_order, get_eval
            FROM fact_order_products
            WHERE user_key = :user_id AND get_eval = 'prior'
        """
        return self._read_sql(sql, {"user_id": user_id})

    def _query_user_dim_product(self, product_keys: List[int]) -> pd.DataFrame:
        """Trae el nombre, departamento y pasillo de una lista de productos.

        Se usa tanto para construir features categoricas como para enriquecer
        la respuesta final con el nombre del producto.
        """
        sql = """
            SELECT product_key, product_name, department_name, aisle_name
            FROM dim_product
            WHERE product_key = ANY(:product_keys)
        """
        return self._read_sql(sql, {"product_keys": product_keys})

    def _query_product_features(self, product_keys: List[int]) -> pd.DataFrame:
        """Calcula features estadísticas de los productos directamente en la base de datos.

        Usa CTEs para calcular en paralelo:
        - product_stats: total de compras, tasa de recompra, posición promedio en el carrito
        - department_stats: tasa de recompra promedio del departamento del producto
        - aisle_stats: tasa de recompra promedio del pasillo del producto

        Solo se consultan los productos que ya compró el usuario (product_keys).
        """
        sql = """
            WITH selected_products AS (
                SELECT product_key, department_name, aisle_name
                FROM dim_product
                WHERE product_key = ANY(:product_keys)
            ),
            product_stats AS (
                SELECT
                    f.product_key,
                    COUNT(f.order_key)::int AS product_total_purchases,
                    AVG(f.reordered::float) AS product_reorder_rate,
                    AVG(f.add_to_cart_order::float) AS product_avg_add_to_cart,
                    COUNT(DISTINCT f.user_key)::int AS product_unique_users
                FROM fact_order_products f
                WHERE f.get_eval = 'prior' AND f.product_key = ANY(:product_keys)
                GROUP BY f.product_key
            ),
            department_stats AS (
                SELECT
                    p.department_name,
                    AVG(f.reordered::float) AS p_department_reorder_rate
                FROM fact_order_products f
                JOIN dim_product p ON p.product_key = f.product_key
                WHERE f.get_eval = 'prior'
                GROUP BY p.department_name
            ),
            aisle_stats AS (
                SELECT
                    p.aisle_name,
                    AVG(f.reordered::float) AS p_aisle_reorder_rate
                FROM fact_order_products f
                JOIN dim_product p ON p.product_key = f.product_key
                WHERE f.get_eval = 'prior'
                GROUP BY p.aisle_name
            )
            SELECT
                s.product_key,
                ps.product_total_purchases,
                ps.product_reorder_rate,
                ps.product_avg_add_to_cart,
                ps.product_unique_users,
                ds.p_department_reorder_rate,
                ais.p_aisle_reorder_rate
            FROM selected_products s
            LEFT JOIN product_stats ps ON ps.product_key = s.product_key
            LEFT JOIN department_stats ds ON ds.department_name = s.department_name
            LEFT JOIN aisle_stats ais ON ais.aisle_name = s.aisle_name
        """
        return self._read_sql(sql, {"product_keys": product_keys})

    def _query_user_order_count(self, user_id: int) -> int:
        """Devuelve la cantidad de órdenes prior distintas del usuario.

        Retorna 0 si el usuario no existe en la tabla o no tiene historial prior.
        Se usa antes de entrar al pipeline de features para decidir si aplicar cold-start.
        """
        sql = """
            SELECT COUNT(DISTINCT order_key) AS n_orders
            FROM fact_order_products
            WHERE user_key = :user_id AND get_eval = 'prior'
        """
        result = self._read_sql(sql, {"user_id": user_id})
        return int(result["n_orders"].iloc[0]) if not result.empty else 0

    def _cold_start_top_products(self, user_id: int, top_k: int) -> List[dict]:
        """Devuelve los top_k productos más comprados por el usuario como fallback cold-start.

        Se usa cuando el usuario tiene menos de MIN_ORDERS_FOR_MODEL órdenes y no hay
        suficiente historial para construir el feature set completo del modelo.
        La 'probability' es la frecuencia de compra normalizada por órdenes totales
        (cuántas de sus órdenes incluyeron ese producto), expresada entre 0 y 1.
        """
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
            SELECT
                pc.product_key,
                pc.purchase_count,
                ROUND(pc.purchase_count::numeric / us.n_orders, 4) AS probability
            FROM product_counts pc
            CROSS JOIN user_stats us
        """
        top = self._read_sql(sql, {"user_id": user_id, "top_k": top_k})

        if top.empty:
            return []

        # Enriquece con el nombre del producto
        product_keys = top["product_key"].astype(int).tolist()
        product_names = (
            self._query_user_dim_product(product_keys)
            .loc[:, ["product_key", "product_name"]]
            .drop_duplicates(subset=["product_key"])
        )
        top = top.merge(product_names, on="product_key", how="left")

        return [
            {
                "product_key": int(row.product_key),
                "product_name": None if pd.isna(row.product_name) else str(row.product_name),
                "probability": float(row.probability),
            }
            for row in top.itertuples(index=False)
        ]

    @staticmethod
    def _assign_user_cluster(
        matrix: pd.DataFrame,
        user_id: int,
        scaler_user,
        kmeans_user,
        user_profiles_ref: pd.DataFrame,
    ) -> pd.DataFrame:
        """Asigna el cluster de comportamiento del usuario usando KMeans.

        Si el usuario no estaba presente en el entrenamiento del KMeans, se le asigna -1
        (cluster desconocido). De lo contrario, se escalan sus features y se predice
        a cuál de los 5 clusters pertenece.
        """
        matrix = matrix.copy()
        # Usuario nuevo/desconocido: no se puede asignar cluster → -1
        if user_id not in user_profiles_ref.index:
            matrix["user_cluster"] = -1
            return matrix

        # Calcula el perfil promedio del usuario con las features de clustering
        profile = matrix.groupby("user_key")[USER_CLUSTER_FEATURES].mean()
        scaled = scaler_user.transform(profile)
        cluster = int(kmeans_user.predict(scaled)[0])
        matrix["user_cluster"] = cluster
        return matrix

    @staticmethod
    def _assign_product_clusters(
        matrix: pd.DataFrame,
        scaler_product,
        kmeans_product,
        product_profiles_ref: pd.DataFrame,
    ) -> pd.DataFrame:
        """Asigna el cluster de popularidad/comportamiento a cada producto usando KMeans.

        Solo se asigna cluster a los productos que estaban en el set de entrenamiento
        del KMeans. Los productos nuevos/desconocidos reciben -1.
        """
        matrix = matrix.copy()

        # Calcula el perfil promedio de cada producto con sus features de clustering
        profiles = matrix.groupby("product_key")[PRODUCT_CLUSTER_FEATURES].mean()
        # Por defecto todos los productos arrancan con cluster -1 (desconocido)
        clusters = pd.Series(-1, index=profiles.index, name="product_cluster", dtype="int8")

        # Solo predice para productos que el KMeans conoce
        known = profiles.index.isin(product_profiles_ref.index)
        if known.any():
            known_profiles = profiles[known].fillna(profiles[known].median())  # Imputa NaNs con la mediana
            known_scaled = scaler_product.transform(known_profiles)
            clusters.loc[known_profiles.index] = kmeans_product.predict(known_scaled).astype("int8")

        return matrix.merge(clusters.reset_index(), on="product_key", how="left")

    def _build_online_matrix(self, user_id: int) -> pd.DataFrame:
        """Construye la matriz de features online para un usuario, lista para el modelo.

        Flujo:
        1. Trae el historial de compras previas del usuario desde la DB
        2. Calcula features del usuario (frecuencia, recencia, etc.)
        3. Calcula features de interacción usuario-producto (cuantas veces compró cada uno)
        4. Trae features estadísticas de los productos desde la DB
        5. Filtra productos con poco historial global (< 50 compras totales)
        6. Une todas las features en una sola matriz por par usuario-producto
        7. Calcula features derivadas (up_reorder_rate, up_orders_since_last_purchase)
        8. Asigna clusters de usuario y producto con KMeans

        Cada fila de la matriz resultante representa un producto candidato
        a ser recomendado al usuario.
        """
        # 1. Historial de compras del usuario
        prior = self._query_user_prior(user_id)
        if prior.empty:
            raise UserNotFoundError(f"user_id {user_id} no existe en fact_order_products para get_eval='prior'.")

        # Aseguramos tipos numéricos correctos (la DB puede devolver strings en algunos drivers)
        for col in ["user_key", "product_key", "order_key", "order_number", "reordered", "add_to_cart_order"]:
            if col in prior.columns:
                prior[col] = pd.to_numeric(prior[col], errors="coerce")

        # 2. Lista única de productos que compró el usuario
        product_keys = sorted(prior["product_key"].dropna().astype(int).unique().tolist())
        dim_product_user = self._query_user_dim_product(product_keys)
        dim_product_norm = _normalize_dim_product(dim_product_user)  # Normaliza nombres de dept/pasillo

        # 3. Calcula todas las features por separado
        user_feat = get_user_features(prior, min_user_orders=1)           # Features del usuario
        up_feat = get_user_product_features(prior)                         # Features usuario-producto
        user_dept = get_user_department_feature(prior, dim_product_norm)   # Preferencia por departamento
        user_aisle = get_user_aisle_feature(prior, dim_product_norm)       # Preferencia por pasillo

        # 4. Features globales de los productos (calculadas en la DB con toda la historia)
        product_feat = self._query_product_features(product_keys)
        # 5. Filtra productos con muy poco historial global (no son confiables para el modelo)
        product_feat = product_feat.dropna(subset=["product_total_purchases"]).copy()
        product_feat = product_feat[product_feat["product_total_purchases"] >= 50].copy()
        if product_feat.empty:
            raise UserNotFoundError(
                f"user_id {user_id} no tiene productos con historial suficiente para inferencia."
            )

        # Filtra la matriz de interacciones para quedarse solo con productos válidos
        valid_products = set(product_feat["product_key"].astype(int).tolist())
        matrix = up_feat[up_feat["product_key"].isin(valid_products)].copy()
        if matrix.empty:
            raise UserNotFoundError(
                f"user_id {user_id} no tiene pares usuario-producto válidos tras filtros de inferencia."
            )

        # 6. Une todas las features en una sola matriz
        matrix = matrix.merge(user_feat, on="user_key", how="left")
        matrix = matrix.merge(product_feat, on="product_key", how="left")
        matrix = matrix.merge(user_dept, on="user_key", how="left")
        matrix = matrix.merge(user_aisle, on="user_key", how="left")

        # 7. Features derivadas: tasa de recompra del par usuario-producto
        matrix["up_reorder_rate"] = (
            matrix["up_times_purchased"] / matrix["user_total_orders"]
        ).astype("float32")
        # Cuántas órdenes pasaron desde la última vez que compró este producto
        matrix["up_orders_since_last_purchase"] = (
            matrix["user_total_orders"] - matrix["up_last_order_number"]
        ).clip(lower=0).astype("int16")

        # 8. Asigna cluster de usuario y de producto con los modelos KMeans entrenados
        cluster_models = self._artifacts.cluster_models
        matrix = self._assign_user_cluster(
            matrix=matrix,
            user_id=user_id,
            scaler_user=cluster_models["scaler_user"],
            kmeans_user=cluster_models["kmeans_user"],
            user_profiles_ref=cluster_models["user_profiles"],
        )
        matrix = self._assign_product_clusters(
            matrix=matrix,
            scaler_product=cluster_models["scaler_product"],
            kmeans_product=cluster_models["kmeans_product"],
            product_profiles_ref=cluster_models["product_profiles"],
        )

        return matrix

    def _align_and_validate(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """Verifica que la matriz tenga exactamente las columnas que espera el modelo.

        El modelo LightGBM fue entrenado con un conjunto fijo de features (feature_cols).
        Si la matriz online tiene columnas de más, de menos, o en distinto orden, la
        predicción sería incorrecta. Esta función actúa como contrato de calidad.
        """
        # Detecta si falta alguna feature que el modelo entrenado requiere
        missing = [col for col in self.feature_cols if col not in matrix.columns]
        if missing:
            raise FeatureContractError(f"Columnas faltantes para inferencia: {missing}")

        # Selecciona y ordena las columnas exactamente igual que en el entrenamiento
        X = matrix[self.feature_cols].copy()
        if X.shape[1] != self.n_features:
            raise FeatureContractError(
                f"n_features inválido: esperado={self.n_features}, recibido={X.shape[1]}"
            )

        return X

    def recommend_user(self, user_id: int, top_k: int = 10) -> List[dict]:
        """Genera las top_k recomendaciones de productos para un usuario.

        Si el usuario tiene 0 órdenes prior → lanza UserNotFoundError (404).
        Si el usuario tiene entre 1 y MIN_ORDERS_FOR_MODEL-1 órdenes (cold-start) →
            devuelve el ranking de sus productos más comprados (sin usar el modelo ML).
        Si el usuario tiene >= MIN_ORDERS_FOR_MODEL órdenes → flujo completo de inferencia:
            1. Construye la matriz de features online
            2. Alinea y valida las features contra el contrato del modelo
            3. Predice la probabilidad de recompra con LightGBM
            4. Ordena por probabilidad descendente y toma los top_k
            5. Enriquece con el nombre del producto y devuelve como lista de dicts

        Retorna una lista de dicts con product_key, product_name y probability.
        """
        # Verifica si el usuario tiene historia y si hay suficiente para el modelo completo
        n_orders = self._query_user_order_count(user_id)
        if n_orders == 0:
            raise UserNotFoundError(
                f"user_id {user_id} no existe en fact_order_products para get_eval='prior'."
            )
        if n_orders < MIN_ORDERS_FOR_MODEL:
            # Cold-start: devuelve los productos más comprados en lugar de usar el modelo
            return self._cold_start_top_products(user_id, top_k)

        # 1. Construye la matriz de features (una fila por producto candidato)
        matrix = self._build_online_matrix(user_id)
        # 2. Valida y selecciona solo las columnas que el modelo espera
        X = self._align_and_validate(matrix)

        # 3. Predice: predict_proba devuelve [prob_clase_0, prob_clase_1]
        #    Tomamos [:, 1] = probabilidad de que el usuario recompre ese producto
        probabilities = self._artifacts.model.predict_proba(X)[:, 1]
        scored = matrix[["product_key"]].copy()
        scored["probability"] = probabilities
        # 4. Ordena de mayor a menor probabilidad y se queda con los top_k
        scored = scored.sort_values("probability", ascending=False).head(top_k)

        # 5. Agrega el nombre del producto (consulta adicional a la DB)
        product_names = (
            self._query_user_dim_product(scored["product_key"].astype(int).tolist())
            .loc[:, ["product_key", "product_name"]]
            .drop_duplicates(subset=["product_key"])
        )
        scored = scored.merge(product_names, on="product_key", how="left")

        # Serializa a lista de dicts (compatible con JSON para la respuesta HTTP)
        return [
            {
                "product_key": int(row.product_key),
                "product_name": None if pd.isna(row.product_name) else str(row.product_name),
                "probability": float(row.probability),
            }
            for row in scored.itertuples(index=False)
        ]
