"""
src/features/feature_engineering.py
=====================================
RESPONSABILIDAD: transformar los DataFrames del schema dimensional en el
feature matrix listo para entrenamiento.

Este módulo NO abre conexión a la base de datos, NO entrena modelos y
NO imputa nulos con criterio estadístico — los NaN intencionales son
consumidos directamente por LightGBM y CatBoost. Su único trabajo es
calcular features desde prior y construir el label desde train.

Cambios cloud-native (Fargate):
  - Eliminado os.makedirs('./reports/logs') — el FS del contenedor es read-only.
  - Eliminado logging.FileHandler — los logs van a stdout y CloudWatch los captura.
  - build_feature_matrix: output_path=None por defecto — no escribe a disco local.
    Para guardar en S3, el caller debe subir el DataFrame resultante.

Flujo de datos
--------------
    load_data_from_aws()  →  dict[str, pd.DataFrame]
            ↓  [este módulo]
    feature_matrix (DataFrame)   ← tiene NaN intencionales
            ↓  [train.py]
    modelos LightGBM / CatBoost  ← manejan NaN de forma nativa

Unidad de observación
---------------------
    Cada fila es un par (user_key, product_key) observado en prior.
    El label indica si ese par apareció en la orden de train (eval_set='train').

Sobre el eval_set de Instacart
-------------------------------
    prior = historial de compras → se usa para calcular TODAS las features
    train = última orden real    → se usa SOLO para construir el label
    test  = sin ground truth     → se ignora completamente

    Regla clave: prior entra como input, train solo se toca para el label.
    Mezclarlos generaría data leakage.

Columnas del feature matrix (26 total)
---------------------------------------
    IDs             : user_key, product_key

    Usuario (6)     : user_total_orders, user_avg_basket_size,
                      user_days_since_last_order,
                      user_reorder_ratio,        ← extensión
                      user_distinct_products,    ← extensión
                      user_segment_code          ← extensión

    Producto (7)    : product_total_purchases, product_reorder_rate,
                      product_avg_add_to_cart,   ← extensión
                      product_unique_users,      ← extensión
                      p_department_reorder_rate, ← extensión
                      p_aisle_reorder_rate       ← extensión

    Interacción u×p (10):
                      up_times_purchased, up_reorder_rate,
                      up_orders_since_last_purchase, up_first_order_number,
                      up_last_order_number, up_avg_add_to_cart_order,
                      up_days_since_last,         ← extensión
                      up_avg_days_between_orders, ← extensión (NaN intencional)
                      up_delta_days,              ← extensión (NaN intencional)
                      u_favorite_department,      ← extensión
                      u_favorite_aisle            ← extensión

    Label           : label

NaN intencionales — manejados de forma nativa por LightGBM y CatBoost
---------------------------------------------------------------------
    p_department_reorder_rate  : NaN si dim_product no tiene department_key
    p_aisle_reorder_rate       : NaN si dim_product no tiene aisle_key
    up_avg_days_between_orders : NaN cuando up_times_purchased == 1
    up_delta_days              : NaN donde up_avg_days_between_orders es NaN

Filtros aplicados
-----------------
    - Solo eval_set = 'prior' para features (sin leakage)
    - Usuarios con >= MIN_USER_ORDERS órdenes prior (default: 5)
    - Productos con >= MIN_PRODUCT_ORDERS compras en prior (default: 50)
    - Label construido desde eval_set = 'train'
"""

import logging
import sys
import os
import numpy as np
import pandas as pd
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Logging — stdout únicamente (Fargate / CloudWatch compatible)
#
# CAMBIO CLOUD-NATIVE: se eliminaron os.makedirs('./reports/logs') y
# logging.FileHandler('./reports/logs/feature_engineering.log').
# El sistema de archivos del contenedor Fargate es read-only excepto /tmp/.
# CloudWatch Logs captura stdout/stderr automáticamente desde el awslogs driver.
# ---------------------------------------------------------------------------
logger = logging.getLogger("feature_engineering")
logger.setLevel(logging.DEBUG)
logger.propagate = False  # Evita duplicados hacia el logger raíz de Python

if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setLevel(logging.DEBUG)
    _handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    logger.addHandler(_handler)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
MIN_USER_ORDERS    = 5    # usuarios con menos órdenes tienen historial insuficiente
MIN_PRODUCT_ORDERS = 50   # productos con menos compras tienen tasas de reorden ruidosas

SEGMENT_BINS  = [0, 5, 10, 20, 40, 101]
SEGMENT_CODES = [1, 2, 3, 4, 5]

FEATURE_MATRIX_COLUMNS = [
    "user_key", "product_key",
    "user_total_orders", "user_avg_basket_size", "user_days_since_last_order",
    "user_reorder_ratio", "user_distinct_products", "user_segment_code",
    "product_total_purchases", "product_reorder_rate",
    "product_avg_add_to_cart", "product_unique_users",
    "p_department_reorder_rate", "p_aisle_reorder_rate",
    "up_times_purchased", "up_reorder_rate",
    "up_orders_since_last_purchase", "up_first_order_number", "up_last_order_number",
    "up_avg_add_to_cart_order",
    "up_days_since_last", "up_avg_days_between_orders", "up_delta_days",
    "u_favorite_department", "u_favorite_aisle",
    "label",
]

NAN_INTENCIONALES = {"up_avg_days_between_orders", "up_delta_days"}

NULLABLE_COLUMNS = [
    "p_department_reorder_rate",
    "p_aisle_reorder_rate",
    "up_avg_days_between_orders",
    "up_delta_days",
]


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _log_step(label: str, df: pd.DataFrame) -> None:
    logger.info("[%s] %10d filas | %.1f MB", label, len(df),
                df.memory_usage(deep=True).sum() / 1e6)


def _get_prior(fact: pd.DataFrame) -> pd.DataFrame:
    return fact[fact["get_eval"] == "prior"].copy()


def _get_train(fact: pd.DataFrame) -> pd.DataFrame:
    return fact[fact["get_eval"] == "train"].copy()


# ---------------------------------------------------------------------------
# Normalización de dim_product
# ---------------------------------------------------------------------------

def _normalize_dim_product(dim_product: pd.DataFrame) -> pd.DataFrame:
    """Garantiza que dim_product tenga department_key y aisle_key como int.

    Maneja los tres casos posibles:
      - Ya tiene _key  → no hace nada.
      - Tiene _name    → genera _key con cat.codes.
      - No tiene nada  → warning; features de esa dimensión quedan NaN.
    """
    dim_product = dim_product.copy()

    if "department_key" not in dim_product.columns:
        if "department_name" in dim_product.columns:
            logger.info("dim_product: generando department_key desde department_name")
            dim_product["department_key"] = (
                dim_product["department_name"].astype("category").cat.codes.astype("int8")
            )
        else:
            logger.warning("dim_product sin department_key ni department_name — features de departamento omitidas")

    if "aisle_key" not in dim_product.columns:
        if "aisle_name" in dim_product.columns:
            logger.info("dim_product: generando aisle_key desde aisle_name")
            dim_product["aisle_key"] = (
                dim_product["aisle_name"].astype("category").cat.codes.astype("int16")
            )
        elif "aisle_id" in dim_product.columns:
            dim_product["aisle_key"] = dim_product["aisle_id"].astype("int16")
        else:
            logger.warning("dim_product sin aisle_key ni aisle_name — features de aisle omitidas")

    return dim_product


# ---------------------------------------------------------------------------
# Features de usuario
# ---------------------------------------------------------------------------

def get_user_features(
    prior: pd.DataFrame,
    min_user_orders: int = MIN_USER_ORDERS,
) -> pd.DataFrame:
    """Features a nivel usuario calculadas sobre prior.

    Base: user_total_orders, user_avg_basket_size, user_days_since_last_order.
    Extensiones: user_reorder_ratio, user_distinct_products, user_segment_code.
    Filtro: usuarios con >= min_user_orders órdenes prior.
    """
    logger.info("Calculando features de usuario...")

    basket = (
        prior.groupby(["user_key", "order_key"])["product_key"]
        .count()
        .reset_index(name="basket_size")
    )
    user_feat = (
        basket.groupby("user_key")
        .agg(
            user_total_orders    =("order_key",   "nunique"),
            user_avg_basket_size =("basket_size", "mean"),
        )
        .reset_index()
    )

    before = len(user_feat)
    user_feat = user_feat[user_feat["user_total_orders"] >= min_user_orders]
    logger.info("  Filtro min_user_orders=%d: %d → %d usuarios",
                min_user_orders, before, len(user_feat))

    last_order = (
        prior.sort_values("order_number")
        .groupby("user_key")["days_since_prior_order"]
        .last()
        .reset_index(name="user_days_since_last_order")
    )
    user_feat = user_feat.merge(last_order, on="user_key", how="left")

    user_reorder = (
        prior.groupby("user_key")["reordered"]
        .mean()
        .reset_index(name="user_reorder_ratio")
    )
    user_feat = user_feat.merge(user_reorder, on="user_key", how="left")

    user_distinct = (
        prior.groupby("user_key")["product_key"]
        .nunique()
        .reset_index(name="user_distinct_products")
    )
    user_feat = user_feat.merge(user_distinct, on="user_key", how="left")

    user_feat["user_segment_code"] = pd.cut(
        user_feat["user_total_orders"],
        bins=SEGMENT_BINS, labels=SEGMENT_CODES, right=True,
    ).astype("float").fillna(0).astype("int8")

    user_feat["user_key"]                   = user_feat["user_key"].astype("int32")
    user_feat["user_total_orders"]          = user_feat["user_total_orders"].astype("int16")
    user_feat["user_avg_basket_size"]       = user_feat["user_avg_basket_size"].astype("float32")
    user_feat["user_days_since_last_order"] = user_feat["user_days_since_last_order"].astype("float32")
    user_feat["user_reorder_ratio"]         = user_feat["user_reorder_ratio"].astype("float32")
    user_feat["user_distinct_products"]     = user_feat["user_distinct_products"].astype("int32")

    _log_step("user_features", user_feat)
    return user_feat


# ---------------------------------------------------------------------------
# Features de producto
# ---------------------------------------------------------------------------

def get_product_features(
    prior: pd.DataFrame,
    dim_product: pd.DataFrame,
    min_product_orders: int = MIN_PRODUCT_ORDERS,
) -> pd.DataFrame:
    """Features a nivel producto calculadas sobre prior.

    Base: product_total_purchases, product_reorder_rate.
    Extensiones: product_avg_add_to_cart, product_unique_users,
                 p_department_reorder_rate, p_aisle_reorder_rate.
    Filtro: productos con >= min_product_orders compras en prior.
    """
    logger.info("Calculando features de producto...")

    prod_feat = (
        prior.groupby("product_key")
        .agg(
            product_total_purchases=("order_key",         "count"),
            product_reorder_rate   =("reordered",         "mean"),
            product_avg_add_to_cart=("add_to_cart_order", "mean"),
            product_unique_users   =("user_key",          "nunique"),
        )
        .reset_index()
    )

    before = len(prod_feat)
    prod_feat = prod_feat[prod_feat["product_total_purchases"] >= min_product_orders]
    logger.info("  Filtro min_product_orders=%d: %d → %d productos",
                min_product_orders, before, len(prod_feat))

    if "department_key" in dim_product.columns:
        prior_dept   = prior.merge(dim_product[["product_key", "department_key"]],
                                   on="product_key", how="left")
        dept_reorder = (
            prior_dept.groupby("department_key")["reordered"]
            .mean()
            .reset_index(name="p_department_reorder_rate")
        )
        prod_with_dept = dim_product[["product_key", "department_key"]].drop_duplicates()
        prod_with_dept = prod_with_dept.merge(dept_reorder, on="department_key", how="left")
        prod_feat = prod_feat.merge(
            prod_with_dept[["product_key", "p_department_reorder_rate"]],
            on="product_key", how="left",
        )
    else:
        prod_feat["p_department_reorder_rate"] = np.nan
        logger.warning("dim_product sin department_key — p_department_reorder_rate = NaN")

    if "aisle_key" in dim_product.columns:
        prior_aisle  = prior.merge(dim_product[["product_key", "aisle_key"]],
                                   on="product_key", how="left")
        aisle_reorder = (
            prior_aisle.groupby("aisle_key")["reordered"]
            .mean()
            .reset_index(name="p_aisle_reorder_rate")
        )
        prod_with_aisle = dim_product[["product_key", "aisle_key"]].drop_duplicates()
        prod_with_aisle = prod_with_aisle.merge(aisle_reorder, on="aisle_key", how="left")
        prod_feat = prod_feat.merge(
            prod_with_aisle[["product_key", "p_aisle_reorder_rate"]],
            on="product_key", how="left",
        )
    else:
        prod_feat["p_aisle_reorder_rate"] = np.nan
        logger.warning("dim_product sin aisle_key — p_aisle_reorder_rate = NaN")

    prod_feat["product_key"]               = prod_feat["product_key"].astype("int32")
    prod_feat["product_total_purchases"]   = prod_feat["product_total_purchases"].astype("int32")
    prod_feat["product_reorder_rate"]      = prod_feat["product_reorder_rate"].astype("float32")
    prod_feat["product_avg_add_to_cart"]   = prod_feat["product_avg_add_to_cart"].astype("float32")
    prod_feat["product_unique_users"]      = prod_feat["product_unique_users"].astype("int32")
    prod_feat["p_department_reorder_rate"] = prod_feat["p_department_reorder_rate"].astype("float32")
    prod_feat["p_aisle_reorder_rate"]      = prod_feat["p_aisle_reorder_rate"].astype("float32")

    _log_step("product_features", prod_feat)
    return prod_feat


# ---------------------------------------------------------------------------
# Features de interacción usuario × producto
# ---------------------------------------------------------------------------

def get_user_product_features(prior: pd.DataFrame) -> pd.DataFrame:
    """Features de interacción usuario × producto calculadas sobre prior.

    Base: up_times_purchased, up_first_order_number, up_last_order_number,
          up_avg_add_to_cart_order.
    Extensiones: up_days_since_last, up_avg_days_between_orders, up_delta_days.
    """
    logger.info("Calculando features u×p...")

    up = (
        prior.groupby(["user_key", "product_key"])
        .agg(
            up_times_purchased      =("order_key",         "count"),
            up_first_order_number   =("order_number",      "min"),
            up_last_order_number    =("order_number",      "max"),
            up_avg_add_to_cart_order=("add_to_cart_order", "mean"),
        )
        .reset_index()
    )

    user_orders = (
        prior[["user_key", "order_number", "days_since_prior_order"]]
        .drop_duplicates()
    )
    after_last = user_orders.merge(
        up[["user_key", "product_key", "up_last_order_number"]],
        on="user_key", how="left",
    )
    after_last = after_last[after_last["order_number"] > after_last["up_last_order_number"]]
    after_last = after_last.dropna(subset=["days_since_prior_order"])
    days_since = (
        after_last.groupby(["user_key", "product_key"])["days_since_prior_order"]
        .sum()
        .reset_index(name="up_days_since_last")
    )
    up = up.merge(days_since, on=["user_key", "product_key"], how="left")
    up["up_days_since_last"] = up["up_days_since_last"].fillna(0).astype("float32")

    up_order_days = (
        prior.merge(up[["user_key", "product_key"]], on=["user_key", "product_key"], how="inner")
        .groupby(["user_key", "product_key"])["days_since_prior_order"]
        .sum()
        .reset_index(name="_total_days_span")
    )
    up = up.merge(up_order_days, on=["user_key", "product_key"], how="left")
    up["up_avg_days_between_orders"] = (
        up["_total_days_span"] / (up["up_times_purchased"] - 1)
    ).replace([np.inf, -np.inf], np.nan).astype("float32")

    n_nan = up["up_avg_days_between_orders"].isna().sum()
    if n_nan > 0:
        logger.info("  up_avg_days_between_orders: %d NaN (up_times_purchased==1) "
                    "— manejados nativamente por LightGBM/CatBoost", n_nan)

    up["up_delta_days"] = (
        up["up_days_since_last"] - up["up_avg_days_between_orders"]
    ).astype("float32")

    up = up.drop(columns=["_total_days_span"])

    up["user_key"]                  = up["user_key"].astype("int32")
    up["product_key"]               = up["product_key"].astype("int32")
    up["up_times_purchased"]        = up["up_times_purchased"].astype("int16")
    up["up_first_order_number"]     = up["up_first_order_number"].astype("int16")
    up["up_last_order_number"]      = up["up_last_order_number"].astype("int16")
    up["up_avg_add_to_cart_order"]  = up["up_avg_add_to_cart_order"].astype("float32")

    _log_step("up_features", up)
    return up


# ---------------------------------------------------------------------------
# Departamento y aisle favorito del usuario
# ---------------------------------------------------------------------------

def get_user_department_feature(
    prior: pd.DataFrame,
    dim_product: pd.DataFrame,
) -> pd.DataFrame:
    """Extensión: u_favorite_department (int8)."""
    logger.info("Calculando u_favorite_department...")
    if "department_key" not in dim_product.columns:
        logger.warning("dim_product sin department_key — u_favorite_department omitido")
        return pd.DataFrame(columns=["user_key", "u_favorite_department"])

    prior_dept = prior.merge(
        dim_product[["product_key", "department_key"]], on="product_key", how="left"
    )
    user_dept = (
        prior_dept.groupby(["user_key", "department_key"])["order_key"]
        .count()
        .reset_index(name="dept_count")
    )
    user_fav = (
        user_dept.sort_values("dept_count", ascending=False)
        .groupby("user_key").first()
        .reset_index()[["user_key", "department_key"]]
        .rename(columns={"department_key": "u_favorite_department"})
    )
    user_fav["user_key"]              = user_fav["user_key"].astype("int32")
    user_fav["u_favorite_department"] = user_fav["u_favorite_department"].astype("int8")
    return user_fav


def get_user_aisle_feature(
    prior: pd.DataFrame,
    dim_product: pd.DataFrame,
) -> pd.DataFrame:
    """Extensión: u_favorite_aisle (int16)."""
    logger.info("Calculando u_favorite_aisle...")
    if "aisle_key" not in dim_product.columns:
        logger.warning("dim_product sin aisle_key — u_favorite_aisle omitido")
        return pd.DataFrame(columns=["user_key", "u_favorite_aisle"])

    prior_aisle = prior.merge(
        dim_product[["product_key", "aisle_key"]], on="product_key", how="left"
    )
    user_aisle = (
        prior_aisle.groupby(["user_key", "aisle_key"])["order_key"]
        .count()
        .reset_index(name="aisle_count")
    )
    user_fav = (
        user_aisle.sort_values("aisle_count", ascending=False)
        .groupby("user_key").first()
        .reset_index()[["user_key", "aisle_key"]]
        .rename(columns={"aisle_key": "u_favorite_aisle"})
    )
    user_fav["user_key"]         = user_fav["user_key"].astype("int32")
    user_fav["u_favorite_aisle"] = user_fav["u_favorite_aisle"].astype("int16")
    return user_fav


# ---------------------------------------------------------------------------
# Label
# ---------------------------------------------------------------------------

def get_label(train: pd.DataFrame) -> pd.DataFrame:
    """Construye el label desde eval_set = 'train'.

    label=1 → el usuario reordenó el producto en su última orden real.
    label=0 → ausencia en train (fillna(0) en build_feature_matrix).
    """
    logger.info("Construyendo label desde train...")
    label_df = (
        train[["user_key", "product_key"]]
        .drop_duplicates()
        .assign(label=1)
    )
    label_df["user_key"]    = label_df["user_key"].astype("int32")
    label_df["product_key"] = label_df["product_key"].astype("int32")
    label_df["label"]       = label_df["label"].astype("int8")
    logger.info("  Pares positivos (label=1): %d", len(label_df))
    return label_df


def _check_leakage(prior: pd.DataFrame, train: pd.DataFrame) -> None:
    """Verifica que no haya data leakage entre prior y train."""
    logger.info("Verificando ausencia de leakage prior → train...")

    prior_max = (
        prior.groupby("user_key")["order_number"]
        .max()
        .reset_index(name="prior_max_order")
    )
    train_order = (
        train.groupby("user_key")["order_number"]
        .max()
        .reset_index(name="train_order")
    )
    check      = prior_max.merge(train_order, on="user_key", how="inner")
    violations = check[check["prior_max_order"] >= check["train_order"]]

    if len(violations) > 0:
        logger.error("LEAKAGE DETECTADO: %d usuarios con prior_max_order >= train_order",
                     len(violations))
        raise ValueError(
            f"Data leakage en {len(violations)} usuarios. "
            "Revisar _get_prior() y _get_train()."
        )

    train_counts = train.groupby("user_key")["order_number"].nunique()
    multi_train  = train_counts[train_counts > 1]
    if len(multi_train) > 0:
        logger.warning("%d usuarios con más de una orden en train — revisar eval_set",
                       len(multi_train))

    logger.info("OK — %d usuarios verificados, sin leakage", len(check))


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def build_feature_matrix(
    data:               Dict[str, pd.DataFrame],
    min_user_orders:    int           = MIN_USER_ORDERS,
    min_product_orders: int           = MIN_PRODUCT_ORDERS,
    output_path:        Optional[str] = None,   # CAMBIO: None por defecto (sin escritura local)
) -> pd.DataFrame:
    """Construye el feature matrix completo desde los DataFrames.

    Parameters
    ----------
    data : dict[str, pd.DataFrame]
        Requiere: 'fact_order_products', 'dim_product'.
    min_user_orders : int
        Mínimo de órdenes prior por usuario (default: 5).
    min_product_orders : int
        Mínimo de compras en prior por producto (default: 50).
    output_path : str | None
        Ruta local de salida del parquet.
        - None (default en Fargate): no escribe a disco.
        - Ruta explícita: escribe el parquet localmente.
          En entornos cloud, escribir a /tmp/ o subir el resultado a S3.

    Returns
    -------
    pd.DataFrame con NaN intencionales en columnas nullable.
    """
    logger.info("=" * 60)
    logger.info("Iniciando build_feature_matrix v4 (cloud-native)")
    logger.info("  min_user_orders=%d | min_product_orders=%d",
                min_user_orders, min_product_orders)
    logger.info("=" * 60)

    fact        = data["fact_order_products"]
    dim_product = _normalize_dim_product(data.get("dim_product", pd.DataFrame()))
    prior       = _get_prior(fact)
    train       = _get_train(fact)

    _check_leakage(prior, train)

    # ── Features ──────────────────────────────────────────────────────────────
    user_feat  = get_user_features(prior, min_user_orders)
    prod_feat  = get_product_features(prior, dim_product, min_product_orders)
    up_feat    = get_user_product_features(prior)
    user_dept  = get_user_department_feature(prior, dim_product)
    user_aisle = get_user_aisle_feature(prior, dim_product)
    label_df   = get_label(train)

    # ── Filtro de pares válidos ────────────────────────────────────────────────
    valid_users    = set(user_feat["user_key"])
    valid_products = set(prod_feat["product_key"])
    df = up_feat[
        up_feat["user_key"].isin(valid_users) &
        up_feat["product_key"].isin(valid_products)
    ].copy()
    logger.info("Pares u×p después de filtros: %d", len(df))

    # ── Merge ─────────────────────────────────────────────────────────────────
    df = df.merge(user_feat,  on="user_key",    how="left")
    df = df.merge(prod_feat,  on="product_key", how="left")
    df = df.merge(user_dept,  on="user_key",    how="left")
    df = df.merge(user_aisle, on="user_key",    how="left")

    # ── Features derivadas ────────────────────────────────────────────────────
    df["up_reorder_rate"] = (
        df["up_times_purchased"] / df["user_total_orders"]
    ).astype("float32")
    df["up_orders_since_last_purchase"] = (
        df["user_total_orders"] - df["up_last_order_number"]
    ).clip(lower=0).astype("int16")

    # ── Label ─────────────────────────────────────────────────────────────────
    df = df.merge(label_df, on=["user_key", "product_key"], how="left")
    df["label"] = df["label"].fillna(0).astype("int8")

    n_pos   = int((df["label"] == 1).sum())
    n_neg   = int((df["label"] == 0).sum())
    ratio   = n_neg / max(n_pos, 1)
    logger.info("Label=1: %d | Label=0: %d | Ratio 0:1 = %.2f:1", n_pos, n_neg, ratio)
    logger.info("→ scale_pos_weight recomendado para LightGBM: %.2f", ratio)

    # ── Orden canónico de columnas ─────────────────────────────────────────────
    df = df[[c for c in FEATURE_MATRIX_COLUMNS if c in df.columns]]

    # ── Imputaciones de negocio ───────────────────────────────────────────────
    FILLNA_ZERO = {
        "up_days_since_last": "0 días = producto comprado en la última orden",
        "label":              "0 = no apareció en train",
        "user_segment_code":  "0 = caso borde fuera de bins",
    }
    for col, razon in FILLNA_ZERO.items():
        if col in df.columns:
            n = df[col].isna().sum()
            if n > 0:
                df[col] = df[col].fillna(0)
                logger.info("  fillna(0) en %s: %d nulos — %s", col, n, razon)

    # ── Reporte de NaN intencionales ──────────────────────────────────────────
    nan_report = {c: int(df[c].isna().sum()) for c in NULLABLE_COLUMNS if c in df.columns}
    nan_con_nulos = {c: n for c, n in nan_report.items() if n > 0}
    if nan_con_nulos:
        logger.info("Columnas con NaN intencionales (LightGBM/CatBoost los manejan nativamente):")
        for col, n in nan_con_nulos.items():
            logger.info("  %s: %d NaN (%.1f%%)", col, n, n / len(df) * 100)

    # ── Duplicados ────────────────────────────────────────────────────────────
    dupes = int(df.duplicated(subset=["user_key", "product_key"]).sum())
    if dupes > 0:
        df = df.drop_duplicates(subset=["user_key", "product_key"])
        logger.warning("Duplicados eliminados: %d", dupes)

    # ── Resumen ───────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Feature matrix: %d filas x %d columnas", len(df), len(df.columns))
    logger.info("Usuarios únicos : %d", df["user_key"].nunique())
    logger.info("Productos únicos: %d", df["product_key"].nunique())
    logger.info("Memoria         : %.1f MB", df.memory_usage(deep=True).sum() / 1e6)
    logger.info("=" * 60)

    # ── Guardar (opcional) ────────────────────────────────────────────────────
    # CAMBIO: output_path=None por defecto — en Fargate no se escribe a disco local.
    # Si se requiere persistencia, el caller debe pasar output_path='/tmp/...'
    # o subir el DataFrame directamente a S3.
    if output_path:
        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        df.to_parquet(output_path, index=False)
        logger.info("Guardado en %s", output_path)

    return df