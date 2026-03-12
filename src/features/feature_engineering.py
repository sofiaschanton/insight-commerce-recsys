"""
src/features/feature_engineering.py
=====================================
RESPONSABILIDAD: transformar los DataFrames del schema dimensional en el
feature matrix listo para entrenamiento.

Este módulo NO abre conexión a la base de datos, NO entrena modelos y
NO imputa nulos con criterio estadístico — los NaN intencionales son
consumidos directamente por LightGBM y CatBoost. Su único trabajo es
calcular features desde prior y construir el label desde train.

Flujo de datos
--------------
    load_data_from_neon()  →  dict[str, pd.DataFrame]
            ↓  [este módulo]
    data/processed/feature_matrix.parquet   ← tiene NaN intencionales
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
                                 (sin ciclo de recompra calculable)
    up_delta_days              : NaN donde up_avg_days_between_orders es NaN

    Estos NaN NO se imputan aquí porque imputar con 0 sería incorrecto
    semánticamente. LightGBM y CatBoost aprenden splits óptimos con NaN.

Filtros aplicados
-----------------
    - Solo eval_set = 'prior' para features (sin leakage)
    - Usuarios con >= MIN_USER_ORDERS órdenes prior (default: 5)
    - Productos con >= MIN_PRODUCT_ORDERS compras en prior (default: 50)
    - Label construido desde eval_set = 'train'

Uso
---
    from src.data.data_loader import load_data_from_neon
    from src.features.feature_engineering import build_feature_matrix

    data   = load_data_from_neon()
    matrix = build_feature_matrix(data)
    # → guarda feature_matrix.parquet con NaN intencionales
    # → siguiente paso: train.py
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional

# ─── Logging ──────────────────────────────────────────────────────────────────
os.makedirs('./reports/logs', exist_ok=True)

logger = logging.getLogger('feature_engineering')
logger.setLevel(logging.DEBUG)
logger.propagate = False

if not logger.handlers:
    fmt = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh = logging.FileHandler('./reports/logs/feature_engineering.log', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)

# ─── Constantes ───────────────────────────────────────────────────────────────
# Ajustar según el tamaño de la muestra y la distribución del dataset.
# Valores más bajos = más pares pero features menos confiables.
# Valores más altos = features más confiables pero menos cobertura.
MIN_USER_ORDERS    = 5    # usuarios con menos órdenes tienen historial insuficiente
MIN_PRODUCT_ORDERS = 50   # productos con menos compras tienen tasas de reorden ruidosas

# Segmentos de frecuencia de compra (user_segment_code)
# Diseño: bins asimétricos porque la distribución de órdenes está sesgada a la derecha.
# 1=esporádico(1-5) 2=ocasional(6-10) 3=regular(11-20) 4=frecuente(21-40) 5=power_user(41+)
SEGMENT_BINS  = [0, 5, 10, 20, 40, 101]
SEGMENT_CODES = [1, 2, 3, 4, 5]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _log_step(label: str, df: pd.DataFrame) -> None:
    """Loguea filas y memoria de un DataFrame intermedio para monitorear el pipeline."""
    logger.info(f"[{label}] {len(df):>10,} filas | {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")


def _get_prior(fact: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra solo las órdenes de historial (eval_set = 'prior').
    Todo el cálculo de features se hace sobre prior — nunca sobre train.
    """
    return fact[fact['get_eval'] == 'prior'].copy()


def _get_train(fact: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra la última orden real de cada usuario (eval_set = 'train').
    Se usa ÚNICAMENTE para construir el label — no para features.
    Mezclar prior y train para features generaría data leakage.
    """
    return fact[fact['get_eval'] == 'train'].copy()


# ─── Features de usuario ──────────────────────────────────────────────────────
# Capturan el comportamiento general del usuario a lo largo de todo su historial.
# Son las mismas para todos los productos del mismo usuario — se calculan una vez
# y se mergean al feature matrix en build_feature_matrix.

def get_user_features(
    prior: pd.DataFrame,
    min_user_orders: int = MIN_USER_ORDERS,
) -> pd.DataFrame:
    """
    Features a nivel usuario calculadas sobre prior.

    Base del equipo:
        user_total_orders         : órdenes distintas en prior
        user_avg_basket_size      : promedio de productos por orden
        user_days_since_last_order: días entre penúltima y última orden

    Extensiones:
        user_reorder_ratio        : % de compras que fueron reorden
                                    (perfil de hábito global del usuario)
        user_distinct_products    : productos únicos comprados en prior
                                    (alto = explorador, bajo = rutinario)
        user_segment_code         : segmento de frecuencia 1-5

    Filtro: usuarios con >= min_user_orders órdenes prior.
    """
    logger.info("Calculando features de usuario...")

    # Basket size: contamos productos por orden primero, luego promediamos por usuario
    basket = (
        prior.groupby(['user_key', 'order_key'])['product_key']
        .count()
        .reset_index(name='basket_size')
    )
    user_feat = (
        basket.groupby('user_key')
        .agg(
            user_total_orders    = ('order_key',   'nunique'),
            user_avg_basket_size = ('basket_size', 'mean'),
        )
        .reset_index()
    )

    # Filtro mínimo de órdenes — usuarios con menos historial se excluyen
    # porque sus features son estadísticamente poco confiables
    before = len(user_feat)
    user_feat = user_feat[user_feat['user_total_orders'] >= min_user_orders]
    logger.info(f"  Filtro min_user_orders={min_user_orders}: {before:,} -> {len(user_feat):,} usuarios")

    # days_since_last_order: tomamos el valor de la última orden prior
    # Instacart no tiene fechas absolutas, solo días desde la orden anterior
    last_order = (
        prior.sort_values('order_number')
        .groupby('user_key')['days_since_prior_order']
        .last()
        .reset_index(name='user_days_since_last_order')
    )
    user_feat = user_feat.merge(last_order, on='user_key', how='left')

    # Extensión: reorder_ratio — media de la columna 'reordered' de Instacart
    # 'reordered' = 1 si el producto ya había sido comprado antes en prior
    # NO confundir con el label — este mira hacia atrás dentro de prior (sin leakage)
    user_reorder = (
        prior.groupby('user_key')['reordered']
        .mean()
        .reset_index(name='user_reorder_ratio')
    )
    user_feat = user_feat.merge(user_reorder, on='user_key', how='left')

    # Extensión: distinct_products — complementa reorder_ratio
    # dos usuarios con el mismo ratio pueden tener catálogos muy distintos
    user_distinct = (
        prior.groupby('user_key')['product_key']
        .nunique()
        .reset_index(name='user_distinct_products')
    )
    user_feat = user_feat.merge(user_distinct, on='user_key', how='left')

    # Extensión: segment_code — agrupa usuarios por frecuencia de compra
    # fillna(0) correcto aquí: caso borde cuando total_orders no cae en ningún bin
    user_feat['user_segment_code'] = pd.cut(
        user_feat['user_total_orders'],
        bins=SEGMENT_BINS, labels=SEGMENT_CODES, right=True
    ).astype('float').fillna(0).astype('int8')

    user_feat['user_key']                   = user_feat['user_key'].astype('int32')
    user_feat['user_total_orders']          = user_feat['user_total_orders'].astype('int16')
    user_feat['user_avg_basket_size']       = user_feat['user_avg_basket_size'].astype('float32')
    user_feat['user_days_since_last_order'] = user_feat['user_days_since_last_order'].astype('float32')
    user_feat['user_reorder_ratio']         = user_feat['user_reorder_ratio'].astype('float32')
    user_feat['user_distinct_products']     = user_feat['user_distinct_products'].astype('int32')

    _log_step('user_features', user_feat)
    return user_feat


# ─── Normalización de dim_product ────────────────────────────────────────────
# Neon guarda department_name y aisle_name como strings en dim_product,
# sin columnas de clave numérica separadas. Esta función genera department_key
# y aisle_key automáticamente usando category codes, para que el resto del
# pipeline no dependa del formato exacto de Neon.
#
# Decisión de diseño: usamos cat.codes en vez de un mapeo fijo porque el
# dataset de Instacart tiene IDs estables y no necesitamos consistencia
# entre runs. Si en el futuro se despliega en producción con datos nuevos,
# habría que guardar el mapeo en un artefacto separado.

def _normalize_dim_product(dim_product: pd.DataFrame) -> pd.DataFrame:
    """
    Garantiza que dim_product tenga department_key y aisle_key como int,
    independientemente de si Neon los tiene como columnas separadas
    o aplanados en department_name / aisle_name.

    Casos que maneja:
        - Tiene _key                      → no hace nada
        - Tiene _name pero no _key        → genera _key desde _name con cat.codes
        - Tiene aisle_id (legacy)         → renombra a aisle_key
        - No tiene ninguno                → warning, features de esa dimensión quedan NaN
    """
    dim_product = dim_product.copy()

    # department_key — 21 departamentos en Instacart (int8 suficiente)
    if 'department_key' not in dim_product.columns:
        if 'department_name' in dim_product.columns:
            logger.info("  dim_product: generando department_key desde department_name")
            dim_product['department_key'] = (
                dim_product['department_name'].astype('category').cat.codes.astype('int8')
            )
            logger.info(f"  {dim_product['department_key'].nunique()} departamentos codificados")
        else:
            logger.warning("  dim_product sin department_key ni department_name — features de departamento omitidas")

    # aisle_key — ~134 aisles en Instacart (int16 porque el rango > 127)
    if 'aisle_key' not in dim_product.columns:
        if 'aisle_name' in dim_product.columns:
            logger.info("  dim_product: generando aisle_key desde aisle_name")
            dim_product['aisle_key'] = (
                dim_product['aisle_name'].astype('category').cat.codes.astype('int16')
            )
            logger.info(f"  {dim_product['aisle_key'].nunique()} aisles codificados")
        elif 'aisle_id' in dim_product.columns:
            # Compatibilidad con versiones anteriores que usaban aisle_id
            dim_product['aisle_key'] = dim_product['aisle_id'].astype('int16')
        else:
            logger.warning("  dim_product sin aisle_key ni aisle_name — features de aisle omitidas")

    return dim_product


# ─── Features de producto ─────────────────────────────────────────────────────
# Capturan la popularidad y comportamiento global del producto en todo el dataset.
# Son las mismas para todos los usuarios del mismo producto.
# Las features de departamento y aisle requieren dim_product con claves numéricas
# — si no están disponibles quedan como NaN (ver _normalize_dim_product).

def get_product_features(
    prior: pd.DataFrame,
    dim_product: pd.DataFrame,
    min_product_orders: int = MIN_PRODUCT_ORDERS,
) -> pd.DataFrame:
    """
    Features a nivel producto calculadas sobre prior.

    Base del equipo (2 features):
        product_total_purchases   : total de compras en prior
                                    (volumen absoluto de demanda)
        product_reorder_rate      : media de 'reordered' — tasa de reorden global
                                    (productos de consumo habitual tienen rate alto)

    Extensiones (5 features):
        product_avg_add_to_cart   : posición promedio en el carrito
                                    valores cercanos a 1 = producto de hábito
                                    (los usuarios lo agregan primero, automáticamente)
        product_unique_users      : usuarios distintos que lo compraron
                                    (diferencia con product_total_purchases:
                                     muchas compras de pocos usuarios vs. pocos de muchos)
        p_department_reorder_rate : tasa de reorden promedio del departamento
                                    (contexto de categoría — lácteos > snacks)
        p_aisle_reorder_rate      : tasa de reorden promedio del aisle
                                    (más granular que departamento)

    Filtro: productos con >= min_product_orders compras en prior.
    Productos raros tienen tasas de reorden estadísticamente ruidosas.

    NaN: p_department_reorder_rate y p_aisle_reorder_rate quedan NaN si
    dim_product no tiene las claves. LightGBM y CatBoost los manejan de forma nativa.
    """
    logger.info("Calculando features de producto...")

    prod_feat = (
        prior.groupby('product_key')
        .agg(
            product_total_purchases = ('order_key',         'count'),
            product_reorder_rate    = ('reordered',         'mean'),
            product_avg_add_to_cart = ('add_to_cart_order', 'mean'),
            product_unique_users    = ('user_key',          'nunique'),
        )
        .reset_index()
    )

    # Filtro mínimo — productos con poco historial tienen rates estadísticamente ruidosas
    before = len(prod_feat)
    prod_feat = prod_feat[prod_feat['product_total_purchases'] >= min_product_orders]
    logger.info(f"  Filtro min_product_orders={min_product_orders}: {before:,} -> {len(prod_feat):,} productos")

    # p_department_reorder_rate: tasa media del departamento al que pertenece el producto
    # Requiere department_key en dim_product (generado por _normalize_dim_product si no existe)
    if 'department_key' in dim_product.columns:
        prior_dept   = prior.merge(dim_product[['product_key', 'department_key']],
                                    on='product_key', how='left')
        dept_reorder = (
            prior_dept.groupby('department_key')['reordered']
            .mean()
            .reset_index(name='p_department_reorder_rate')
        )
        prod_with_dept = dim_product[['product_key', 'department_key']].drop_duplicates()
        prod_with_dept = prod_with_dept.merge(dept_reorder, on='department_key', how='left')
        prod_feat = prod_feat.merge(
            prod_with_dept[['product_key', 'p_department_reorder_rate']],
            on='product_key', how='left'
        )
        logger.info(f"  p_department_reorder_rate media={prod_feat['p_department_reorder_rate'].mean():.3f}")
    else:
        # NaN intencional — 0 significaría tasa de reorden cero, que es falso
        prod_feat['p_department_reorder_rate'] = np.nan
        logger.warning("  dim_product sin department_key — p_department_reorder_rate = NaN")

    # p_aisle_reorder_rate: tasa media del aisle — más granular que departamento
    # (~134 aisles vs 21 departamentos en Instacart)
    if 'aisle_key' in dim_product.columns:
        prior_aisle  = prior.merge(dim_product[['product_key', 'aisle_key']],
                                    on='product_key', how='left')
        aisle_reorder = (
            prior_aisle.groupby('aisle_key')['reordered']
            .mean()
            .reset_index(name='p_aisle_reorder_rate')
        )
        prod_with_aisle = dim_product[['product_key', 'aisle_key']].drop_duplicates()
        prod_with_aisle = prod_with_aisle.merge(aisle_reorder, on='aisle_key', how='left')
        prod_feat = prod_feat.merge(
            prod_with_aisle[['product_key', 'p_aisle_reorder_rate']],
            on='product_key', how='left'
        )
        logger.info(f"  p_aisle_reorder_rate media={prod_feat['p_aisle_reorder_rate'].mean():.3f}")
    else:
        # NaN intencional — misma razón que p_department_reorder_rate
        prod_feat['p_aisle_reorder_rate'] = np.nan
        logger.warning("  dim_product sin aisle_key — p_aisle_reorder_rate = NaN")

    prod_feat['product_key']               = prod_feat['product_key'].astype('int32')
    prod_feat['product_total_purchases']   = prod_feat['product_total_purchases'].astype('int32')
    prod_feat['product_reorder_rate']      = prod_feat['product_reorder_rate'].astype('float32')
    prod_feat['product_avg_add_to_cart']   = prod_feat['product_avg_add_to_cart'].astype('float32')
    prod_feat['product_unique_users']      = prod_feat['product_unique_users'].astype('int32')
    prod_feat['p_department_reorder_rate'] = prod_feat['p_department_reorder_rate'].astype('float32')
    prod_feat['p_aisle_reorder_rate']      = prod_feat['p_aisle_reorder_rate'].astype('float32')

    _log_step('product_features', prod_feat)
    return prod_feat


# ─── Features de interacción usuario × producto ───────────────────────────────
# Son las features más importantes del modelo — capturan la relación específica
# entre un usuario y un producto en particular.
# Un usuario puede tener user_reorder_ratio alto pero no reordenar ciertos productos.
# Estas features capturan esa granularidad que las features de usuario/producto solos no pueden.

def get_user_product_features(prior: pd.DataFrame) -> pd.DataFrame:
    """
    Features de interacción usuario × producto calculadas sobre prior.

    Base del equipo (4 features):
        up_times_purchased       : veces que el usuario compró el producto
                                   (fidelidad directa al producto)
        up_first_order_number    : número de orden de la primera compra
                                   (qué tan temprano adoptó el producto)
        up_last_order_number     : número de orden de la última compra
                                   (qué tan reciente fue la última compra)
        up_avg_add_to_cart_order : posición promedio en el carrito del usuario
                                   para este producto específico
                                   (cercano a 1 = hábito automatizado)

    Extensiones (3 features):
        up_days_since_last         : días acumulados desde la última compra del par
                                     hasta el fin de prior.
                                     fillna(0) correcto: 0 = comprado en la última orden.

        up_avg_days_between_orders : ciclo promedio de recompra en días
                                     = total_days_span / (times_purchased - 1)
                                     NaN cuando times_purchased == 1 — sin ciclo calculable.
                                     NO imputar con 0 (significaría ciclo instantáneo).
                                     → LightGBM/CatBoost aprenden el split óptimo con NaN.

        up_delta_days              : up_days_since_last - up_avg_days_between_orders
                                     > 0 : ya pasó el ciclo → señal fuerte de reorden
                                     < 0 : comprado muy recientemente → poco probable
                                     NaN donde up_avg_days_between_orders es NaN.
                                     → LightGBM/CatBoost manejan NaN de forma nativa.

    Features derivadas calculadas en build_feature_matrix (no aquí):
        up_reorder_rate              = up_times_purchased / user_total_orders
        up_orders_since_last_purchase = user_total_orders - up_last_order_number
    """
    logger.info("Calculando features u×p...")

    up = (
        prior.groupby(['user_key', 'product_key'])
        .agg(
            up_times_purchased       = ('order_key',         'count'),
            up_first_order_number    = ('order_number',      'min'),
            up_last_order_number     = ('order_number',      'max'),
            up_avg_add_to_cart_order = ('add_to_cart_order', 'mean'),
        )
        .reset_index()
    )

    # up_days_since_last: suma los días de todas las órdenes del usuario
    # POSTERIORES a la última compra del par.
    # Importante: se parte de órdenes del usuario (sin filtrar por producto)
    # porque necesitamos órdenes donde el producto NO fue comprado también.
    # La primera orden siempre tiene days_since_prior_order = NaN → se dropea.
    user_orders = (
        prior[['user_key', 'order_number', 'days_since_prior_order']]
        .drop_duplicates()
    )
    after_last = user_orders.merge(
        up[['user_key', 'product_key', 'up_last_order_number']],
        on='user_key', how='left'
    )
    after_last = after_last[after_last['order_number'] > after_last['up_last_order_number']]
    after_last = after_last.dropna(subset=['days_since_prior_order'])
    days_since = (
        after_last.groupby(['user_key', 'product_key'])['days_since_prior_order']
        .sum()
        .reset_index(name='up_days_since_last')
    )
    up = up.merge(days_since, on=['user_key', 'product_key'], how='left')
    up['up_days_since_last'] = up['up_days_since_last'].fillna(0).astype('float32')

    # up_avg_days_between_orders: total de días acumulados del par / (veces comprado - 1)
    # La división por 0 ocurre cuando times_purchased == 1 → NaN intencional.
    # No imputar con 0 porque significaría "ciclo de recompra instantáneo" — incorrecto.
    # LightGBM y CatBoost aprenden el split óptimo con NaN.
    up_order_days = (
        prior.merge(up[['user_key', 'product_key']], on=['user_key', 'product_key'], how='inner')
        .groupby(['user_key', 'product_key'])['days_since_prior_order']
        .sum()
        .reset_index(name='_total_days_span')
    )
    up = up.merge(up_order_days, on=['user_key', 'product_key'], how='left')
    up['up_avg_days_between_orders'] = (
        up['_total_days_span'] / (up['up_times_purchased'] - 1)
    ).replace([np.inf, -np.inf], np.nan).astype('float32')
    # NaN cuando up_times_purchased == 1 (comprado solo una vez, sin ciclo de recompra)
    # NO imputar con 0 (significaría ciclo instantáneo) — LightGBM/CatBoost manejan NaN de forma nativa.
    n_nan = up['up_avg_days_between_orders'].isna().sum()
    if n_nan > 0:
        logger.info(f"  up_avg_days_between_orders: {n_nan:,} NaN (up_times_purchased==1) — manejados nativamente por el modelo")

    # up_delta_days: señal clave para el modelo
    # > 0 = ya pasó el ciclo de recompra esperado → alta probabilidad de reorden
    # < 0 = comprado muy recientemente → baja probabilidad de reorden
    # Hereda los NaN de up_avg_days_between_orders — LightGBM/CatBoost los manejan de forma nativa
    up['up_delta_days'] = (
        up['up_days_since_last'] - up['up_avg_days_between_orders']
    ).astype('float32')

    up = up.drop(columns=['_total_days_span'])

    up['user_key']                  = up['user_key'].astype('int32')
    up['product_key']               = up['product_key'].astype('int32')
    up['up_times_purchased']        = up['up_times_purchased'].astype('int16')
    up['up_first_order_number']     = up['up_first_order_number'].astype('int16')
    up['up_last_order_number']      = up['up_last_order_number'].astype('int16')
    up['up_avg_add_to_cart_order']  = up['up_avg_add_to_cart_order'].astype('float32')

    _log_step('up_features', up)
    return up


# ─── Departamento y aisle favorito del usuario ────────────────────────────────
# Estas dos features capturan la afinidad del usuario con categorías de productos.
# Se usan tanto como features del modelo como señal en el Componente 2 de
# recomendación (descubrimiento de productos nuevos en recommendation.py).
# Se mantienen ambas porque capturan granularidades distintas: un usuario puede
# tener departamento favorito = 'produce' pero aisle favorito = 'fresh herbs',
# que es una señal más precisa para recomendar productos nuevos.

def get_user_department_feature(
    prior: pd.DataFrame,
    dim_product: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extensión: u_favorite_department
        Departamento donde el usuario concentra más compras en prior.
        Guardado como int8 (código numérico).
    """
    logger.info("Calculando u_favorite_department...")
    if 'department_key' not in dim_product.columns:
        logger.warning("  dim_product sin department_key — u_favorite_department omitido")
        return pd.DataFrame(columns=['user_key', 'u_favorite_department'])

    prior_dept = prior.merge(
        dim_product[['product_key', 'department_key']], on='product_key', how='left'
    )
    user_dept = (
        prior_dept.groupby(['user_key', 'department_key'])['order_key']
        .count()
        .reset_index(name='dept_count')
    )
    # Tomar el departamento con mayor count de compras por usuario
    user_fav = (
        user_dept.sort_values('dept_count', ascending=False)
        .groupby('user_key').first()
        .reset_index()[['user_key', 'department_key']]
        .rename(columns={'department_key': 'u_favorite_department'})
    )
    user_fav['user_key']              = user_fav['user_key'].astype('int32')
    user_fav['u_favorite_department'] = user_fav['u_favorite_department'].astype('int8')
    return user_fav


def get_user_aisle_feature(
    prior: pd.DataFrame,
    dim_product: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extensión: u_favorite_aisle
        Aisle donde el usuario concentra más compras en prior.
        Más granular que u_favorite_department (~130 aisles vs 21 departamentos).
        Guardado como int16.
    """
    logger.info("Calculando u_favorite_aisle...")
    if 'aisle_key' not in dim_product.columns:
        logger.warning("  dim_product sin aisle_key — u_favorite_aisle omitido")
        return pd.DataFrame(columns=['user_key', 'u_favorite_aisle'])

    prior_aisle = prior.merge(
        dim_product[['product_key', 'aisle_key']], on='product_key', how='left'
    )
    user_aisle = (
        prior_aisle.groupby(['user_key', 'aisle_key'])['order_key']
        .count()
        .reset_index(name='aisle_count')
    )
    # Tomar el aisle con mayor count de compras por usuario
    user_fav = (
        user_aisle.sort_values('aisle_count', ascending=False)
        .groupby('user_key').first()
        .reset_index()[['user_key', 'aisle_key']]
        .rename(columns={'aisle_key': 'u_favorite_aisle'})
    )
    user_fav['user_key']         = user_fav['user_key'].astype('int32')
    user_fav['u_favorite_aisle'] = user_fav['u_favorite_aisle'].astype('int16')
    return user_fav


# ─── Label ────────────────────────────────────────────────────────────────────
# El label es la variable objetivo del modelo — lo que queremos predecir.
# Pregunta: dado el historial de (usuario, producto) en prior,
# ¿apareció ese par en la última orden real del usuario (train)?
#
# label = 1 → el usuario reordenó el producto en train
# label = 0 → el usuario NO reordenó (o nunca lo compró) en train
#
# El dataset está muy desbalanceado — la mayoría de los pares tienen label=0.
# El ratio real se imprime en el log del pipeline.
# LightGBM maneja el desbalance con scale_pos_weight en train.py.

def get_label(train: pd.DataFrame) -> pd.DataFrame:
    """
    Construye el label desde eval_set = 'train'.

    Proceso:
        1. Tomar todos los pares (user_key, product_key) que aparecen en train
        2. Asignarles label = 1
        3. En build_feature_matrix, el merge left convierte pares ausentes
           en NaN → fillna(0) → label = 0

    fillna(0) para el label ES correcto: ausencia en train = no reordenó.
    No es un dato faltante — es información real del comportamiento del usuario.
    """
    logger.info("Construyendo label desde train...")
    label_df = (
        train[['user_key', 'product_key']]
        .drop_duplicates()
        .assign(label=1)
    )
    label_df['user_key']    = label_df['user_key'].astype('int32')
    label_df['product_key'] = label_df['product_key'].astype('int32')
    label_df['label']       = label_df['label'].astype('int8')
    logger.info(f"  Pares positivos (label=1): {len(label_df):,}")
    return label_df


def _check_leakage(prior: pd.DataFrame, train: pd.DataFrame) -> None:
    """
    Verifica que no haya data leakage entre prior y train.

    La única información temporal disponible es order_number.
    La regla es estricta: para cada usuario, el order_number máximo
    de prior debe ser menor al order_number de train.

    Si algún usuario viola esto significa que órdenes de train
    entraron en el cálculo de features → leakage.

    Se llama al inicio de build_feature_matrix antes de calcular
    cualquier feature.
    """
    logger.info("Verificando ausencia de leakage prior → train...")

    # Máximo order_number en prior por usuario
    prior_max = (
        prior.groupby('user_key')['order_number']
        .max()
        .reset_index(name='prior_max_order')
    )

    # order_number en train por usuario (debe ser exactamente uno por usuario)
    train_order = (
        train.groupby('user_key')['order_number']
        .max()
        .reset_index(name='train_order')
    )

    check = prior_max.merge(train_order, on='user_key', how='inner')

    # Violaciones: prior_max >= train_order
    violations = check[check['prior_max_order'] >= check['train_order']]

    if len(violations) > 0:
        logger.error(f"  LEAKAGE DETECTADO: {len(violations):,} usuarios con prior_max_order >= train_order")
        logger.error(f"  Ejemplos:\n{violations.head(5).to_string()}")
        raise ValueError(
            f"Data leakage detectado en {len(violations):,} usuarios. "
            f"Revisar _get_prior() y _get_train() — prior no debe incluir la última orden."
        )

    # Verificación adicional: cada usuario debe tener exactamente una orden en train
    train_counts = train.groupby('user_key')['order_number'].nunique()
    multi_train  = train_counts[train_counts > 1]
    if len(multi_train) > 0:
        logger.warning(f"  {len(multi_train):,} usuarios con más de una orden en train — revisar eval_set")

    logger.info(f"  OK — {len(check):,} usuarios verificados, sin leakage detectado")



# Función principal que orquesta todo el pipeline de features.
# Parte de up_feat (pares u×p) como base y agrega features de usuario/producto
# como columnas adicionales. El orden de los merges importa — siempre left join
# desde up_feat para no perder pares.

def build_feature_matrix(
    data:               Dict[str, pd.DataFrame],
    min_user_orders:    int = MIN_USER_ORDERS,
    min_product_orders: int = MIN_PRODUCT_ORDERS,
    output_path:        Optional[str] = './data/processed/feature_matrix.parquet',
) -> pd.DataFrame:
    """
    Construye el feature matrix completo desde los DataFrames.

    Parameters
    ----------
    data : dict[str, pd.DataFrame]
        Requiere: 'fact_order_products', 'dim_product'.
        Nota: 'dim_user' no se usa — user_age excluida (datos sintéticos).
    min_user_orders : int
        Mínimo de órdenes prior por usuario. Default: 5.
    min_product_orders : int
        Mínimo de compras en prior por producto. Default: 50.
    output_path : str | None
        Ruta de salida del parquet. None = no guarda.

    Returns
    -------
    pd.DataFrame
        26 columnas (2 IDs + 6 usuario + 7 producto + 10 u×p + 1 label).
        Puede tener NaN en: p_department_reorder_rate, p_aisle_reorder_rate,
        up_avg_days_between_orders, up_delta_days.
        El log final detalla cuántos NaN hay en cada columna.
        Siguiente paso: train.py — LightGBM/CatBoost manejan los NaN de forma nativa.
    """
    logger.info("=" * 60)
    logger.info("Iniciando build_feature_matrix v4")
    logger.info(f"  min_user_orders={min_user_orders} | min_product_orders={min_product_orders}")
    logger.info("=" * 60)

    # Separar fact en prior y train — son dos conjuntos completamente distintos
    fact        = data['fact_order_products']
    dim_product = _normalize_dim_product(data.get('dim_product', pd.DataFrame()))
    prior       = _get_prior(fact)   # → features
    train       = _get_train(fact)   # → solo label

    # Verificar leakage antes de calcular cualquier feature
    # Falla ruidosamente si prior contiene órdenes de train
    _check_leakage(prior, train)

    # ── 1. Calcular features ──────────────────────────────────────────────────
    user_feat  = get_user_features(prior, min_user_orders)
    prod_feat  = get_product_features(prior, dim_product, min_product_orders)
    up_feat    = get_user_product_features(prior)
    user_dept  = get_user_department_feature(prior, dim_product)
    user_aisle = get_user_aisle_feature(prior, dim_product)
    label_df   = get_label(train)

    # ── 2. Filtrar pares u×p válidos ──────────────────────────────────────────
    valid_users    = set(user_feat['user_key'])
    valid_products = set(prod_feat['product_key'])
    df = up_feat[
        up_feat['user_key'].isin(valid_users) &
        up_feat['product_key'].isin(valid_products)
    ].copy()
    logger.info(f"Pares u×p después de filtros: {len(df):,}")

    # ── 3. Merge features ─────────────────────────────────────────────────────
    # Left join: up_feat es la base, se agregan columnas de cada dimensión.
    # Todos los pares se mantienen — si falta una feature queda NaN (ver paso 7).
    df = df.merge(user_feat,   on='user_key',    how='left')
    df = df.merge(prod_feat,   on='product_key', how='left')
    df = df.merge(user_dept,   on='user_key',    how='left')
    df = df.merge(user_aisle,  on='user_key',    how='left')

    # ── 4. Features derivadas ─────────────────────────────────────────────────
    # Se calculan acá porque dependen de columnas de distintos grupos (up + usuario).

    # up_reorder_rate: frecuencia del par normalizada por actividad del usuario.
    # Nota: up_purchase_frequency y up_order_rate del equipo son la misma fórmula
    # — eliminadas para evitar colinealidad.
    df['up_reorder_rate'] = (
        df['up_times_purchased'] / df['user_total_orders']
    ).astype('float32')

    # up_orders_since_last_purchase: órdenes del usuario desde la última compra del par.
    # clip(lower=0) evita negativos por inconsistencias menores de datos.
    df['up_orders_since_last_purchase'] = (
        df['user_total_orders'] - df['up_last_order_number']
    ).clip(lower=0).astype('int16')

    # ── 5. Label ──────────────────────────────────────────────────────────────
    # Left join: pares en train → label=1, pares ausentes → NaN → fillna(0) → label=0.
    # fillna(0) ES correcto aquí: ausencia en train = no reordenó.
    df = df.merge(label_df, on=['user_key', 'product_key'], how='left')
    df['label'] = df['label'].fillna(0).astype('int8')

    n_pos   = int((df['label'] == 1).sum())
    n_neg   = int((df['label'] == 0).sum())
    n_total = len(df)
    ratio   = n_neg / max(n_pos, 1)
    logger.info("─" * 60)
    logger.info("Desbalance de clases:")
    logger.info(f"  label=1 (reordenó)    : {n_pos:>10,}  ({n_pos/n_total*100:.1f}%)")
    logger.info(f"  label=0 (no reordenó) : {n_neg:>10,}  ({n_neg/n_total*100:.1f}%)")
    logger.info(f"  Ratio 0:1             : {ratio:.2f}:1")
    logger.info(f"  → scale_pos_weight recomendado para LightGBM: {ratio:.2f}")
    logger.info("─" * 60)

    # ── 6. Orden de columnas ──────────────────────────────────────────────────
    # Orden explícito para que el parquet sea legible y reproducible.
    # Las columnas ausentes (si una feature falló) se omiten silenciosamente.
    COLUMN_ORDER = [
        # IDs
        'user_key', 'product_key',
        # Usuario — base (3) + extensiones (3)
        'user_total_orders', 'user_avg_basket_size', 'user_days_since_last_order',
        'user_reorder_ratio', 'user_distinct_products', 'user_segment_code',
        # Producto — base (2) + extensiones (5)
        'product_total_purchases', 'product_reorder_rate',
        'product_avg_add_to_cart', 'product_unique_users',
        'p_department_reorder_rate', 'p_aisle_reorder_rate',
        # Interacción u×p — base (6) + extensiones (4)
        'up_times_purchased', 'up_reorder_rate',
        'up_orders_since_last_purchase', 'up_first_order_number', 'up_last_order_number',
        'up_avg_add_to_cart_order',
        'up_days_since_last', 'up_avg_days_between_orders', 'up_delta_days',
        'u_favorite_department', 'u_favorite_aisle',
        # Label
        'label',
    ]
    df = df[[c for c in COLUMN_ORDER if c in df.columns]]

    # ── 7. Validaciones finales ───────────────────────────────────────────────

    # Columnas que se imputan con 0 por lógica del negocio
    FILLNA_ZERO = {
        'up_days_since_last':   '0 días = producto comprado en la última orden',
        'label':                '0 = no apareció en train',
        'user_segment_code':    '0 = caso borde fuera de bins',
    }
    for col, razon in FILLNA_ZERO.items():
        if col in df.columns:
            n = df[col].isna().sum()
            if n > 0:
                df[col] = df[col].fillna(0)
                logger.info(f"  fillna(0) en {col}: {n:,} nulos — {razon}")

    # Columnas con NaN intencionales — manejados de forma nativa por LightGBM y CatBoost
    NAN_INTENCIONALES = [
        'p_department_reorder_rate',
        'p_aisle_reorder_rate',
        'up_avg_days_between_orders',
        'up_delta_days',
    ]
    nan_report = {col: int(df[col].isna().sum()) for col in NAN_INTENCIONALES if col in df.columns}
    nan_con_nulos = {col: n for col, n in nan_report.items() if n > 0}
    if nan_con_nulos:
        logger.info("  Columnas con NaN intencionales — manejados por LightGBM/CatBoost:")
        for col, n in nan_con_nulos.items():
            pct = n / len(df) * 100
            logger.info(f"    {col}: {n:,} NaN ({pct:.1f}%)")
    else:
        logger.info("  Sin NaN intencionales en el feature matrix")

    # Cualquier otro nulo inesperado — warning explícito, NO fillna silencioso
    remaining_nulls = df.isnull().sum()
    remaining_nulls = remaining_nulls[remaining_nulls > 0]
    unexpected = [c for c in remaining_nulls.index if c not in NAN_INTENCIONALES]
    if unexpected:
        logger.warning(f"  NaN inesperados en columnas no contempladas: {unexpected}")
        logger.warning("  Revisar pipeline — estos NaN NO se imputan automáticamente")

    dupes = int(df.duplicated(subset=['user_key', 'product_key']).sum())
    if dupes > 0:
        df = df.drop_duplicates(subset=['user_key', 'product_key'])
        logger.warning(f"  Duplicados eliminados: {dupes:,}")

    # ── 8. Resumen ────────────────────────────────────────────────────────────
    label_dist = df['label'].value_counts()
    ratio      = label_dist.get(0, 0) / max(label_dist.get(1, 1), 1)
    logger.info("=" * 60)
    logger.info(f"Feature matrix: {len(df):,} filas x {len(df.columns)} columnas")
    logger.info(f"Usuarios unicos  : {df['user_key'].nunique():,}")
    logger.info(f"Productos unicos : {df['product_key'].nunique():,}")
    logger.info(f"Label=0: {label_dist.get(0,0):,} | Label=1: {label_dist.get(1,0):,} | Ratio: {ratio:.2f}:1")
    logger.info(f"Memoria: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    logger.info("=" * 60)

    # ── 9. Guardar ────────────────────────────────────────────────────────────
    # Parquet preserva tipos, comprime mejor y es más rápido que CSV.
    # El parquet de salida tiene NaN intencionales — LightGBM/CatBoost los manejan de forma nativa.
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_parquet(output_path, index=False)
        logger.info(f"  Guardado en {output_path}")

    return df


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    from src.data.data_loader import load_data_from_neon

    data   = load_data_from_neon()
    matrix = build_feature_matrix(data)

    print("\nFeature matrix generada:")
    print(f"  Shape   : {matrix.shape}")
    print(f"  Columnas: {matrix.columns.tolist()}")
    print(f"\n{matrix.dtypes}")
    print(f"\nPrimeras filas:\n{matrix.head()}")
