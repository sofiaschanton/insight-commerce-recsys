"""
src/features/preprocessing.py
================================
RESPONSABILIDAD: construir y ejecutar el pipeline de preprocesamiento
sobre el feature matrix calculado por feature_engineering.py.

Expone dos funciones principales:

    build_preprocessing_pipeline(...)  → sklearn.Pipeline
        Construye el pipeline de preprocesamiento con los parámetros
        especificados. NO fitea ni transforma — solo define la arquitectura.
        El nombre sigue la convención del enunciado: recibe los parámetros
        de configuración y devuelve un Pipeline listo para fit/transform.

    preprocess(matrix, ...)    → X_train, X_test, y_train, y_test, artifacts
        Orquesta el flujo completo: split → fit en train → transform train/test.
        Usa build_preprocessing_pipeline internamente.

Dos funciones principales
--------------------------
    build_preprocessing_pipeline(...)  → sklearn.Pipeline
        Solo define la arquitectura del pipeline.
        No fitea, no transforma, no hace split.
        Recibe parámetros de configuración y devuelve un Pipeline listo para usar.

    preprocess(matrix, ...)  → X_train, X_test, y_train, y_test, artifacts
        Orquesta el flujo completo:
            feature_matrix.parquet
                ↓  separar X e y
                ↓  split train/test (stratify=label)
                ↓  build_preprocessing_pipeline()  → Pipeline
                ↓  pipe.fit_transform(X_train)      ← aprende caps, mediana, media/std
                ↓  pipe.transform(X_test)           ← aplica lo aprendido, sin leakage
            X_train, X_test, y_train, y_test
            models/preprocessing_pipeline.pkl

Pipeline interno (en orden)
----------------------------
    1. cap          : QuantileCapper
                      aprende caps p1/p99 en train, aplica en test e inference
    2. imputer      : ColumnTransformer con SimpleImputer(strategy)
                      rellena NaN con mediana de train
                      - numéricas  : SimpleImputer
                      - categóricas: passthrough
    3. to_df        : FunctionTransformer
                      reconstruye DataFrame con nombres de columna
                      (necesario para que delta_recalc opere por nombre)
    4. delta_recalc : DeltaDaysRecalculator
                      recalcula up_delta_days = up_days_since_last - up_avg_days_between_orders
                      post-imputación para garantizar consistencia entre columnas relacionadas
    5. scaler       : ColumnTransformer con Scaler(scaling_strategy)
                      escala con media/std aprendida en train
                      - numéricas  : StandardScaler / MinMaxScaler / RobustScaler
                      - categóricas: passthrough

Por qué imputer y scaler van separados
---------------------------------------
    Si fueran un solo ColumnTransformer (imputer + scaler juntos), no habría
    forma de intercalar el recalculo de up_delta_days entre los dos pasos.
    El recalculo necesita los valores ya imputados de up_avg_days_between_orders
    pero antes de que el scaler los transforme.

Columnas que NO se tocan
------------------------
    user_key, product_key  → IDs, se dropean antes del pipeline
    label                  → target (y), sale del pipeline
    categóricas enteras    → user_segment_code, u_favorite_department,
                             u_favorite_aisle — passthrough para LightGBM

Uso
---
    from src.features.preprocessing import build_preprocessing_pipeline, preprocess
    import pandas as pd

    matrix = pd.read_parquet('data/processed/feature_matrix.parquet')

    # Solo construir el pipeline (sin fitear)
    pipe = build_preprocessing_pipeline(
        categorical_features=['user_segment_code', 'u_favorite_department', 'u_favorite_aisle'],
        numerical_features=['user_total_orders', 'up_days_since_last', ...],
        imputation_strategy='median',
        scaling_strategy='standard',
        target_col='label',
    )

    # O correr el flujo completo con split
    X_train, X_test, y_train, y_test, artifacts = preprocess(matrix)
"""

import os
import logging
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, FunctionTransformer
from sklearn.model_selection import train_test_split

# ─── Logging ──────────────────────────────────────────────────────────────────
os.makedirs('./reports/logs', exist_ok=True)

logger = logging.getLogger('preprocessing')
logger.setLevel(logging.DEBUG)
logger.propagate = False

if not logger.handlers:
    fmt = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh = logging.FileHandler('./reports/logs/preprocessing.log', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)


# ─── Defaults ─────────────────────────────────────────────────────────────────
# Valores por defecto para los parámetros del enunciado.
# Se usan cuando no se pasan explícitamente a build_preprocessing_pipeline.

COLS_IDS = ['user_key', 'product_key']
COL_TARGET_DEFAULT = 'label'

DEFAULT_CATEGORICAL_FEATURES = [
    'user_segment_code',
    'u_favorite_department',
    'u_favorite_aisle',
]

DEFAULT_NUMERICAL_FEATURES = [
    'user_total_orders', 'user_avg_basket_size', 'user_days_since_last_order',
    'user_reorder_ratio', 'user_distinct_products',
    'product_total_purchases', 'product_reorder_rate',
    'product_avg_add_to_cart', 'product_unique_users',
    'p_department_reorder_rate', 'p_aisle_reorder_rate',
    'up_times_purchased', 'up_reorder_rate', 'up_orders_since_last_purchase',
    'up_first_order_number', 'up_last_order_number', 'up_avg_add_to_cart_order',
    'up_days_since_last', 'up_avg_days_between_orders', 'up_delta_days',
]

# Columnas con distribución de cola pesada — se capean con QuantileCapper
DEFAULT_HEAVY_TAIL_COLS = [
    'user_avg_basket_size',
    'up_days_since_last',
    'up_avg_days_between_orders',
    'up_avg_add_to_cart_order',
    'up_delta_days',
]

SCALING_STRATEGIES = {
    'standard': StandardScaler,
    'minmax':   MinMaxScaler,
    'robust':   RobustScaler,
}

CATEGORICAL_ENCODINGS = ['passthrough', 'onehot']


# ─── Transformers custom ──────────────────────────────────────────────────────


class DataFrameReconstructor(BaseEstimator, TransformerMixin):
    """
    Reconstruye un DataFrame con nombres de columna desde un array numpy.

    Necesario entre el imputer y el DeltaDaysRecalculator porque:
    - ColumnTransformer devuelve un array numpy sin nombres de columna
    - DeltaDaysRecalculator necesita acceder a columnas por nombre

    Se define como clase (no lambda) para ser serializable con joblib.
    """
    def __init__(self, columns: List[str]):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame(X, columns=self.columns)

class DeltaDaysRecalculator(BaseEstimator, TransformerMixin):
    """
    Recalcula up_delta_days DESPUÉS de que el imputer rellenó
    up_avg_days_between_orders.

    Por qué es necesario aunque up_delta_days ya se calcula en feature_engineering.py
    ---------------------------------------------------------------------------------
    up_delta_days llega con NaN donde up_times_purchased == 1 (55% de los pares).
    Si el imputer lo rellena de forma independiente, puede quedar inconsistente:

        up_days_since_last         =  5   (calculado, correcto)
        up_avg_days_between_orders = 14   (imputado con mediana de train)
        up_delta_days              = 18   (imputado independientemente ← INCORRECTO)
                                          debería ser 5 - 14 = -9

    El modelo vería tres columnas relacionadas con valores contradictorios.

    Solución: recalcular up_delta_days justo después de que el imputer rellenó
    up_avg_days_between_orders, así la resta siempre es consistente.

    No hay leakage — es una resta aritmética pura sobre columnas de prior.
    No aprende nada del target ni de test.

    Posición en el pipeline
    -----------------------
    Debe ir ANTES del QuantileCapper y del ColumnTransformer porque opera
    sobre el DataFrame con nombres de columna. El ColumnTransformer convierte
    a array numpy y los nombres se pierden.

    El orden correcto es:
        1. imputer  (dentro del ColumnTransformer) → rellena up_avg_days_between_orders
        2. delta_recalc                            → recalcula up_delta_days
        3. cap                                     → clipea ambas columnas ya consistentes
        4. scaler                                  → escala

    Nota de implementación: como el ColumnTransformer aplica imputer + scaler
    juntos, usamos un pipeline de dos pasos separados para poder intercalar
    el recalculo entre imputer y scaler:
        pre_impute  → solo imputer
        delta_recalc
        pre_scale   → solo scaler
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if 'up_days_since_last' in X.columns and 'up_avg_days_between_orders' in X.columns:
            X['up_delta_days'] = (
                X['up_days_since_last'] - X['up_avg_days_between_orders']
            ).astype('float32')
        return X


class QuantileCapper(BaseEstimator, TransformerMixin):
    """
    Clipping por cuantiles aprendido en TRAIN y aplicado igual en TEST e inference.

    Aprende los percentiles p1/p99 del conjunto de entrenamiento y los aplica
    consistentemente en cualquier conjunto posterior. Evita que outliers en test
    contaminen los límites de clipping.
    """
    def __init__(self, cols: Optional[List[str]] = None, low_q: float = 0.01, high_q: float = 0.99):
        self.cols   = cols
        self.low_q  = low_q
        self.high_q = high_q

    def fit(self, X, y=None):
        cols = self.cols or X.select_dtypes(include='number').columns.tolist()
        self.caps_ = {}
        for c in cols:
            if c not in X.columns:
                continue
            s = pd.to_numeric(X[c], errors='coerce').dropna()
            if s.empty:
                continue
            lo, hi = s.quantile([self.low_q, self.high_q])
            if lo < hi:
                self.caps_[c] = (float(lo), float(hi))
                logger.debug(f"  cap aprendido {c}: [{lo:.3f}, {hi:.3f}]")
        return self

    def transform(self, X):
        X = X.copy()
        for c, (lo, hi) in self.caps_.items():
            if c in X.columns:
                X[c] = pd.to_numeric(X[c], errors='coerce').clip(lo, hi)
        return X


# ─── Pipeline principal ───────────────────────────────────────────────────────

def build_preprocessing_pipeline(
    categorical_features: Optional[List[str]] = None,
    numerical_features:   Optional[List[str]] = None,
    imputation_strategy:  str = 'median',
    scaling_strategy:     str = 'standard',
    categorical_encoding: str = 'passthrough',
    target_col:           str = 'label',
    cap_cols:             Optional[List[str]] = None,
    low_q:                float = 0.01,
    high_q:               float = 0.99,
) -> Pipeline:
    """
    Construye el pipeline de preprocesamiento del feature matrix.

    Devuelve un sklearn.Pipeline listo para fit/transform — NO fitea ni transforma.
    El split train/test debe hacerse ANTES de llamar a pipe.fit() para evitar leakage.

    Pasos internos del pipeline:
        1. QuantileCapper    → aprende caps p1/p99 en train
        2. ColumnTransformer → imputer + scaler en numéricas,
                               passthrough en categóricas

    Parameters
    ----------
    categorical_features : list | None
        Columnas categóricas — se pasan sin escalar (passthrough).
        LightGBM las maneja directamente como int.
        Default: user_segment_code, u_favorite_department, u_favorite_aisle.
    numerical_features : list | None
        Columnas numéricas — entran al pipeline de imputer + scaler.
        Default: las 20 columnas continuas del feature matrix.
    imputation_strategy : str
        Estrategia de imputación para SimpleImputer.
        Opciones: 'mean', 'median', 'most_frequent', 'constant'.
        Default: 'median' — recomendado para distribuciones asimétricas.
    scaling_strategy : str
        Método de escalado.
        Opciones: 'standard' (StandardScaler), 'minmax' (MinMaxScaler),
                  'robust' (RobustScaler).
        Default: 'standard'. Para LightGBM no es necesario pero permite
        comparar con otros modelos.
    target_col : str
        Nombre de la variable objetivo. Solo se usa para validación —
        el pipeline no la recibe como columna.
        Default: 'label'.
    categorical_encoding : str
        Cómo tratar las columnas categóricas.
        - 'passthrough' : sin transformar — para LightGBM que las maneja como int.
        - 'onehot'      : OneHotEncoder(drop='first') — para modelos que no manejan
                          categóricas (regresión logística, SVM, redes neuronales).
        Default: 'passthrough'.
    cap_cols : list | None
        Columnas a capear con QuantileCapper.
        Default: columnas de cola pesada (días, basket size, add_to_cart).
    low_q, high_q : float
        Percentiles para el capping. Default: p1/p99.

    Returns
    -------
    sklearn.Pipeline
        Pipeline con pasos: delta_recalc → cap → pre.
        Llamar pipe.fit(X_train) y pipe.transform(X_test).
    """
    cat_features = categorical_features or DEFAULT_CATEGORICAL_FEATURES
    num_features = numerical_features   or DEFAULT_NUMERICAL_FEATURES
    heavy_cols   = cap_cols             or DEFAULT_HEAVY_TAIL_COLS

    # Validar scaling_strategy
    if scaling_strategy not in SCALING_STRATEGIES:
        raise ValueError(
            f"scaling_strategy='{scaling_strategy}' no válido. "
            f"Opciones: {list(SCALING_STRATEGIES.keys())}"
        )
    ScalerClass = SCALING_STRATEGIES[scaling_strategy]

    # Validar categorical_encoding
    if categorical_encoding not in CATEGORICAL_ENCODINGS:
        raise ValueError(
            f"categorical_encoding='{categorical_encoding}' no válido. "
            f"Opciones: {CATEGORICAL_ENCODINGS}"
        )
    if categorical_encoding == 'onehot':
        cat_transformer = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
    else:
        cat_transformer = 'passthrough'

    logger.info(f"Construyendo pipeline — imputation={imputation_strategy} | scaling={scaling_strategy} | categorical_encoding={categorical_encoding}")
    logger.info(f"  numerical_features  : {len(num_features)} columnas")
    logger.info(f"  categorical_features: {len(cat_features)} columnas")

    # Nombres de columna después del imputer — necesarios para reconstruir el DataFrame
    # antes de que DeltaDaysRecalculator pueda operar por nombre de columna.
    col_names_after_imputer = num_features + cat_features

    # ── Paso A: solo imputación ───────────────────────────────────────────────
    # up_avg_days_between_orders llega con NaN (up_times_purchased == 1).
    # Se imputa acá con mediana de train. up_delta_days todavía tiene NaN
    # en las mismas filas — se recalcula en el paso C para ser consistente.
    imputer_ct = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy=imputation_strategy), num_features),
            ('cat', cat_transformer, cat_features),
        ],
        remainder='drop',
        verbose_feature_names_out=False,
    )

    # ── Paso B: reconstruir DataFrame con nombres de columna ─────────────────
    # ColumnTransformer devuelve un array numpy sin nombres.
    # DeltaDaysRecalculator necesita acceder a columnas por nombre ('up_days_since_last',
    # 'up_avg_days_between_orders'). DataFrameReconstructor reconstruye el DataFrame.
    # Se usa una clase en lugar de lambda para que el pipeline sea serializable con joblib.
    to_df = DataFrameReconstructor(columns=col_names_after_imputer)

    # ── Paso C: solo escala ───────────────────────────────────────────────────
    # Después del recalculo, escalar todas las columnas numéricas.
    # Aprende media/std solo desde train.
    scaler_ct = ColumnTransformer(
        transformers=[
            ('num', ScalerClass(),  num_features),
            ('cat', cat_transformer,  cat_features),
        ],
        remainder='drop',
        verbose_feature_names_out=False,
    )

    # ── Pipeline final ────────────────────────────────────────────────────────
    # Orden:
    #   1. cap          → clipea outliers (aprende caps en train)
    #   2. imputer      → rellena NaN con mediana de train
    #   3. to_df        → reconstruye DataFrame con nombres de columna
    #   4. delta_recalc → recalcula up_delta_days = up_days_since_last - up_avg_days_between_orders
    #                     ahora consistente: usa el valor imputado de up_avg_days_between_orders
    #   5. scaler       → escala (aprende media/std en train)
    return Pipeline(steps=[
        ('cap',          QuantileCapper(cols=heavy_cols, low_q=low_q, high_q=high_q)),
        ('imputer',      imputer_ct),
        ('to_df',        to_df),
        ('delta_recalc', DeltaDaysRecalculator()),
        ('scaler',       scaler_ct),
    ])


# ─── Flujo completo con split ─────────────────────────────────────────────────

def preprocess(
    matrix:               pd.DataFrame,
    categorical_features: Optional[List[str]] = None,
    numerical_features:   Optional[List[str]] = None,
    imputation_strategy:  str = 'median',
    scaling_strategy:     str = 'standard',
    categorical_encoding: str = 'passthrough',
    target_col:           str = 'label',
    cap_cols:             Optional[List[str]] = None,
    low_q:                float = 0.01,
    high_q:               float = 0.99,
    test_size:            float = 0.2,
    random_state:         int   = 42,
    pipeline_path:        Optional[str] = 'models/preprocessing_pipeline.pkl',
    verbose:              bool  = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, Any]]:
    """
    Flujo completo: split → fit en train → transform train y test.

    Usa build_preprocessing_pipeline internamente para construir el pipeline.
    El split se hace ANTES del fit para evitar leakage — el scaler,
    el imputer y el capper aprenden SOLO desde train.

    Parameters
    ----------
    matrix : pd.DataFrame
        Feature matrix — output de feature_engineering.build_preprocessing_pipeline().
    categorical_features, numerical_features, imputation_strategy,
    scaling_strategy, target_col, cap_cols, low_q, high_q :
        Se pasan directamente a build_preprocessing_pipeline. Ver su docstring.
    test_size : float
        Proporción para test. Default: 0.2.
    random_state : int
        Semilla para reproducibilidad. Default: 42.
    pipeline_path : str | None
        Ruta para guardar el pipeline. None = no persiste.

    Returns
    -------
    X_train, X_test : pd.DataFrame
        Features preprocesadas con nombres de columna.
    y_train, y_test : pd.Series
        Target alineado con X_train/X_test.
    artifacts : dict
        pipeline, feature_names, split_config, balance de clases,
        scale_pos_weight para LightGBM.
    """
    if verbose:
        logger.info("=" * 60)
        logger.info("Iniciando preprocess()")
        logger.info(f"  Shape entrada        : {matrix.shape}")
        logger.info(f"  imputation_strategy  : {imputation_strategy}")
        logger.info(f"  scaling_strategy     : {scaling_strategy}")
        logger.info(f"  test_size            : {test_size} | random_state: {random_state}")
        logger.info("=" * 60)

    # ── 1. Separar IDs, X e y ─────────────────────────────────────────────────
    y = matrix[target_col].reset_index(drop=True)
    X = matrix.drop(columns=COLS_IDS + [target_col], errors='ignore').reset_index(drop=True)

    logger.info(f"[1/5] X: {X.shape} | y: {y.shape}")
    logger.info(f"      Label=0: {(y==0).sum():,} | Label=1: {(y==1).sum():,} | "
                f"Ratio: {(y==0).sum() / max((y==1).sum(), 1):.2f}:1")

    # ── 2. Split train/test ───────────────────────────────────────────────────
    # stratify=y mantiene la proporción de clases en ambos splits.
    # Se hace ANTES del fit para evitar leakage.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    X_train, y_train = X_train.align(y_train, axis=0, join='inner')
    X_test,  y_test  = X_test.align(y_test,   axis=0, join='inner')

    assert len(X_train) == len(y_train), "Desalineado train"
    assert len(X_test)  == len(y_test),  "Desalineado test"

    dist_train = y_train.value_counts(normalize=True).round(4).to_dict()
    dist_test  = y_test.value_counts(normalize=True).round(4).to_dict()
    logger.info(f"[2/5] Split — train: {X_train.shape} | test: {X_test.shape}")
    logger.info(f"      Distribución train: {dist_train}")
    logger.info(f"      Distribución test : {dist_test}")

    # ── 3. Construir pipeline y fit en train ──────────────────────────────────
    pipe = build_preprocessing_pipeline(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        imputation_strategy=imputation_strategy,
        scaling_strategy=scaling_strategy,
        categorical_encoding=categorical_encoding,
        target_col=target_col,
        cap_cols=cap_cols,
        low_q=low_q,
        high_q=high_q,
    )

    logger.info("[3/5] Fit del pipeline en train (imputer + capper + scaler)...")
    X_train_arr = pipe.fit_transform(X_train, y_train)

    # ── 4. Transform test ─────────────────────────────────────────────────────
    X_test_arr = pipe.transform(X_test)
    logger.info("[4/5] Transform en test completado")

    # ── 5. Reconstruir DataFrames ─────────────────────────────────────────────
    # Los nombres de columna se construyen explícitamente desde los parámetros
    # en lugar de depender de get_feature_names_out() — más robusto porque el
    # pipeline tiene pasos intermedios (DataFrameReconstructor) que pueden
    # romper la cadena de nombres de sklearn.
    _num = numerical_features or DEFAULT_NUMERICAL_FEATURES
    _cat = categorical_features or DEFAULT_CATEGORICAL_FEATURES
    feature_names = np.array(_num + _cat)

    X_train_p = pd.DataFrame(X_train_arr, columns=feature_names).reset_index(drop=True)
    X_test_p  = pd.DataFrame(X_test_arr,  columns=feature_names).reset_index(drop=True)
    y_train   = y_train.reset_index(drop=True)
    y_test    = y_test.reset_index(drop=True)

    logger.info(f"[5/5] Features finales: {len(feature_names)}")
    logger.info(f"      X_train_p: {X_train_p.shape} | X_test_p: {X_test_p.shape}")

    # ── Guardar pipeline ──────────────────────────────────────────────────────
    if pipeline_path:
        os.makedirs(os.path.dirname(pipeline_path), exist_ok=True)
        joblib.dump(pipe, pipeline_path)
        logger.info(f"  Pipeline guardado en {pipeline_path}")

    # ── Artefactos ────────────────────────────────────────────────────────────
    artifacts = {
        'pipeline':            pipe,
        'feature_names':       list(feature_names),
        'split_config': {
            'test_size':       test_size,
            'random_state':    random_state,
            'stratify':        True,
        },
        'preprocessing_config': {
            'imputation_strategy':  imputation_strategy,
            'scaling_strategy':     scaling_strategy,
            'categorical_encoding': categorical_encoding,
            'categorical_features': list(categorical_features or DEFAULT_CATEGORICAL_FEATURES),
            'numerical_features':   list(numerical_features   or DEFAULT_NUMERICAL_FEATURES),
        },
        'class_balance_train': dist_train,
        'class_balance_test':  dist_test,
        'n_train':             len(X_train_p),
        'n_test':              len(X_test_p),
        # Listo para pasarle a LightGBM como scale_pos_weight
        'scale_pos_weight':    round((y_train == 0).sum() / max((y_train == 1).sum(), 1), 4),
    }

    if verbose:
        logger.info("=" * 60)
        logger.info("preprocess() completado")
        logger.info(f"  X_train : {X_train_p.shape} | X_test: {X_test_p.shape}")
        logger.info(f"  Features: {len(feature_names)}")
        logger.info(f"  scale_pos_weight: {artifacts['scale_pos_weight']:.2f}")
        logger.info("=" * 60)

    return X_train_p, X_test_p, y_train, y_test, artifacts


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    matrix = pd.read_parquet('data/processed/feature_matrix.parquet')

    X_train, X_test, y_train, y_test, artifacts = preprocess(matrix)

    print("\nPreprocessing completado:")
    print(f"  X_train : {X_train.shape}")
    print(f"  X_test  : {X_test.shape}")
    print(f"  Features: {len(artifacts['feature_names'])}")
    print(f"  Balance train : {artifacts['class_balance_train']}")
    print(f"  Balance test  : {artifacts['class_balance_test']}")
    print(f"  scale_pos_weight: {artifacts['scale_pos_weight']}")
