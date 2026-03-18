"""
src/models/train.py
====================
RESPONSABILIDAD: entrenar el modelo de recomendacion Next Basket
desde feature_matrix.parquet y serializar los artefactos.

Este script NO calcula features, NO abre conexion a la base de datos
y NO hace preprocessing de imputacion/escalado (eso es responsabilidad
de preprocessing.py a traves del pipeline.pkl).

Flujo:
    data/processed/feature_matrix.parquet
        |
        1. Split por usuarios 70/15/15  (evita leakage entre conjuntos)
        |
        2. K-Means fit en train         (user_cluster, product_cluster)
        |
        3. Optuna -- busqueda de hiperparametros sobre val set
        |
        4. LightGBM optimizado          (fit en train, eval en val)
        |
        5. Guardado MLFlow              (Guardado en MLFlow los resultados)
        |
        5. Evaluacion en test
        |
        6. Serializacion
              models/model.pkl
              models/cluster_models.pkl
              models/model_log.json

Uso:
    python -m src.models.train
    python -m src.models.train --no-optuna     # usa hiperparametros por defecto
    python -m src.models.train --trials 100    # cambia el numero de trials de Optuna

Conexion con el resto del pipeline:
    - Lee:    data/processed/feature_matrix.parquet  (output de pipeline.py)
    - Escribe: models/model.pkl                      (input de recommendation.py)
    - Escribe: models/cluster_models.pkl             (input de recommendation.py)
    - Escribe: models/model_log.json                 (referencia para API y DS-4)
"""

import argparse
import io
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import boto3
import mlflow
import mlflow.lightgbm
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Logging ───────────────────────────────────────────────────────────────────
os.makedirs("reports/logs", exist_ok=True)

logger = logging.getLogger("train")
logger.setLevel(logging.DEBUG)
logger.propagate = False

if not logger.handlers:
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler("reports/logs/train.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)

# ── Constantes ────────────────────────────────────────────────────────────────
FEATURE_MATRIX_PATH = Path("data/processed/feature_matrix.parquet")
MODELS_DIR          = Path("models")

S3_BUCKET = os.getenv("S3_BUCKET", "insight-commerce-artifacts")
USE_S3 = os.getenv("USE_S3", "false").lower() == "true"

ID_COLS    = ["user_key", "product_key"]
LABEL_COL  = "label"

# Features categoricas para LightGBM (passthrough -- no requieren encoding)
LGBM_CAT_FEATURES = [
    "user_segment_code",
    "u_favorite_department",
    "u_favorite_aisle",
    "user_cluster",
    "product_cluster",
]

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

N_CLUSTERS_USER    = 5
N_CLUSTERS_PRODUCT = 5
N_OPTUNA_TRIALS    = 50

F1_THRESHOLD = 0.10   # Umbral mínimo de F1 para aceptar el modelo (Model Gate)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_matrix(path: Path) -> pd.DataFrame:
    """Carga feature_matrix.parquet y valida que exista."""
    if not path.exists():
        raise FileNotFoundError(
            f"feature_matrix.parquet no encontrado en {path}\n"
            "Ejecutar primero: python -m src.features.pipeline"
        )
    matrix = pd.read_parquet(path)
    logger.info(f"Feature matrix cargado: {matrix.shape[0]:,} filas x {matrix.shape[1]} cols")
    return matrix


def split_by_users(
    matrix: pd.DataFrame,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split 70/15/15 por usuarios.

    Separa por user_key para que el mismo usuario no aparezca en mas de
    un conjunto. Evita que las features de usuario (identicas para todos
    los pares del mismo usuario) inflen artificialmente las metricas.
    """
    all_users = matrix["user_key"].unique()
    users_train, users_temp = train_test_split(
        all_users, test_size=0.30, random_state=random_state
    )
    users_val, users_test = train_test_split(
        users_temp, test_size=0.50, random_state=random_state
    )
    train_df = matrix[matrix["user_key"].isin(users_train)].copy()
    val_df   = matrix[matrix["user_key"].isin(users_val)].copy()
    test_df  = matrix[matrix["user_key"].isin(users_test)].copy()

    logger.info(
        f"Split por usuarios -- "
        f"train: {len(users_train):,} usuarios / {len(train_df):,} pares | "
        f"val: {len(users_val):,} / {len(val_df):,} | "
        f"test: {len(users_test):,} / {len(test_df):,}"
    )
    return train_df, val_df, test_df


def _assign_cluster(
    df: pd.DataFrame,
    key_col: str,
    feature_cols: list,
    ref_profiles: pd.DataFrame,
    scaler: StandardScaler,
    kmeans: KMeans,
    col_name: str,
) -> pd.DataFrame:
    """
    Asigna cluster a los pares de df.
    Entidades no vistas en train reciben cluster -1.
    """
    profiles = df.groupby(key_col)[feature_cols].mean()
    known    = profiles.index.isin(ref_profiles.index)
    clusters = pd.Series(-1, index=profiles.index, name=col_name, dtype="int8")
    if known.any():
        scaled           = scaler.transform(profiles[known].fillna(profiles[known].median()))
        clusters[known]  = kmeans.predict(scaled).astype("int8")
    return df.merge(clusters.reset_index(), on=key_col, how="left")


def fit_kmeans(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Fitea K-Means sobre train y transforma los tres conjuntos.

    Devuelve los tres DataFrames con user_cluster y product_cluster
    agregados, y un dict con los modelos para serializacion.
    """
    logger.info("Fiteando K-Means sobre train...")

    # ── Usuarios ──────────────────────────────────────────────────────────────
    user_profiles = train_df.groupby("user_key")[USER_CLUSTER_FEATURES].mean()
    scaler_user   = StandardScaler()
    user_scaled   = scaler_user.fit_transform(user_profiles)
    kmeans_user   = KMeans(n_clusters=N_CLUSTERS_USER, random_state=random_state, n_init=10)
    kmeans_user.fit(user_scaled)

    # ── Productos ─────────────────────────────────────────────────────────────
    product_profiles = train_df.groupby("product_key")[PRODUCT_CLUSTER_FEATURES].mean()
    scaler_product   = StandardScaler()
    product_scaled   = scaler_product.fit_transform(
        product_profiles.fillna(product_profiles.median())
    )
    kmeans_product = KMeans(n_clusters=N_CLUSTERS_PRODUCT, random_state=random_state, n_init=10)
    kmeans_product.fit(product_scaled)

    # ── Asignar clusters a los tres conjuntos ─────────────────────────────────
    for df_ref, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        pass  # se hace abajo con la funcion helper

    train_df = _assign_cluster(train_df, "user_key",    USER_CLUSTER_FEATURES,    user_profiles,    scaler_user,    kmeans_user,    "user_cluster")
    val_df   = _assign_cluster(val_df,   "user_key",    USER_CLUSTER_FEATURES,    user_profiles,    scaler_user,    kmeans_user,    "user_cluster")
    test_df  = _assign_cluster(test_df,  "user_key",    USER_CLUSTER_FEATURES,    user_profiles,    scaler_user,    kmeans_user,    "user_cluster")

    train_df = _assign_cluster(train_df, "product_key", PRODUCT_CLUSTER_FEATURES, product_profiles, scaler_product, kmeans_product, "product_cluster")
    val_df   = _assign_cluster(val_df,   "product_key", PRODUCT_CLUSTER_FEATURES, product_profiles, scaler_product, kmeans_product, "product_cluster")
    test_df  = _assign_cluster(test_df,  "product_key", PRODUCT_CLUSTER_FEATURES, product_profiles, scaler_product, kmeans_product, "product_cluster")

    cluster_models = {
        "kmeans_user"    : kmeans_user,
        "scaler_user"    : scaler_user,
        "kmeans_product" : kmeans_product,
        "scaler_product" : scaler_product,
        "user_profiles"  : user_profiles,
        "product_profiles": product_profiles,
    }
    logger.info(
        f"K-Means OK -- "
        f"user_cluster: {N_CLUSTERS_USER} clusters | "
        f"product_cluster: {N_CLUSTERS_PRODUCT} clusters"
    )
    return train_df, val_df, test_df, cluster_models


def get_Xy(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple:
    """Separa features y label en los tres conjuntos."""
    feature_cols = [c for c in train_df.columns if c not in ID_COLS + [LABEL_COL]]
    X_train, y_train = train_df[feature_cols], train_df[LABEL_COL]
    X_val,   y_val   = val_df[feature_cols],   val_df[LABEL_COL]
    X_test,  y_test  = test_df[feature_cols],  test_df[LABEL_COL]
    logger.info(f"Features: {len(feature_cols)} | {feature_cols}")
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


def eval_metrics(y_true, y_pred, y_proba=None) -> Dict[str, float]:
    """Calcula precision, recall, F1 y opcionalmente AUC-ROC."""
    m = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall"   : float(recall_score(y_true, y_pred, zero_division=0)),
        "f1"       : float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None:
        m["auc"] = float(roc_auc_score(y_true, y_proba))
    return m


# ── Optuna ────────────────────────────────────────────────────────────────────

def run_optuna(
    X_train, y_train,
    X_val,   y_val,
    scale_pos_weight: float,
    lgbm_cat: list,
    n_trials: int,
    random_state: int,
) -> Dict[str, Any]:
    """
    Busqueda bayesiana de hiperparametros con Optuna.
    Optimiza F1 en val set. Devuelve los mejores parametros.
    """
    def objective(trial):
        params = {
            "n_estimators"     : trial.suggest_int("n_estimators", 200, 1000),
            "learning_rate"    : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves"       : trial.suggest_int("num_leaves", 20, 150),
            "max_depth"        : trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample"        : trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha"        : trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda"       : trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight" : scale_pos_weight,
            "random_state"     : random_state,
            "n_jobs"           : -1,
            "verbose"          : -1,
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set            = [(X_val, y_val)],
            categorical_feature = lgbm_cat,
            callbacks           = [
                lgb.early_stopping(30, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )
        return f1_score(y_val, model.predict(X_val), zero_division=0)

    logger.info(f"Optuna -- {n_trials} trials | metrica: F1 en val")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best: Dict[str, Any] = study.best_params
    best.update({
        "scale_pos_weight": scale_pos_weight,
        "random_state"    : random_state,
        "n_jobs"          : -1,
        "verbose"         : -1,
    })
    logger.info(f"Optuna OK -- mejor F1 val: {study.best_value:.4f}")
    return best


# ── Entrenamiento principal ───────────────────────────────────────────────────
def train(
    parquet_path: Path           = FEATURE_MATRIX_PATH,
    models_dir: Path             = MODELS_DIR,
    n_optuna_trials: int         = N_OPTUNA_TRIALS,
    run_optuna_flag: bool        = True,
    random_state: int            = 42,
    matrix: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Flujo completo de entrenamiento.
    Guardado de Resultados en MLFlow

    Parameters
    ----------
    parquet_path : Path
        Ruta al feature_matrix.parquet generado por pipeline.py.
        Solo se usa cuando matrix=None.
    models_dir : Path
        Directorio donde se guardan model.pkl, cluster_models.pkl y model_log.json.
    n_optuna_trials : int
        Numero de trials de Optuna. Default: 50.
    run_optuna_flag : bool
        Si False, usa hiperparametros por defecto sin Optuna.
    random_state : int
        Semilla global. Default: 42.
    matrix : pd.DataFrame | None
        Feature matrix precalculada. Si se pasa, omite la lectura del parquet.
        Permite encadenar con pipeline.py sin escribir a disco.

    Returns
    -------
    dict
        Artefactos: modelo, cluster_models, feature_cols, metrics, best_params.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    start = time.time()
    models_dir.mkdir(parents=True, exist_ok=True)
    
    mlflow.set_experiment("Next_Basket_Recommendation")

    logger.info("=" * 60)
    logger.info("Iniciando train.py")
    logger.info(f"  parquet_path     : {parquet_path}")
    logger.info(f"  n_optuna_trials  : {n_optuna_trials}")
    logger.info(f"  run_optuna       : {run_optuna_flag}")
    logger.info(f"  random_state     : {random_state}")
    logger.info("=" * 60)

    # Con esto podemos crear un RUN en MLFlow para guardar las mejores metricas que se guardaron en el modelo
    # MLFlow es como una bitacora para evaluar cual es el mejor modelo en uno u otro caso
    # El parámetro nested=True permite que este run viva dentro del run del pipeline
    with mlflow.start_run(run_name="lgbm_kmeans_pipeline", nested=True):

        # Registrar parámetros globales del pipeline
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("usa_optuna", run_optuna_flag)
        mlflow.log_param("optuna_trials", n_optuna_trials)
        mlflow.log_param("n_clusters_user", N_CLUSTERS_USER)
        mlflow.log_param("n_clusters_product", N_CLUSTERS_PRODUCT)

        # ── 1. Cargar ─────────────────────────────────────────────────────────────
        if matrix is None:
            matrix = load_matrix(parquet_path)
        else:
            logger.info(
                f"Feature matrix recibida en memoria: {matrix.shape[0]:,} filas x {matrix.shape[1]} cols"
            )

        label_dist       = matrix[LABEL_COL].value_counts()
        scale_pos_weight = float(label_dist[0] / max(label_dist[1], 1))
        logger.info(
            f"Balance -- label=0: {label_dist[0]:,} | label=1: {label_dist[1]:,} | "
            f"scale_pos_weight: {scale_pos_weight:.2f}"
        )

        # ── 2. Split por usuarios ─────────────────────────────────────────────────
        train_df, val_df, test_df = split_by_users(matrix, random_state)

        # ── 3. K-Means (fit solo en train) ────────────────────────────────────────
        train_df, val_df, test_df, cluster_models = fit_kmeans(
            train_df, val_df, test_df, random_state
        )

        # ── 4. Separar X e y ──────────────────────────────────────────────────────
        X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = get_Xy(
            train_df, val_df, test_df
        )
        lgbm_cat = [c for c in LGBM_CAT_FEATURES if c in feature_cols]

        # ── 5. Hiperparametros ────────────────────────────────────────────────────
        if run_optuna_flag:
            best_params = run_optuna(
                X_train, y_train, X_val, y_val,
                scale_pos_weight, lgbm_cat, n_optuna_trials, random_state,
            )
        else:
            logger.info("Optuna omitido -- usando hiperparametros por defecto")
            best_params = {
                "n_estimators"    : 500,
                "learning_rate"   : 0.05,
                "num_leaves"      : 63,
                "scale_pos_weight": scale_pos_weight,
                "random_state"    : random_state,
                "n_jobs"          : -1,
                "verbose"         : -1,
            }
        
        mlflow.log_params(best_params) # Guardamos los logs con las mejores metricas

        # ── 6. Entrenar modelo final ───────────────────────────────────────────────
        logger.info("Entrenando LightGBM con mejores parametros...")
        model = lgb.LGBMClassifier(**best_params)
        model.fit(
            X_train, y_train,
            eval_set            = [(X_val, y_val)],
            categorical_feature = lgbm_cat,
            callbacks           = [
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(100),
            ],
        )
        logger.info(f"  Best iteration: {model.best_iteration_}")

        # ── 7. Evaluar en test ────────────────────────────────────────────────────
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics = eval_metrics(y_test, y_pred, y_proba)

        mlflow.log_metrics(metrics)

        logger.info("-" * 60)
        logger.info("Metricas en test:")
        logger.info(f"  Precision : {metrics['precision']:.4f}")
        logger.info(f"  Recall    : {metrics['recall']:.4f}")
        logger.info(f"  F1        : {metrics['f1']:.4f}")
        logger.info(f"  AUC-ROC   : {metrics['auc']:.4f}")
        logger.info("-" * 60)

        # ── 8. Feature importance ─────────────────────────────────────────────────
        importance = pd.DataFrame({
            "feature"   : feature_cols,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)

        zero_imp = importance[importance["importance"] == 0]["feature"].tolist()
        if zero_imp:
            logger.info(f"  Features con importance=0 ({len(zero_imp)}): {zero_imp}")

        # ── 9. Rollback check ─────────────────────────────────────────────────────
        # Si el F1 nuevo es más de 5% inferior al F1 del modelo anterior,
        # no sobreescribir los artefactos y salir con error claro.
        model_path        = models_dir / "model.pkl"
        cluster_path      = models_dir / "cluster_models.pkl"
        log_path          = models_dir / "model_log.json"

        old_log = None
        if log_path.exists():
            try:
                with open(log_path, encoding="utf-8") as _f:
                    old_log = json.load(_f)
            except (json.JSONDecodeError, KeyError) as _e:
                logger.warning(f"No se pudo leer model_log.json local para rollback check: {_e}. Continuando.")
        elif USE_S3:
            try:
                _s3 = boto3.client("s3")
                _buf = io.BytesIO()
                _s3.download_fileobj(S3_BUCKET, "models/latest/model_log.json", _buf,
                                     ExtraArgs={"ExpectedBucketOwner": os.getenv("AWS_ACCOUNT_ID", "")})
                _buf.seek(0)
                old_log = json.loads(_buf.read().decode("utf-8"))
            except Exception as _e:
                logger.warning(f"No se pudo leer model_log.json desde S3 para rollback check: {_e}. Continuando.")

        if old_log is not None:
            try:
                old_f1 = float(old_log.get("metrics_test", {}).get("f1", 0.0))
                new_f1 = metrics["f1"]
                if old_f1 > 0 and new_f1 < old_f1 * 0.95:
                    logger.warning("=" * 60)
                    logger.warning("ROLLBACK ACTIVADO — los artefactos NO serán sobreescritos.")
                    logger.warning(f"  F1 anterior : {old_f1:.4f}")
                    logger.warning(f"  F1 nuevo    : {new_f1:.4f}  (degradación > 5%)")
                    logger.warning(f"  Umbral      : {old_f1 * 0.95:.4f}  (F1_anterior x 0.95)")
                    logger.warning("=" * 60)
                    raise RuntimeError(
                        f"Rollback: F1 nuevo ({new_f1:.4f}) es más de 5% inferior al F1 anterior "
                        f"({old_f1:.4f}). Umbral mínimo: {old_f1 * 0.95:.4f}. "
                        "Los artefactos del modelo anterior NO fueron sobreescritos."
                    )
            except (json.JSONDecodeError, KeyError) as _e:
                logger.warning(f"No se pudo evaluar rollback desde model_log.json: {_e}. Continuando.")

        # ── 10. Serializar ────────────────────────────────────────────────────────
        joblib.dump(model,          model_path)
        joblib.dump(cluster_models, cluster_path)

        log = {
            "timestamp"        : datetime.now().isoformat(),
            "model_name"       : "LightGBM optimizado",
            "model_path"       : str(model_path),
            "random_seed"      : random_state,
            "scale_pos_weight" : scale_pos_weight,
            "n_features"       : len(feature_cols),
            "feature_cols"     : feature_cols,
            "cat_features_lgbm": lgbm_cat,
            "split": {
                "method"      : "by_users_70_15_15",
                "n_train_users": int(matrix["user_key"].nunique() * 0.70),
                "n_val_users"  : int(matrix["user_key"].nunique() * 0.15),
                "n_test_users" : int(matrix["user_key"].nunique() * 0.15),
                "n_train_pairs": len(train_df),
                "n_val_pairs"  : len(val_df),
                "n_test_pairs" : len(test_df),
            },
            "metrics_test"     : metrics,
            "best_params"      : {
                k: (float(v) if isinstance(v, (float, np.floating)) else v)
                for k, v in best_params.items()
            },
            "importance_top10" : importance.head(10)[["feature", "importance"]].assign(
                importance=lambda x: x["importance"].astype(int)
            ).to_dict("records"),
            "features_zero_importance": zero_imp,
        }
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())

        # ── 11. Model Gate + Safe Deployment en S3 ───────────────────────────────
        use_s3_this_run = USE_S3
        new_f1 = metrics["f1"]

        if new_f1 < F1_THRESHOLD:
            logger.warning("=" * 60)
            logger.warning("MODEL GATE: el modelo NO supera el umbral de calidad.")
            logger.warning(f"  F1 obtenido : {new_f1:.4f}")
            logger.warning(f"  F1 umbral   : {F1_THRESHOLD:.4f}")
            logger.warning("  La subida a S3 queda CANCELADA para esta ejecución.")
            logger.warning("=" * 60)
            use_s3_this_run = False

        if use_s3_this_run:
            run_id = mlflow.active_run().info.run_id
            _s3 = boto3.client("s3")

            # Rutas versionadas — nunca sobreescriben versiones anteriores
            versioned_model   = f"models/{run_id}/model.pkl"
            versioned_cluster = f"models/{run_id}/cluster_models.pkl"
            versioned_log     = f"models/{run_id}/model_log.json"

            _owner = {"ExpectedBucketOwner": os.getenv("AWS_ACCOUNT_ID", "")}
            _s3.upload_file(str(model_path),   S3_BUCKET, versioned_model,   ExtraArgs=_owner)
            _s3.upload_file(str(cluster_path), S3_BUCKET, versioned_cluster, ExtraArgs=_owner)
            _s3.upload_file(str(log_path),     S3_BUCKET, versioned_log,     ExtraArgs=_owner)
            logger.info(
                f"Artefactos versionados subidos a s3://{S3_BUCKET}/{versioned_model} (y cluster + log)"
            )

            # Puntero 'latest' — producción siempre lee lo último que pasó el gate
            copy_source = {"Bucket": S3_BUCKET, "Key": versioned_model}
            _s3.copy(copy_source, S3_BUCKET, "models/latest/model.pkl")
            copy_source["Key"] = versioned_cluster
            _s3.copy(copy_source, S3_BUCKET, "models/latest/cluster_models.pkl")
            copy_source["Key"] = versioned_log
            _s3.copy(copy_source, S3_BUCKET, "models/latest/model_log.json")
            logger.info(f"Puntero 'latest' actualizado en s3://{S3_BUCKET}/models/latest/")

        mlflow.lightgbm.log_model(model, artifact_path="lightgbm_model") # Nombre del modelo lightGBM que se uso y en path donde se guarda en MLFlow
        mlflow.log_artifact(local_path=str(cluster_path), artifact_path="cluster_models") 
        mlflow.log_artifact(local_path=str(log_path), artifact_path="logs")

    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info(f"train.py completado en {elapsed:.1f}s")
    logger.info(f"  model.pkl         -> {model_path}")
    logger.info(f"  cluster_models.pkl-> {cluster_path}")
    logger.info(f"  model_log.json    -> {log_path}")
    logger.info("=" * 60)

    return {
        "model"         : model,
        "cluster_models": cluster_models,
        "feature_cols"  : feature_cols,
        "metrics"       : metrics,
        "best_params"   : best_params,
        "importance"    : importance,
    }

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar modelo Next Basket")
    parser.add_argument(
        "--no-optuna", action="store_true",
        help="Omitir Optuna y usar hiperparametros por defecto"
    )
    parser.add_argument(
        "--trials", type=int, default=N_OPTUNA_TRIALS,
        help=f"Numero de trials de Optuna (default: {N_OPTUNA_TRIALS})"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    train(
        parquet_path     = FEATURE_MATRIX_PATH,
        models_dir       = MODELS_DIR,
        n_optuna_trials  = args.trials,
        run_optuna_flag  = not args.no_optuna,
        random_state     = args.seed,
    )
