"""
src/pipeline.py
===============
Orquestador del pipeline completo: carga de datos → feature engineering → entrenamiento.

Encadena los tres pasos en memoria (sin escribir feature_matrix.parquet a disco)
bajo un único run de MLflow que registra parámetros, métricas y artefactos.

MLFLOW_TRACKING_URI se lee desde variable de entorno.
Default: http://127.0.0.1:5000

Uso:
    python -m src.pipeline
    python -m src.pipeline --no-optuna
    python -m src.pipeline --n-users 5000 --trials 20
"""

import argparse
import io
import logging
import os

import boto3
import mlflow

from src.data.data_loader import load_data_from_aws
from src.data.validate_data import validate as validate_feature_matrix
from src.features.feature_engineering import build_feature_matrix
from src.models.train import MODELS_DIR, N_OPTUNA_TRIALS, train

S3_BUCKET = os.getenv("S3_BUCKET", "insight-commerce-artifacts")
USE_S3 = os.getenv("USE_S3", "false").lower() == "true"

# ── Logging ───────────────────────────────────────────────────────────────────
os.makedirs("reports/logs", exist_ok=True)

logger = logging.getLogger("pipeline")
logger.setLevel(logging.DEBUG)
logger.propagate = False

if not logger.handlers:
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler("reports/logs/pipeline.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)


def run_pipeline(
    n_users: int | None = None,
    n_optuna_trials: int = N_OPTUNA_TRIALS,
    run_optuna_flag: bool = True,
    random_state: int = 42,
) -> dict:
    """
    Ejecuta el pipeline completo load → features → train en un único run de MLflow.

    Parameters
    ----------
    n_users : int | None
        Cantidad de usuarios a samplear de la base de datos. None = todos.
    n_optuna_trials : int
        Número de trials de Optuna para búsqueda de hiperparámetros.
    run_optuna_flag : bool
        Si False, usa hiperparámetros por defecto sin Optuna.
    random_state : int
        Semilla global para reproducibilidad.

    Returns
    -------
    dict
        Artefactos del entrenamiento: model, cluster_models, feature_cols, metrics, best_params.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)

    logger.info("=" * 60)
    logger.info("Iniciando pipeline completo")
    logger.info(f"  MLFLOW_TRACKING_URI : {tracking_uri}")
    logger.info(f"  n_users             : {n_users}")
    logger.info(f"  n_optuna_trials     : {n_optuna_trials}")
    logger.info(f"  run_optuna          : {run_optuna_flag}")
    logger.info(f"  random_state        : {random_state}")
    logger.info("=" * 60)

    with mlflow.start_run(run_name="full_pipeline"):

        # ── Parámetros globales del pipeline ──────────────────────────────────
        mlflow.log_params({
            "n_users"         : n_users if n_users is not None else "all",
            "n_optuna_trials" : n_optuna_trials,
            "run_optuna"      : run_optuna_flag,
            "random_state"    : random_state,
        })

        # ── PASO 1: Carga de datos ────────────────────────────────────────────
        logger.info("─" * 60)
        logger.info("PASO 1 — Carga de datos desde AWS RDS PostgreSQL")
        logger.info("─" * 60)
        data = load_data_from_aws(n_users=n_users, random_state=random_state)

        n_orders = len(data.get("fact_order_products", []))
        mlflow.log_param("n_fact_rows", n_orders)
        logger.info(f"Carga completa: {n_orders:,} filas en fact_order_products")

        # ── PASO 2: Feature engineering ───────────────────────────────────────
        logger.info("─" * 60)
        logger.info("PASO 2 — Feature engineering (en memoria, sin escribir a disco)")
        logger.info("─" * 60)
        # output_path=None → no escribe parquet, devuelve el DataFrame en memoria
        matrix = build_feature_matrix(data, output_path=None)

        # ── PASO 2b: Validación del feature matrix ────────────────────────────
        logger.info("─" * 60)
        logger.info("PASO 2b — Validación de calidad del feature matrix")
        logger.info("─" * 60)
        val_report = validate_feature_matrix(matrix, output_dir="reports/data")
        if not val_report.get("all_passed", False):
            failed_checks = [
                k for k, v in val_report.items()
                if k != "all_passed" and isinstance(v, dict) and not v.get("passed", True)
            ]
            raise ValueError(
                f"Validación del feature matrix fallida. Checks fallidos: {failed_checks}. "
                "Ver reports/data/validation_report.json para detalles."
            )
        logger.info("Validación del feature matrix: OK")

        mlflow.log_params({
            "n_pairs"      : len(matrix),
            "n_users_feat" : matrix["user_key"].nunique(),
            "n_products"   : matrix["product_key"].nunique(),
            "label_ratio"  : round(
                (matrix["label"] == 0).sum() / max((matrix["label"] == 1).sum(), 1), 2
            ),
        })
        logger.info(
            f"Feature matrix: {len(matrix):,} pares | "
            f"{matrix['user_key'].nunique():,} usuarios | "
            f"{matrix['product_key'].nunique():,} productos"
        )

        # ── PASO 3: Entrenamiento ─────────────────────────────────────────────
        logger.info("─" * 60)
        logger.info("PASO 3 — Entrenamiento del modelo LightGBM")
        logger.info("─" * 60)
        result = train(
            matrix          = matrix,       # pasa el DataFrame en memoria
            models_dir      = MODELS_DIR,
            n_optuna_trials = n_optuna_trials,
            run_optuna_flag = run_optuna_flag,
            random_state    = random_state,
        )

        # ── Métricas y artefactos del entrenamiento ───────────────────────────
        mlflow.log_metrics(result["metrics"])
        mlflow.log_artifact(str(MODELS_DIR / "model.pkl"))
        mlflow.log_artifact(str(MODELS_DIR / "cluster_models.pkl"))
        mlflow.log_artifact(str(MODELS_DIR / "model_log.json"))

        # ── Upload feature matrix a S3 como referencia para drift monitoring ──
        if USE_S3:
            _buf = io.BytesIO()
            matrix.to_parquet(_buf, index=False)
            _buf.seek(0)
            boto3.client("s3").put_object(
                Bucket=S3_BUCKET,
                Key="feature_matrix_reference.parquet",
                Body=_buf.read(),
            )
            logger.info(f"feature_matrix_reference.parquet subido a s3://{S3_BUCKET}/")

        logger.info("=" * 60)
        logger.info("Pipeline completado exitosamente")
        logger.info(f"  Precision : {result['metrics'].get('precision', 'N/A'):.4f}")
        logger.info(f"  Recall    : {result['metrics'].get('recall', 'N/A'):.4f}")
        logger.info(f"  F1        : {result['metrics'].get('f1', 'N/A'):.4f}")
        logger.info(f"  AUC-ROC   : {result['metrics'].get('auc', 'N/A'):.4f}")
        logger.info("=" * 60)

    return result


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline completo: load → features → train")
    parser.add_argument(
        "--n-users", type=int, default=None,
        help="Usuarios a samplear de la DB. Default: todos."
    )
    parser.add_argument(
        "--no-optuna", action="store_true",
        help="Omitir Optuna y usar hiperparámetros por defecto."
    )
    parser.add_argument(
        "--trials", type=int, default=N_OPTUNA_TRIALS,
        help=f"Número de trials de Optuna. Default: {N_OPTUNA_TRIALS}."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed. Default: 42."
    )
    args = parser.parse_args()

    run_pipeline(
        n_users         = args.n_users,
        n_optuna_trials = args.trials,
        run_optuna_flag = not args.no_optuna,
        random_state    = args.seed,
    )
