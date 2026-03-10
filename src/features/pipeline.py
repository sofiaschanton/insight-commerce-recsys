"""
pipeline.py
============
Orquestador del feature pipeline para el sistema Next Basket.

Responsabilidad: correr feature_engineering + preprocessing en secuencia
desde un solo comando.

Pasos:
    1. Cargar datos desde NeonDB / Postgress local
    2. feature_engineering.build_feature_matrix()
           → data/processed/feature_matrix.parquet   (585,553 pares, 26 cols)
    3. preprocessing.preprocess()
           → X_train (468,442 × 23)
           → X_test  (117,111 × 23)
           → models/preprocessing_pipeline.pkl

Este módulo NO calcula features directamente, NO entrena modelos.
El split train/test ocurre en preprocessing.preprocess() — no acá.

Uso:
    python pipeline.py
"""

import logging
import time

from src.data.data_loader import load_data_from_neon as load_data
from src.features.feature_engineering import build_feature_matrix
from src.features.preprocessing import preprocess  

logger = logging.getLogger('pipeline')


def run():
    """
    Corre el feature pipeline completo.

    Returns
    -------
    X_train, X_test : pd.DataFrame
        Features preprocesadas listas para train.py.
    y_train, y_test : pd.Series
        Target alineado.
    artifacts : dict
        Pipeline, feature_names, split_config, scale_pos_weight, etc.
    """
    start = time.time()
    logger.info("=" * 60)
    logger.info("Iniciando feature pipeline")
    logger.info("=" * 60)

    # ── 1. Cargar datos ───────────────────────────────────────────────────────
    logger.info("Paso 1 — Cargando datos...")
    t    = time.time()
    data = load_data()
    logger.info(f"  Cargado en {time.time() - t:.1f}s")

    # ── 2. Feature engineering ────────────────────────────────────────────────
    logger.info("Paso 2 — Construyendo feature matrix...")
    t      = time.time()
    matrix = build_feature_matrix(
        data,
        output_path='data/processed/feature_matrix.parquet',
    )
    logger.info(f"  Feature matrix en {time.time() - t:.1f}s — {matrix.shape}")

    # ── 3. Preprocessing con split ────────────────────────────────────────────
    # split train/test ocurre DENTRO de preprocess() antes del fit del pipeline
    # para evitar leakage — el scaler, imputer y capper aprenden solo de train.
    logger.info("Paso 3 — Preprocessing (split + fit en train + transform)...")
    t = time.time()
    X_train, X_test, y_train, y_test, artifacts = preprocess(
        matrix,
        imputation_strategy='median',
        scaling_strategy='standard',
        categorical_encoding='passthrough',   # 'onehot' para modelos no-LightGBM
        target_col='label',
        test_size=0.2,
        random_state=42,
        pipeline_path='models/preprocessing_pipeline.pkl',
    )
    logger.info(f"  Preprocessing en {time.time() - t:.1f}s")

    # ── Resumen ───────────────────────────────────────────────────────────────
    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info(f"Pipeline completo en {elapsed:.1f}s")
    logger.info(f"  feature_matrix.parquet → {matrix.shape[0]:,} filas x {matrix.shape[1]} cols")
    logger.info(f"  X_train                → {X_train.shape}")
    logger.info(f"  X_test                 → {X_test.shape}")
    logger.info(f"  Features               → {len(artifacts['feature_names'])}")
    logger.info(f"  scale_pos_weight       → {artifacts['scale_pos_weight']:.2f}")
    logger.info("=" * 60)

    return X_train, X_test, y_train, y_test, artifacts


if __name__ == '__main__':
    run()
