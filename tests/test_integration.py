"""
tests/test_integration.py
==========================
Tests de integración end-to-end: build_feature_matrix → train.

Foco: verificar que el pipeline completo (feature engineering → split → entrenamiento)
funciona con datos sintéticos que replican la estructura del dataset de Instacart.

Sin DB, sin archivos reales, sin servidor de MLflow.
MLflow usa tracking local en tmp_path para no dejar estado global.
"""

import os

import numpy as np
import pandas as pd
import pytest

from src.features.feature_engineering import FEATURE_MATRIX_COLUMNS, build_feature_matrix
from src.models.train import eval_metrics, split_by_users, train


# ── Fixture: datos sintéticos de Instacart ────────────────────────────────────


def _build_instacart_like_data(
    n_users: int = 40,
    n_products: int = 20,
    n_prior_orders: int = 4,
    products_per_prior_order: int = 4,
    products_per_train_order: int = 2,
    seed: int = 42,
) -> dict:
    """
    Genera fact_order_products y dim_product sintéticos.

    Estructura:
      - Cada usuario tiene n_prior_orders en eval_set='prior' y 1 en 'train'.
      - El order_number del train es siempre > el máximo order_number de prior
        (garantiza que _check_leakage no falle).
      - Los productos del train son un subconjunto de los prior del mismo usuario,
        asegurando que existan pares con label=1 y label=0 en la feature matrix.
    """
    rng = np.random.default_rng(seed)
    rows = []
    order_key = 1

    for user in range(1, n_users + 1):
        seen_products: set = set()

        # ── Órdenes prior ─────────────────────────────────────────────────────
        for order_num in range(1, n_prior_orders + 1):
            chosen = rng.choice(n_products, size=products_per_prior_order, replace=False) + 1
            seen_products.update(int(p) for p in chosen)
            for cart_pos, prod in enumerate(chosen, 1):
                rows.append({
                    "order_key"             : order_key,
                    "user_key"              : user,
                    "product_key"           : int(prod),
                    "order_number"          : order_num,
                    "order_dow"             : int(rng.integers(0, 7)),
                    "order_hour_of_day"     : int(rng.integers(6, 22)),
                    "days_since_prior_order": float(rng.integers(1, 30)) if order_num > 1 else None,
                    "add_to_cart_order"     : cart_pos,
                    "reordered"             : int(rng.integers(0, 2)),
                    "get_eval"              : "prior",
                })
            order_key += 1

        # ── Orden train (label=1 para estos productos) ────────────────────────
        # Seleccionamos del historial previo para garantizar label=1 y label=0.
        prior_list = list(seen_products)
        n_train = min(products_per_train_order, len(prior_list))
        train_products = rng.choice(prior_list, size=n_train, replace=False)
        for cart_pos, prod in enumerate(train_products, 1):
            rows.append({
                "order_key"             : order_key,
                "user_key"              : user,
                "product_key"           : int(prod),
                "order_number"          : n_prior_orders + 1,   # > max prior → sin leakage
                "order_dow"             : int(rng.integers(0, 7)),
                "order_hour_of_day"     : int(rng.integers(6, 22)),
                "days_since_prior_order": float(rng.integers(1, 30)),
                "add_to_cart_order"     : cart_pos,
                "reordered"             : 1,
                "get_eval"              : "train",
            })
        order_key += 1

    fact = pd.DataFrame(rows)
    fact["order_key"]          = fact["order_key"].astype("int32")
    fact["user_key"]           = fact["user_key"].astype("int32")
    fact["product_key"]        = fact["product_key"].astype("int32")
    fact["order_number"]       = fact["order_number"].astype("int16")
    fact["add_to_cart_order"]  = fact["add_to_cart_order"].astype("int16")
    fact["reordered"]          = fact["reordered"].astype("int8")

    dim_product = pd.DataFrame({
        "product_key"    : list(range(1, n_products + 1)),
        "product_name"   : [f"product_{i}" for i in range(1, n_products + 1)],
        "aisle_name"     : [f"aisle_{i % 5}" for i in range(1, n_products + 1)],
        "department_name": [f"dept_{i % 3}" for i in range(1, n_products + 1)],
    })
    dim_product["product_key"] = dim_product["product_key"].astype("int32")

    return {"fact_order_products": fact, "dim_product": dim_product}


@pytest.fixture(scope="module")
def synthetic_data() -> dict:
    return _build_instacart_like_data()


@pytest.fixture(scope="module")
def feature_matrix(synthetic_data) -> pd.DataFrame:
    """Feature matrix construida desde datos sintéticos (sin parquet real)."""
    return build_feature_matrix(
        synthetic_data,
        min_user_orders=2,      # threshold bajo para datos sintéticos
        min_product_orders=1,
        output_path=None,       # no escribe a disco
    )


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_pipeline_output_schema(feature_matrix):
    """El feature matrix tiene exactamente las 26 columnas del contrato."""
    for col in FEATURE_MATRIX_COLUMNS:
        assert col in feature_matrix.columns, f"Columna faltante: '{col}'"
    assert len(feature_matrix.columns) == len(FEATURE_MATRIX_COLUMNS)


def test_no_leakage_end_to_end(feature_matrix):
    """
    Verifica ausencia de leakage user-level entre train y test en el pipeline completo.

    build_feature_matrix ya corre _check_leakage internamente (orden-level).
    Este test verifica además que split_by_users produce conjuntos de usuarios disjuntos.
    """
    train_df, val_df, test_df = split_by_users(feature_matrix, random_state=42)

    train_users = set(train_df["user_key"])
    val_users   = set(val_df["user_key"])
    test_users  = set(test_df["user_key"])

    assert train_users.isdisjoint(test_users), \
        "Leakage: hay usuarios en train y test al mismo tiempo"
    assert train_users.isdisjoint(val_users), \
        "Leakage: hay usuarios en train y val al mismo tiempo"
    assert val_users.isdisjoint(test_users), \
        "Leakage: hay usuarios en val y test al mismo tiempo"


def test_full_pipeline_runs(feature_matrix, tmp_path, monkeypatch):
    """
    El pipeline build_feature_matrix → train completa sin errores
    y devuelve un dict con las claves esperadas.

    Usa run_optuna_flag=False para evitar el costo de Optuna en CI.
    MLflow se redirige a tmp_path para no dejar estado global.
    """
    monkeypatch.setenv("MLFLOW_TRACKING_URI", str(tmp_path / "mlruns"))
    models_dir = tmp_path / "models"

    result = train(
        matrix          = feature_matrix,
        models_dir      = models_dir,
        run_optuna_flag = False,
        random_state    = 42,
    )

    expected_keys = {"model", "cluster_models", "feature_cols", "metrics", "best_params"}
    assert expected_keys.issubset(result.keys()), (
        f"Claves faltantes en el resultado de train(): "
        f"{expected_keys - set(result.keys())}"
    )
    # Las métricas de evaluación deben estar en [0, 1]
    for metric_name, metric_val in result["metrics"].items():
        assert 0.0 <= metric_val <= 1.0, (
            f"Métrica '{metric_name}' fuera de rango [0, 1]: {metric_val}"
        )
