"""
tests/test_train.py
====================
Tests unitarios para src/models/train.py.

Foco: split_by_users y eval_metrics — las dos funciones puras que no
requieren archivos ni conexiones externas.

Todos los tests usan datos sintéticos generados con numpy/pandas.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.train import eval_metrics, split_by_users


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def synthetic_matrix():
    """DataFrame con 100 usuarios distintos, 2 pares por usuario."""
    rng = np.random.default_rng(42)
    n_users = 100
    rows_per_user = 2
    user_keys = np.repeat(np.arange(n_users), rows_per_user)
    return pd.DataFrame(
        {
            "user_key": user_keys,
            "product_key": rng.integers(0, 500, size=n_users * rows_per_user),
            "label": rng.integers(0, 2, size=n_users * rows_per_user),
        }
    )


@pytest.fixture
def binary_predictions():
    """y_true balanceado (100 neg + 100 pos), y_pred aleatorio, y_proba aleatorio."""
    rng = np.random.default_rng(0)
    n = 200
    y_true = np.array([0] * 100 + [1] * 100)
    y_pred = rng.integers(0, 2, size=n)
    y_proba = rng.uniform(0.0, 1.0, size=n)
    return y_true, y_pred, y_proba


# ── Tests: split_by_users ─────────────────────────────────────────────────────


def test_split_by_users_proportions(synthetic_matrix):
    """Split por usuarios produce aproximadamente 70/15/15 en cantidad de usuarios."""
    train_df, val_df, test_df = split_by_users(synthetic_matrix, random_state=42)

    total_users = synthetic_matrix["user_key"].nunique()
    train_users = train_df["user_key"].nunique()
    val_users = val_df["user_key"].nunique()
    test_users = test_df["user_key"].nunique()

    assert abs(train_users / total_users - 0.70) < 0.05
    assert abs(val_users / total_users - 0.15) < 0.05
    assert abs(test_users / total_users - 0.15) < 0.05


def test_split_by_users_no_user_overlap(synthetic_matrix):
    """Ningún usuario aparece en más de un conjunto."""
    train_df, val_df, test_df = split_by_users(synthetic_matrix, random_state=42)

    train_users = set(train_df["user_key"])
    val_users = set(val_df["user_key"])
    test_users = set(test_df["user_key"])

    assert train_users.isdisjoint(val_users)
    assert train_users.isdisjoint(test_users)
    assert val_users.isdisjoint(test_users)


# ── Tests: eval_metrics ───────────────────────────────────────────────────────


def test_eval_metrics_returns_expected_keys(binary_predictions):
    """El dict devuelto tiene exactamente las claves precision, recall, f1, auc."""
    y_true, y_pred, y_proba = binary_predictions
    result = eval_metrics(y_true, y_pred, y_proba)
    assert set(result.keys()) == {"precision", "recall", "f1", "auc"}


def test_eval_metrics_range(binary_predictions):
    """Todas las métricas están en el intervalo [0, 1]."""
    y_true, y_pred, y_proba = binary_predictions
    result = eval_metrics(y_true, y_pred, y_proba)
    for key, value in result.items():
        assert 0.0 <= value <= 1.0, f"Métrica '{key}' fuera de rango [0, 1]: {value}"
