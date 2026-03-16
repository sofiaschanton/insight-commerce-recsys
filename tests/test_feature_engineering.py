"""
tests/test_feature_engineering.py
===================================
Tests unitarios para src/features/feature_engineering.py

Foco principal:
  - _check_leakage: garantiza que prior no contenga datos de train
  - _get_prior / _get_train: separación correcta de eval_set
  - get_label: el label se construye SOLO desde train
  - get_user_features: cálculo correcto con filtro min_user_orders
  - NaN intencionales: up_avg_days_between_orders es NaN cuando times_purchased==1
"""

import pandas as pd
import numpy as np
import pytest

from src.features.feature_engineering import (
    _check_leakage,
    _get_prior,
    _get_train,
    get_label,
    get_user_features,
    get_user_product_features,
)


# ─── Fixtures comunes ─────────────────────────────────────────────────────────

def _make_fact(rows: list[dict]) -> pd.DataFrame:
    """Construye un fact_order_products mínimo para tests."""
    return pd.DataFrame(rows)


def _prior_rows(user_key: int, n_orders: int, products: list[int]) -> list[dict]:
    """Genera filas de prior para un usuario con n órdenes y productos dados."""
    rows = []
    for order_num in range(1, n_orders + 1):
        for prod in products:
            rows.append({
                "user_key": user_key,
                "product_key": prod,
                "order_key": order_num * 100,  # mismo order_key para todos los productos de la orden
                "order_number": order_num,
                "days_since_prior_order": 7.0 if order_num > 1 else float("nan"),
                "reordered": 1 if order_num > 1 else 0,
                "add_to_cart_order": 1,
                "get_eval": "prior",
            })
    return rows


# ─── _check_leakage ───────────────────────────────────────────────────────────

class TestCheckLeakage:

    def test_no_leakage_passes_silently(self):
        """prior_max < train_order → sin excepción."""
        prior = pd.DataFrame({"user_key": [1, 1, 2], "order_number": [1, 2, 3]})
        train = pd.DataFrame({"user_key": [1, 2],    "order_number": [3, 4]})
        _check_leakage(prior, train)  # no debe lanzar

    def test_leakage_equal_order_numbers_raises(self):
        """prior_max == train_order → data leakage detectado → ValueError."""
        prior = pd.DataFrame({"user_key": [1, 1], "order_number": [1, 3]})
        train = pd.DataFrame({"user_key": [1],    "order_number": [3]})
        with pytest.raises(ValueError, match="leakage"):
            _check_leakage(prior, train)

    def test_leakage_prior_greater_than_train_raises(self):
        """prior_max > train_order es leakage grave."""
        prior = pd.DataFrame({"user_key": [1, 1], "order_number": [4, 5]})
        train = pd.DataFrame({"user_key": [1],    "order_number": [3]})
        with pytest.raises(ValueError, match="leakage"):
            _check_leakage(prior, train)

    def test_multiple_users_partial_violation_raises(self):
        """Solo un usuario viola la regla → igual debe lanzar."""
        prior = pd.DataFrame({
            "user_key":     [1, 1, 2, 2],
            "order_number": [1, 2, 1, 5],   # user=2 viola (max=5, train=4)
        })
        train = pd.DataFrame({
            "user_key":     [1, 2],
            "order_number": [3, 4],
        })
        with pytest.raises(ValueError, match="leakage"):
            _check_leakage(prior, train)

    def test_all_users_strictly_before_train(self):
        """Caso canónico: prior estrictamente antes de train para todos los usuarios."""
        prior = pd.DataFrame({
            "user_key":     [1, 1, 1, 2, 2],
            "order_number": [1, 2, 3, 1, 2],
        })
        train = pd.DataFrame({
            "user_key":     [1, 2],
            "order_number": [4, 3],
        })
        _check_leakage(prior, train)  # no debe lanzar


# ─── _get_prior / _get_train ──────────────────────────────────────────────────

class TestGetPriorTrain:

    @pytest.fixture
    def mixed_fact(self):
        return _make_fact([
            {"user_key": 1, "order_number": 1, "get_eval": "prior"},
            {"user_key": 1, "order_number": 2, "get_eval": "prior"},
            {"user_key": 1, "order_number": 3, "get_eval": "train"},
            {"user_key": 2, "order_number": 1, "get_eval": "prior"},
            {"user_key": 2, "order_number": 2, "get_eval": "train"},
        ])

    def test_get_prior_returns_only_prior(self, mixed_fact):
        result = _get_prior(mixed_fact)
        assert set(result["get_eval"].unique()) == {"prior"}
        assert len(result) == 3

    def test_get_train_returns_only_train(self, mixed_fact):
        result = _get_train(mixed_fact)
        assert set(result["get_eval"].unique()) == {"train"}
        assert len(result) == 2

    def test_prior_and_train_are_disjoint(self, mixed_fact):
        """Las filas de prior y train no deben solaparse."""
        prior_orders = set(zip(_get_prior(mixed_fact)["user_key"], _get_prior(mixed_fact)["order_number"]))
        train_orders = set(zip(_get_train(mixed_fact)["user_key"], _get_train(mixed_fact)["order_number"]))
        assert prior_orders.isdisjoint(train_orders)

    def test_get_prior_does_not_include_test(self):
        fact = _make_fact([
            {"user_key": 1, "get_eval": "prior"},
            {"user_key": 1, "get_eval": "train"},
            {"user_key": 1, "get_eval": "test"},
        ])
        result = _get_prior(fact)
        assert "test" not in result["get_eval"].values
        assert "train" not in result["get_eval"].values


# ─── get_label ────────────────────────────────────────────────────────────────

class TestGetLabel:

    def test_all_labels_are_one(self):
        """Todos los pares que vienen de train deben tener label=1."""
        train = pd.DataFrame({"user_key": [1, 2, 3], "product_key": [101, 202, 303]})
        result = get_label(train)
        assert (result["label"] == 1).all()

    def test_deduplicates_user_product_pairs(self):
        """Pares duplicados en train se consolidan en una sola fila."""
        train = pd.DataFrame({
            "user_key":    [1, 1, 2],
            "product_key": [101, 101, 202],
        })
        result = get_label(train)
        assert len(result) == 2

    def test_no_label_zero_in_output(self):
        """get_label solo genera label=1; el label=0 lo asigna build_feature_matrix."""
        train = pd.DataFrame({"user_key": [1], "product_key": [100]})
        result = get_label(train)
        assert 0 not in result["label"].values

    def test_output_columns(self):
        train = pd.DataFrame({"user_key": [1], "product_key": [100]})
        result = get_label(train)
        assert set(result.columns) == {"user_key", "product_key", "label"}


# ─── get_user_features ────────────────────────────────────────────────────────

class TestGetUserFeatures:

    @pytest.fixture
    def prior_multi(self):
        """user=1 tiene 6 órdenes, user=2 tiene 2 órdenes."""
        rows = _prior_rows(user_key=1, n_orders=6, products=[101, 102])
        rows += _prior_rows(user_key=2, n_orders=2, products=[201])
        return pd.DataFrame(rows)

    def test_min_user_orders_excludes_low_history_users(self, prior_multi):
        result = get_user_features(prior_multi, min_user_orders=5)
        assert 1 in result["user_key"].values
        assert 2 not in result["user_key"].values

    def test_min_user_orders_one_includes_all(self, prior_multi):
        result = get_user_features(prior_multi, min_user_orders=1)
        assert {1, 2}.issubset(set(result["user_key"].values))

    def test_user_total_orders_correct(self, prior_multi):
        result = get_user_features(prior_multi, min_user_orders=1)
        user1 = result.loc[result["user_key"] == 1, "user_total_orders"].iloc[0]
        assert user1 == 6

    def test_user_reorder_ratio_between_zero_and_one(self, prior_multi):
        result = get_user_features(prior_multi, min_user_orders=1)
        assert (result["user_reorder_ratio"] >= 0).all()
        assert (result["user_reorder_ratio"] <= 1).all()

    def test_required_columns_present(self, prior_multi):
        result = get_user_features(prior_multi, min_user_orders=1)
        expected = {
            "user_key", "user_total_orders", "user_avg_basket_size",
            "user_days_since_last_order", "user_reorder_ratio",
            "user_distinct_products", "user_segment_code",
        }
        assert expected.issubset(set(result.columns))


# ─── NaN intencionales en up_avg_days_between_orders ─────────────────────────

class TestIntentionalNaN:

    def test_up_avg_days_is_nan_when_purchased_once(self):
        """Si el usuario compró el producto solo una vez, up_avg_days_between_orders debe ser NaN."""
        prior = pd.DataFrame({
            "user_key":              [1, 1, 1],
            "product_key":           [101, 102, 101],   # 101: 2 veces, 102: 1 vez
            "order_key":             [1, 2, 3],
            "order_number":          [1, 1, 2],
            "days_since_prior_order":[float("nan"), float("nan"), 7.0],
            "reordered":             [0, 0, 1],
            "add_to_cart_order":     [1, 1, 1],
        })
        result = get_user_product_features(prior)

        row_102 = result[result["product_key"] == 102]
        assert row_102["up_avg_days_between_orders"].isna().all(), (
            "up_avg_days_between_orders debe ser NaN cuando times_purchased==1"
        )

    def test_up_delta_days_inherits_nan(self):
        """up_delta_days debe ser NaN donde up_avg_days_between_orders es NaN."""
        prior = pd.DataFrame({
            "user_key":              [1],
            "product_key":           [101],
            "order_key":             [1],
            "order_number":          [1],
            "days_since_prior_order":[float("nan")],
            "reordered":             [0],
            "add_to_cart_order":     [1],
        })
        result = get_user_product_features(prior)
        assert result["up_delta_days"].isna().all()
