"""
tests/test_data_loader.py
==========================
Tests de schema y calidad sobre data/processed/feature_matrix.parquet.

Estos tests validan:
  - Que estén presentes las 26 columnas del contrato
  - Que no haya columnas extra (regresión de schema)
  - Que las columnas no-nulables no tengan NaN
  - Que el label sea binario (0 o 1)
  - Que no haya pares (user_key, product_key) duplicados
  - Que columnas numéricas clave sean de tipo float/int correcto
  - Que los valores mínimos tengan sentido de negocio

Los tests saltan automáticamente si el parquet aún no fue generado.
Para generarlo: python -m src.features.feature_engineering
"""

import os
import pytest
import pandas as pd

PARQUET_PATH = "data/processed/feature_matrix.parquet"

# Contrato de columnas — debe coincidir con COLUMN_ORDER en feature_engineering.py
EXPECTED_COLUMNS = [
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

# Columnas que intencionalmente pueden tener NaN (LightGBM las maneja de forma nativa)
NULLABLE_COLUMNS = {
    "p_department_reorder_rate",
    "p_aisle_reorder_rate",
    "up_avg_days_between_orders",
    "up_delta_days",
}

NON_NULLABLE_COLUMNS = [c for c in EXPECTED_COLUMNS if c not in NULLABLE_COLUMNS]


@pytest.fixture(scope="module")
def feature_matrix() -> pd.DataFrame:
    if not os.path.exists(PARQUET_PATH):
        pytest.skip(
            f"Parquet no encontrado en '{PARQUET_PATH}'. "
            "Ejecutar feature engineering primero: python -m src.features.feature_engineering"
        )
    return pd.read_parquet(PARQUET_PATH)


# ─── Schema ───────────────────────────────────────────────────────────────────

class TestParquetSchema:

    def test_all_expected_columns_present(self, feature_matrix):
        missing = [c for c in EXPECTED_COLUMNS if c not in feature_matrix.columns]
        assert not missing, f"Columnas faltantes en el parquet: {missing}"

    def test_no_unexpected_extra_columns(self, feature_matrix):
        extra = [c for c in feature_matrix.columns if c not in EXPECTED_COLUMNS]
        assert not extra, f"Columnas inesperadas en el parquet: {extra}"

    def test_column_count_matches_contract(self, feature_matrix):
        assert len(feature_matrix.columns) == len(EXPECTED_COLUMNS), (
            f"Se esperaban {len(EXPECTED_COLUMNS)} columnas, "
            f"se encontraron {len(feature_matrix.columns)}"
        )


# ─── Nulos ────────────────────────────────────────────────────────────────────

class TestNullValues:

    def test_non_nullable_columns_have_no_nulls(self, feature_matrix):
        for col in NON_NULLABLE_COLUMNS:
            if col not in feature_matrix.columns:
                continue
            nulls = int(feature_matrix[col].isna().sum())
            assert nulls == 0, (
                f"Columna no-nulable '{col}' tiene {nulls} NaN inesperados"
            )

    def test_nullable_columns_are_not_all_null(self, feature_matrix):
        """Las columnas con NaN intencionales no deben ser 100% nulas."""
        for col in NULLABLE_COLUMNS:
            if col not in feature_matrix.columns:
                continue
            non_null = feature_matrix[col].notna().sum()
            assert non_null > 0, f"'{col}' está completamente vacía — revisar pipeline"


# ─── Calidad del label ────────────────────────────────────────────────────────

class TestLabel:

    def test_label_is_binary(self, feature_matrix):
        unique_vals = set(feature_matrix["label"].unique())
        assert unique_vals.issubset({0, 1}), (
            f"label tiene valores no binarios: {unique_vals - {0, 1}}"
        )

    def test_label_has_both_classes(self, feature_matrix):
        unique_vals = set(feature_matrix["label"].unique())
        assert 0 in unique_vals, "No hay ejemplos negativos (label=0)"
        assert 1 in unique_vals, "No hay ejemplos positivos (label=1)"

    def test_label_is_integer_dtype(self, feature_matrix):
        assert pd.api.types.is_integer_dtype(feature_matrix["label"]), (
            f"label debe ser entero, dtype actual: {feature_matrix['label'].dtype}"
        )

    def test_class_imbalance_within_expected_range(self, feature_matrix):
        """El ratio negativo:positivo debe estar entre 3:1 y 50:1 (rango razonable para Instacart)."""
        n_pos = (feature_matrix["label"] == 1).sum()
        n_neg = (feature_matrix["label"] == 0).sum()
        ratio = n_neg / max(n_pos, 1)
        assert 3 <= ratio <= 50, (
            f"Ratio negativo:positivo fuera de rango esperado: {ratio:.1f}:1 "
            f"(n_pos={n_pos:,}, n_neg={n_neg:,})"
        )


# ─── Integridad de IDs y pares ────────────────────────────────────────────────

class TestIdentifiers:

    def test_no_duplicate_user_product_pairs(self, feature_matrix):
        dupes = feature_matrix.duplicated(subset=["user_key", "product_key"]).sum()
        assert dupes == 0, f"Se encontraron {dupes:,} pares (user_key, product_key) duplicados"

    def test_user_key_is_positive(self, feature_matrix):
        assert (feature_matrix["user_key"] > 0).all(), "user_key debe ser > 0"

    def test_product_key_is_positive(self, feature_matrix):
        assert (feature_matrix["product_key"] > 0).all(), "product_key debe ser > 0"


# ─── Tipos y rangos ───────────────────────────────────────────────────────────

class TestDtypesAndRanges:

    def test_float_features_are_float_dtype(self, feature_matrix):
        float_cols = [
            "user_avg_basket_size", "user_reorder_ratio",
            "product_reorder_rate", "up_reorder_rate",
        ]
        for col in float_cols:
            if col not in feature_matrix.columns:
                continue
            assert pd.api.types.is_float_dtype(feature_matrix[col]), (
                f"'{col}' debería ser float, dtype actual: {feature_matrix[col].dtype}"
            )

    def test_count_features_are_non_negative(self, feature_matrix):
        count_cols = [
            "user_total_orders", "product_total_purchases",
            "up_times_purchased", "user_distinct_products",
        ]
        for col in count_cols:
            if col not in feature_matrix.columns:
                continue
            assert (feature_matrix[col] >= 0).all(), f"'{col}' tiene valores negativos"

    def test_reorder_rates_between_zero_and_one(self, feature_matrix):
        rate_cols = ["user_reorder_ratio", "product_reorder_rate", "up_reorder_rate"]
        for col in rate_cols:
            if col not in feature_matrix.columns:
                continue
            vals = feature_matrix[col].dropna()
            assert (vals >= 0).all() and (vals <= 1).all(), (
                f"'{col}' tiene valores fuera del rango [0, 1]"
            )

    def test_up_reorder_rate_derived_correctly(self, feature_matrix):
        """up_reorder_rate = up_times_purchased / user_total_orders — nunca > 1."""
        if "up_reorder_rate" not in feature_matrix.columns:
            pytest.skip("up_reorder_rate no está en el parquet")
        assert (feature_matrix["up_reorder_rate"] <= 1.0 + 1e-5).all(), (
            "up_reorder_rate > 1 implica que up_times_purchased > user_total_orders"
        )
