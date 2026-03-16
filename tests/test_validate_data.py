"""
tests/test_validate_data.py
===========================
Tests unitarios para src/data/validate_data.py.

Usan DataFrames sintéticos en memoria; no leen parquet ni archivos reales.
El reporte JSON se escribe en tmp_path para no contaminar el proyecto.
"""

import pandas as pd
import pytest

from src.data.validate_data import validate
from src.features.feature_engineering import FEATURE_MATRIX_COLUMNS, NULLABLE_COLUMNS

_NON_NULLABLE = [c for c in FEATURE_MATRIX_COLUMNS if c not in NULLABLE_COLUMNS]


# ── Fixture: DataFrame mínimo válido ──────────────────────────────────────────

def _make_valid_df(n: int = 4) -> pd.DataFrame:
    """Genera un DataFrame sintético que pasa todas las validaciones."""
    data: dict = {}
    for col in FEATURE_MATRIX_COLUMNS:
        data[col] = [1.0] * n

    # IDs únicos → sin duplicados
    data["user_key"] = list(range(1, n + 1))
    data["product_key"] = list(range(101, 101 + n))

    # Columnas con restricciones específicas
    data["label"] = [0, 1, 0, 1][:n]
    data["up_times_purchased"] = [2, 3, 1, 4][:n]
    data["user_total_orders"] = [5, 8, 6, 7][:n]

    # Columnas nullable pueden tener NaN (son aceptadas)
    for col in NULLABLE_COLUMNS:
        data[col] = [None, None, 1.0, 2.0][:n]

    return pd.DataFrame(data)


# ── Tests: caso feliz ─────────────────────────────────────────────────────────

def test_valid_dataframe_passes_all_checks(tmp_path):
    df = _make_valid_df()
    report = validate(df, output_dir=str(tmp_path))

    assert report["all_passed"] is True
    assert report["columns_present"]["passed"] is True
    assert report["no_unexpected_nulls"]["passed"] is True
    assert report["no_duplicate_pairs"]["passed"] is True
    assert report["label_binary"]["passed"] is True
    assert report["up_times_purchased_positive"]["passed"] is True
    assert report["user_total_orders_positive"]["passed"] is True


def test_report_json_is_written(tmp_path):
    df = _make_valid_df()
    validate(df, output_dir=str(tmp_path))

    report_file = tmp_path / "validation_report.json"
    assert report_file.exists()


# ── Tests: columnas presentes ────────────────────────────────────────────────

def test_missing_column_fails(tmp_path):
    df = _make_valid_df().drop(columns=["user_total_orders"])
    report = validate(df, output_dir=str(tmp_path))

    assert report["all_passed"] is False
    assert report["columns_present"]["passed"] is False
    assert "user_total_orders" in report["columns_present"]["detail"]


def test_multiple_missing_columns_reported(tmp_path):
    df = _make_valid_df().drop(columns=["user_total_orders", "product_reorder_rate"])
    report = validate(df, output_dir=str(tmp_path))

    assert report["columns_present"]["passed"] is False
    assert "user_total_orders" in report["columns_present"]["detail"]
    assert "product_reorder_rate" in report["columns_present"]["detail"]


# ── Tests: nulos inesperados ──────────────────────────────────────────────────

def test_null_in_non_nullable_column_fails(tmp_path):
    df = _make_valid_df()
    df.loc[0, "user_total_orders"] = None
    report = validate(df, output_dir=str(tmp_path))

    assert report["all_passed"] is False
    assert report["no_unexpected_nulls"]["passed"] is False
    assert "user_total_orders" in report["no_unexpected_nulls"]["detail"]


def test_null_in_nullable_column_passes(tmp_path):
    df = _make_valid_df()
    # up_avg_days_between_orders es NULLABLE — NaN permitido
    df["up_avg_days_between_orders"] = None
    report = validate(df, output_dir=str(tmp_path))

    assert report["no_unexpected_nulls"]["passed"] is True


def test_multiple_nulls_in_non_nullable_reported(tmp_path):
    df = _make_valid_df()
    df.loc[0, "user_total_orders"] = None
    df.loc[1, "up_times_purchased"] = None
    report = validate(df, output_dir=str(tmp_path))

    detail = report["no_unexpected_nulls"]["detail"]
    assert "user_total_orders" in detail
    assert "up_times_purchased" in detail


# ── Tests: duplicados ─────────────────────────────────────────────────────────

def test_duplicate_user_product_pair_fails(tmp_path):
    df = _make_valid_df()
    # Agrega una fila duplicada para (user_key=1, product_key=101)
    duplicate = df.iloc[[0]].copy()
    df = pd.concat([df, duplicate], ignore_index=True)

    report = validate(df, output_dir=str(tmp_path))

    assert report["all_passed"] is False
    assert report["no_duplicate_pairs"]["passed"] is False
    assert "1 pares" in report["no_duplicate_pairs"]["detail"]


def test_no_duplicates_passes(tmp_path):
    df = _make_valid_df()
    report = validate(df, output_dir=str(tmp_path))

    assert report["no_duplicate_pairs"]["passed"] is True


# ── Tests: label binario ──────────────────────────────────────────────────────

def test_non_binary_label_fails(tmp_path):
    df = _make_valid_df()
    df.loc[0, "label"] = 2
    report = validate(df, output_dir=str(tmp_path))

    assert report["all_passed"] is False
    assert report["label_binary"]["passed"] is False
    assert "2" in str(report["label_binary"]["detail"])


def test_binary_labels_pass(tmp_path):
    df = _make_valid_df()
    report = validate(df, output_dir=str(tmp_path))

    assert report["label_binary"]["passed"] is True


def test_all_zeros_label_passes(tmp_path):
    df = _make_valid_df()
    df["label"] = 0
    report = validate(df, output_dir=str(tmp_path))

    assert report["label_binary"]["passed"] is True


# ── Tests: up_times_purchased > 0 ────────────────────────────────────────────

def test_zero_up_times_purchased_fails(tmp_path):
    df = _make_valid_df()
    df.loc[0, "up_times_purchased"] = 0
    report = validate(df, output_dir=str(tmp_path))

    assert report["all_passed"] is False
    assert report["up_times_purchased_positive"]["passed"] is False
    assert "1 filas" in report["up_times_purchased_positive"]["detail"]


def test_negative_up_times_purchased_fails(tmp_path):
    df = _make_valid_df()
    df.loc[0, "up_times_purchased"] = -1
    report = validate(df, output_dir=str(tmp_path))

    assert report["up_times_purchased_positive"]["passed"] is False


def test_positive_up_times_purchased_passes(tmp_path):
    df = _make_valid_df()
    report = validate(df, output_dir=str(tmp_path))

    assert report["up_times_purchased_positive"]["passed"] is True


# ── Tests: user_total_orders > 0 ─────────────────────────────────────────────

def test_zero_user_total_orders_fails(tmp_path):
    df = _make_valid_df()
    df.loc[0, "user_total_orders"] = 0
    report = validate(df, output_dir=str(tmp_path))

    assert report["all_passed"] is False
    assert report["user_total_orders_positive"]["passed"] is False
    assert "1 filas" in report["user_total_orders_positive"]["detail"]


def test_negative_user_total_orders_fails(tmp_path):
    df = _make_valid_df()
    df.loc[1, "user_total_orders"] = -5
    report = validate(df, output_dir=str(tmp_path))

    assert report["user_total_orders_positive"]["passed"] is False


def test_positive_user_total_orders_passes(tmp_path):
    df = _make_valid_df()
    report = validate(df, output_dir=str(tmp_path))

    assert report["user_total_orders_positive"]["passed"] is True


# ── Tests: múltiples fallos simultáneos ──────────────────────────────────────

def test_multiple_failures_all_reported(tmp_path):
    df = _make_valid_df()
    df.loc[0, "label"] = 99
    df.loc[1, "up_times_purchased"] = 0
    df.loc[2, "user_total_orders"] = 0

    report = validate(df, output_dir=str(tmp_path))

    assert report["all_passed"] is False
    assert report["label_binary"]["passed"] is False
    assert report["up_times_purchased_positive"]["passed"] is False
    assert report["user_total_orders_positive"]["passed"] is False
