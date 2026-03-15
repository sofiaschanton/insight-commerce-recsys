"""
tests/test_inference.py
========================
Tests del contrato de features para src/api/inference.py.

Foco: RecommendationService._align_and_validate — la función que garantiza
que la matriz de features online coincide exactamente con lo que espera
el modelo LightGBM entrenado.

Todos los tests usan mocks para evitar dependencias externas
(base de datos PostgreSQL, archivos de modelo .pkl).
"""

import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock

from src.api.inference import RecommendationService, FeatureContractError


# ─── Helper ───────────────────────────────────────────────────────────────────

def _make_service(feature_cols: list[str]) -> RecommendationService:
    """
    Crea un RecommendationService con artefactos mockeados.
    Solo configura los atributos que usa _align_and_validate.
    """
    service = object.__new__(RecommendationService)
    service.feature_cols = feature_cols
    service.n_features = len(feature_cols)
    service._artifacts = MagicMock()
    service._engine = MagicMock()
    return service


def _make_matrix(cols: list[str], n_rows: int = 3) -> pd.DataFrame:
    """Construye un DataFrame con columnas dadas y valores aleatorios."""
    return pd.DataFrame(
        np.random.rand(n_rows, len(cols)),
        columns=cols,
    )


# ─── FeatureContractError: columnas faltantes ─────────────────────────────────

class TestMissingColumns:

    def test_raises_when_single_column_missing(self):
        service = _make_service(["col_a", "col_b", "col_c"])
        matrix = _make_matrix(["col_a", "col_b"])          # falta col_c
        with pytest.raises(FeatureContractError, match="col_c"):
            service._align_and_validate(matrix)

    def test_raises_when_multiple_columns_missing(self):
        service = _make_service(["col_a", "col_b", "col_c", "col_d"])
        matrix = _make_matrix(["col_a"])
        with pytest.raises(FeatureContractError):
            service._align_and_validate(matrix)

    def test_raises_when_matrix_is_empty_of_required_columns(self):
        service = _make_service(["user_total_orders", "product_reorder_rate"])
        matrix = pd.DataFrame({"irrelevante": [1, 2, 3]})
        with pytest.raises(FeatureContractError):
            service._align_and_validate(matrix)


# ─── FeatureContractError: n_features no coincide ────────────────────────────

class TestNFeaturesMismatch:

    def test_raises_when_n_features_overridden_to_wrong_value(self):
        """El modelo dice tener 3 features pero feature_cols solo lista 2."""
        service = _make_service(["col_a", "col_b"])
        service.n_features = 3     # inconsistencia intencional
        matrix = _make_matrix(["col_a", "col_b"])
        with pytest.raises(FeatureContractError, match="n_features"):
            service._align_and_validate(matrix)


# ─── Selección y ordenamiento correcto de columnas ───────────────────────────

class TestColumnSelection:

    def test_selects_only_feature_cols(self):
        """Columnas extra en la matriz no deben aparecer en el output."""
        service = _make_service(["col_a", "col_b"])
        matrix = _make_matrix(["col_a", "col_b", "columna_extra", "otra_extra"])
        result = service._align_and_validate(matrix)
        assert "columna_extra" not in result.columns
        assert "otra_extra" not in result.columns

    def test_preserves_feature_col_order(self):
        """Las columnas del output deben estar en el mismo orden que feature_cols."""
        service = _make_service(["col_c", "col_a", "col_b"])
        matrix = _make_matrix(["col_a", "col_b", "col_c"])   # orden distinto
        result = service._align_and_validate(matrix)
        assert list(result.columns) == ["col_c", "col_a", "col_b"]

    def test_output_shape_matches_input_rows(self):
        """El número de filas no debe cambiar."""
        service = _make_service(["col_a", "col_b"])
        matrix = _make_matrix(["col_a", "col_b"], n_rows=7)
        result = service._align_and_validate(matrix)
        assert result.shape == (7, 2)


# ─── Casos borde ──────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_passes_with_exact_features_in_correct_order(self):
        """El caso feliz: no debe lanzar excepción."""
        cols = ["user_total_orders", "product_reorder_rate", "up_times_purchased"]
        service = _make_service(cols)
        matrix = _make_matrix(cols)
        result = service._align_and_validate(matrix)
        assert result.shape == (3, 3)
        assert list(result.columns) == cols

    def test_empty_matrix_zero_rows_passes(self):
        """DataFrame con cero filas y columnas correctas debe pasar el contrato."""
        cols = ["col_a", "col_b"]
        service = _make_service(cols)
        matrix = pd.DataFrame(columns=cols)
        result = service._align_and_validate(matrix)
        assert len(result) == 0
        assert list(result.columns) == cols

    def test_single_feature_contract(self):
        """Modelo con una sola feature debe funcionar."""
        service = _make_service(["solo_feature"])
        matrix = pd.DataFrame({"solo_feature": [0.5, 0.8], "ruido": [1, 2]})
        result = service._align_and_validate(matrix)
        assert list(result.columns) == ["solo_feature"]
        assert result.shape == (2, 1)

    def test_values_are_preserved_unchanged(self):
        """Los valores de las features no deben ser modificados por _align_and_validate."""
        service = _make_service(["col_a"])
        original_values = [1.1, 2.2, 3.3]
        matrix = pd.DataFrame({"col_a": original_values, "extra": [0, 0, 0]})
        result = service._align_and_validate(matrix)
        assert list(result["col_a"]) == original_values

    def test_nan_values_are_preserved(self):
        """Los NaN intencionales (ej. up_avg_days_between_orders) no se deben imputar aquí."""
        service = _make_service(["col_with_nan"])
        matrix = pd.DataFrame({"col_with_nan": [1.0, float("nan"), 3.0]})
        result = service._align_and_validate(matrix)
        assert result["col_with_nan"].isna().sum() == 1
