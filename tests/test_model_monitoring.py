"""
tests/test_model_monitoring.py
================================
Tests unitarios para src/model_monitoring.py.

Foco:
  - _compute_psi: función pura sobre arrays numpy.
  - compute_drift_metrics: flujo completo con parquets sintéticos en tmp_path.

Todos los tests usan datos sintéticos — sin archivos reales ni conexión a DB.
Los umbrales probados coinciden con los definidos en model_monitoring.py:
    PSI >= 0.25 → drift significativo
    KS  >= 0.30 → diferencia estadística significativa
"""

import numpy as np
import pandas as pd
import pytest

from src.model_monitoring import (
    MONITORED_FEATURES,
    _compute_psi,
    compute_drift_metrics,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def stable_array():
    """Array de 500 muestras con distribución N(5, 1)."""
    return np.random.default_rng(42).normal(loc=5.0, scale=1.0, size=500)


@pytest.fixture
def parquet_paths(tmp_path):
    """
    Devuelve (ref_path, stable_path, shifted_path) con parquets sintéticos.

    - ref / stable  : N(5, 1) → distribuciones similares → PSI ≈ 0
    - shifted       : N(30, 1) → distribución muy desplazada → PSI >> 0.25
    """
    rng = np.random.default_rng(42)
    n = 500

    df_ref = pd.DataFrame(
        {feat: rng.normal(loc=5.0, scale=1.0, size=n) for feat in MONITORED_FEATURES}
    )
    df_stable = pd.DataFrame(
        {feat: rng.normal(loc=5.0, scale=1.0, size=n) for feat in MONITORED_FEATURES}
    )
    df_shifted = pd.DataFrame(
        {feat: rng.normal(loc=30.0, scale=1.0, size=n) for feat in MONITORED_FEATURES}
    )

    ref_path = tmp_path / "reference.parquet"
    stable_path = tmp_path / "current_stable.parquet"
    shifted_path = tmp_path / "current_shifted.parquet"

    df_ref.to_parquet(ref_path, index=False)
    df_stable.to_parquet(stable_path, index=False)
    df_shifted.to_parquet(shifted_path, index=False)

    return ref_path, stable_path, shifted_path


# ── Tests: _compute_psi ───────────────────────────────────────────────────────


def test_compute_drift_psi_stable_data(stable_array):
    """PSI debe ser cercano a 0 cuando referencia y actual son la misma distribución."""
    psi = _compute_psi(stable_array, stable_array)
    assert psi < 0.05


# ── Tests: compute_drift_metrics ─────────────────────────────────────────────


def test_compute_drift_alerts_on_shift(tmp_path, parquet_paths):
    """drift_detected debe ser True cuando la distribución actual está muy desplazada (PSI >> 0.25)."""
    ref_path, _, shifted_path = parquet_paths

    result = compute_drift_metrics(
        current_path=str(shifted_path),
        reference_path=str(ref_path),
        output_dir=str(tmp_path),
    )

    assert result["drift_detected"] is True


def test_drift_report_json_structure(tmp_path, parquet_paths):
    """El dict devuelto contiene las claves psi, ks, drift_detected, psi_by_feature, ks_by_feature."""
    ref_path, stable_path, _ = parquet_paths

    result = compute_drift_metrics(
        current_path=str(stable_path),
        reference_path=str(ref_path),
        output_dir=str(tmp_path),
    )

    for key in ("psi", "ks", "drift_detected", "psi_by_feature", "ks_by_feature"):
        assert key in result, f"Clave '{key}' ausente en drift_report"
