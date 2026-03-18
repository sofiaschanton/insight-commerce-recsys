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

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.model_monitoring import (
    MONITORED_FEATURES,
    S3_BUCKET,
    _compute_psi,
    _load_data,
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


# ── Tests: _load_data con S3 ──────────────────────────────────────────────────


def _make_parquet_bytes() -> bytes:
    """Genera bytes de un parquet sintético con las columnas monitoreadas."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {feat: rng.normal(size=10) for feat in MONITORED_FEATURES}
    )
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


class TestLoadDataS3:

    def test_load_data_s3_reads_current_from_s3(self, monkeypatch, tmp_path):
        """Cuando USE_S3=true, _load_data descarga feature_matrix.parquet desde S3."""
        monkeypatch.setattr("src.model_monitoring.USE_S3", True)

        parquet_bytes = _make_parquet_bytes()
        call_count = 0

        def download_side_effect(bucket, key, fileobj):
            nonlocal call_count
            call_count += 1
            fileobj.write(parquet_bytes)

        mock_s3 = MagicMock()
        mock_s3.download_fileobj.side_effect = download_side_effect

        with patch("src.model_monitoring.boto3.client", return_value=mock_s3):
            current_path = str(tmp_path / "feature_matrix.parquet")
            reference_path = str(tmp_path / "feature_matrix_reference.parquet")
            df_ref, df_curr = _load_data(current_path, reference_path)

        assert mock_s3.download_fileobj.call_count == 2
        first_call_args = mock_s3.download_fileobj.call_args_list[0][0]
        assert first_call_args[0] == S3_BUCKET
        assert first_call_args[1] == "monitoring/actual/feature_matrix.parquet"
        assert df_curr is not None
        assert df_ref is not None

    def test_load_data_s3_returns_none_when_reference_missing(self, monkeypatch, tmp_path):
        """Cuando USE_S3=true y el reference parquet no existe en S3, devuelve (None, None)."""
        monkeypatch.setattr("src.model_monitoring.USE_S3", True)

        parquet_bytes = _make_parquet_bytes()
        call_count = 0

        def download_side_effect(bucket, key, fileobj):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                fileobj.write(parquet_bytes)
            else:
                raise OSError("NoSuchKey: reference not found in S3")

        mock_s3 = MagicMock()
        mock_s3.download_fileobj.side_effect = download_side_effect

        with patch("src.model_monitoring.boto3.client", return_value=mock_s3):
            current_path = str(tmp_path / "feature_matrix.parquet")
            reference_path = str(tmp_path / "feature_matrix_reference.parquet")
            result = _load_data(current_path, reference_path)

        assert result == (None, None)

    def test_load_data_s3_exits_when_current_missing(self, monkeypatch, tmp_path):
        """Cuando USE_S3=true y el current parquet no existe en S3, llama sys.exit(1)."""
        monkeypatch.setattr("src.model_monitoring.USE_S3", True)

        mock_s3 = MagicMock()
        mock_s3.download_fileobj.side_effect = OSError("NoSuchKey: current not found in S3")

        with patch("src.model_monitoring.boto3.client", return_value=mock_s3):
            current_path = str(tmp_path / "feature_matrix.parquet")
            reference_path = str(tmp_path / "feature_matrix_reference.parquet")
            with pytest.raises(SystemExit) as exc_info:
                _load_data(current_path, reference_path)

        assert exc_info.value.code == 1
