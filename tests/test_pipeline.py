"""
tests/test_pipeline.py
======================
Tests unitarios para src/pipeline.py.

Mockea load_data_from_aws, build_feature_matrix, validate_feature_matrix,
train y mlflow para probar solo la lógica de orquestación del pipeline,
sin conexión a base de datos, archivos ni MLflow real.
"""

from unittest.mock import MagicMock, patch, call

import pandas as pd
import pytest

# ── Constantes de los mocks ───────────────────────────────────────────────────

_FAKE_DATA = {
    "fact_order_products": pd.DataFrame({"order_id": range(100)}),
    "dim_product": pd.DataFrame({"product_key": range(50)}),
}

_FAKE_MATRIX = pd.DataFrame({
    "user_key": [1, 2, 3],
    "product_key": [101, 102, 103],
    "label": [0, 1, 0],
})

_FAKE_TRAIN_RESULT = {
    "model": MagicMock(),
    "cluster_models": MagicMock(),
    "feature_cols": ["user_key", "product_key"],
    "metrics": {"precision": 0.85, "recall": 0.80, "f1": 0.82, "auc": 0.90},
    "best_params": {"n_estimators": 100},
}

_VALIDATION_OK = {
    "all_passed": True,
    "columns_present": {"passed": True},
    "no_unexpected_nulls": {"passed": True},
    "no_duplicate_pairs": {"passed": True},
    "label_binary": {"passed": True},
    "up_times_purchased_positive": {"passed": True},
    "user_total_orders_positive": {"passed": True},
}

_VALIDATION_FAILED = {
    "all_passed": False,
    "columns_present": {"passed": True},
    "label_binary": {"passed": False, "detail": "valores inválidos: [2]"},
}


# ── Fixture: patch de todas las dependencias externas ────────────────────────

@pytest.fixture
def pipeline_mocks():
    """
    Parchea todas las dependencias del pipeline y devuelve un dict con los mocks
    para que cada test pueda inspeccionarlos o modificar su comportamiento.
    """
    with (
        patch("src.pipeline.load_data_from_aws", return_value=_FAKE_DATA) as mock_load,
        patch("src.pipeline.build_feature_matrix", return_value=_FAKE_MATRIX) as mock_build,
        patch("src.pipeline.validate_feature_matrix", return_value=_VALIDATION_OK) as mock_validate,
        patch("src.pipeline.train", return_value=_FAKE_TRAIN_RESULT) as mock_train,
        patch("src.pipeline.mlflow") as mock_mlflow,
    ):
        # Configurar mlflow.start_run() como context manager válido
        mock_run = MagicMock()
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        # Configurar MODELS_DIR artifacts que se loguean
        with patch("src.pipeline.MODELS_DIR", MagicMock()) as mock_models_dir:
            mock_models_dir.__truediv__ = lambda self, x: MagicMock(__str__=lambda s: f"models/{x}")

            yield {
                "load": mock_load,
                "build": mock_build,
                "validate": mock_validate,
                "train": mock_train,
                "mlflow": mock_mlflow,
            }


# ── Tests: flujo exitoso ──────────────────────────────────────────────────────

def test_run_pipeline_returns_train_result(pipeline_mocks):
    from src.pipeline import run_pipeline
    result = run_pipeline()
    assert result == _FAKE_TRAIN_RESULT


def test_run_pipeline_calls_load_data(pipeline_mocks):
    from src.pipeline import run_pipeline
    run_pipeline()
    pipeline_mocks["load"].assert_called_once()


def test_run_pipeline_calls_build_feature_matrix(pipeline_mocks):
    from src.pipeline import run_pipeline
    run_pipeline(n_users=500)
    pipeline_mocks["build"].assert_called_once_with(_FAKE_DATA, output_path=None)


def test_run_pipeline_calls_validate(pipeline_mocks):
    from src.pipeline import run_pipeline
    run_pipeline()
    pipeline_mocks["validate"].assert_called_once_with(_FAKE_MATRIX, output_dir="reports/data")


def test_run_pipeline_calls_train(pipeline_mocks):
    from src.pipeline import run_pipeline
    run_pipeline(n_users=None, n_optuna_trials=5, run_optuna_flag=True, random_state=42)
    pipeline_mocks["train"].assert_called_once()


def test_run_pipeline_passes_n_users_to_load(pipeline_mocks):
    from src.pipeline import run_pipeline
    run_pipeline(n_users=1000)
    call_kwargs = pipeline_mocks["load"].call_args
    assert call_kwargs.kwargs.get("n_users") == 1000 or call_kwargs.args[0] == 1000


def test_run_pipeline_starts_mlflow_run(pipeline_mocks):
    from src.pipeline import run_pipeline
    run_pipeline()
    pipeline_mocks["mlflow"].start_run.assert_called_once_with(run_name="full_pipeline")


def test_run_pipeline_logs_mlflow_params(pipeline_mocks):
    from src.pipeline import run_pipeline
    run_pipeline()
    pipeline_mocks["mlflow"].log_params.assert_called()


def test_run_pipeline_logs_mlflow_metrics(pipeline_mocks):
    from src.pipeline import run_pipeline
    run_pipeline()
    pipeline_mocks["mlflow"].log_metrics.assert_called_once_with(_FAKE_TRAIN_RESULT["metrics"])


# ── Tests: propagación de error de validación ────────────────────────────────

def test_run_pipeline_raises_value_error_when_validation_fails(pipeline_mocks):
    pipeline_mocks["validate"].return_value = _VALIDATION_FAILED

    from src.pipeline import run_pipeline
    with pytest.raises(ValueError, match="Validación del feature matrix fallida"):
        run_pipeline()


def test_run_pipeline_error_message_includes_failed_checks(pipeline_mocks):
    pipeline_mocks["validate"].return_value = _VALIDATION_FAILED

    from src.pipeline import run_pipeline
    with pytest.raises(ValueError) as exc_info:
        run_pipeline()

    assert "label_binary" in str(exc_info.value)


def test_run_pipeline_does_not_call_train_when_validation_fails(pipeline_mocks):
    pipeline_mocks["validate"].return_value = _VALIDATION_FAILED

    from src.pipeline import run_pipeline
    with pytest.raises(ValueError):
        run_pipeline()

    pipeline_mocks["train"].assert_not_called()


# ── Tests: parámetros opcionales ─────────────────────────────────────────────

def test_run_pipeline_no_optuna(pipeline_mocks):
    from src.pipeline import run_pipeline
    result = run_pipeline(run_optuna_flag=False)
    assert result == _FAKE_TRAIN_RESULT
    call_kwargs = pipeline_mocks["train"].call_args
    assert call_kwargs.kwargs.get("run_optuna_flag") is False


def test_run_pipeline_custom_random_state(pipeline_mocks):
    from src.pipeline import run_pipeline
    run_pipeline(random_state=123)
    call_kwargs = pipeline_mocks["train"].call_args
    assert call_kwargs.kwargs.get("random_state") == 123


def test_run_pipeline_custom_trials(pipeline_mocks):
    from src.pipeline import run_pipeline
    run_pipeline(n_optuna_trials=50)
    call_kwargs = pipeline_mocks["train"].call_args
    assert call_kwargs.kwargs.get("n_optuna_trials") == 50
