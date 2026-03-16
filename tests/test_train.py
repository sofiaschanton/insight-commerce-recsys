"""
tests/test_train.py
====================
Tests unitarios para src/models/train.py.

Cubre:
  - split_by_users y eval_metrics  (funciones puras — tests originales)
  - load_matrix                    (FileNotFoundError + éxito)
  - fit_kmeans / _assign_cluster   (con datos sintéticos reales de sklearn KMeans)
  - get_Xy                         (separación de features y label)
  - train() run_optuna_flag=False  (flujo completo con mocks de LightGBM y MLflow)
  - train() rollback               (RuntimeError si F1 nuevo < F1 anterior × 0.95)
  - train() rollback JSON inválido (JSONDecodeError capturado, continúa)
  - train() run_optuna_flag=True   (mock de optuna.create_study)

Todos los tests usan datos sintéticos sin DB ni archivos reales.
MLflow y joblib se mockean con monkeypatch.
LightGBM se mockea para evitar entrenamiento real.
"""

import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock, call

from src.models.train import (
    USER_CLUSTER_FEATURES,
    PRODUCT_CLUSTER_FEATURES,
    eval_metrics,
    fit_kmeans,
    get_Xy,
    load_matrix,
    split_by_users,
    train,
)


# ── Fixtures compartidos ──────────────────────────────────────────────────────

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


@pytest.fixture
def train_matrix():
    """
    Feature matrix sintética con todas las columnas necesarias para train().

    30 usuarios × 10 productos = 300 pares.
    Incluye USER_CLUSTER_FEATURES y PRODUCT_CLUSTER_FEATURES para que
    fit_kmeans funcione correctamente (necesita ≥ N_CLUSTERS_USER=5 usuarios).
    """
    rng = np.random.default_rng(99)
    n_users, n_prods = 30, 10
    rows = []
    for u in range(1, n_users + 1):
        for p in range(1, n_prods + 1):
            rows.append({
                "user_key":                   u,
                "product_key":                p,
                "label":                      int(rng.integers(0, 2)),
                # USER_CLUSTER_FEATURES
                "user_total_orders":          int(rng.integers(5, 25)),
                "user_avg_basket_size":       float(rng.uniform(3.0, 15.0)),
                "user_days_since_last_order": int(rng.integers(1, 60)),
                "user_reorder_ratio":         float(rng.uniform(0.1, 0.9)),
                "user_distinct_products":     int(rng.integers(5, 50)),
                # PRODUCT_CLUSTER_FEATURES
                "product_total_purchases":    int(rng.integers(50, 500)),
                "product_reorder_rate":       float(rng.uniform(0.1, 0.9)),
                "product_avg_add_to_cart":    float(rng.uniform(1.0, 10.0)),
                "product_unique_users":       int(rng.integers(10, 200)),
                "p_department_reorder_rate":  float(rng.uniform(0.1, 0.9)),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def mock_mlflow(monkeypatch):
    """Reemplaza el módulo mlflow en train.py por un MagicMock."""
    m = MagicMock()
    # start_run() necesita ser un context manager válido
    m.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
    m.start_run.return_value.__exit__ = MagicMock(return_value=False)
    monkeypatch.setattr("src.models.train.mlflow", m)
    return m


@pytest.fixture
def mock_joblib(monkeypatch):
    """Reemplaza joblib en train.py para no escribir archivos binarios reales."""
    m = MagicMock()
    monkeypatch.setattr("src.models.train.joblib", m)
    return m


@pytest.fixture
def mock_lgbm(monkeypatch):
    """
    Reemplaza lgb.LGBMClassifier por un mock que:
    - Setea feature_importances_ en fit() según la cantidad de features.
    - predict() devuelve todo 0s (garantiza F1≈0, útil para rollback).
    - predict_proba() devuelve probabilidades [1,0] (clase 0 = 1.0, clase 1 = 0.0).
    """
    instance = MagicMock()
    instance.best_iteration_ = 42

    def _fit(X, y, **kwargs):
        instance.feature_importances_ = np.ones(X.shape[1], dtype=int)
        instance.feature_importances_[0] = 10  # al menos uno no-zero
        return instance

    def _predict(X):
        return np.zeros(len(X), dtype=int)

    def _predict_proba(X):
        return np.column_stack([np.ones(len(X)), np.zeros(len(X))])

    instance.fit.side_effect = _fit
    instance.predict.side_effect = _predict
    instance.predict_proba.side_effect = _predict_proba

    cls = MagicMock(return_value=instance)
    monkeypatch.setattr("src.models.train.lgb.LGBMClassifier", cls)
    return instance


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


# ── Tests: load_matrix ────────────────────────────────────────────────────────

def test_load_matrix_raises_if_not_found(tmp_path):
    missing = tmp_path / "no_existe.parquet"
    with pytest.raises(FileNotFoundError, match="feature_matrix.parquet"):
        load_matrix(missing)


def test_load_matrix_returns_dataframe(tmp_path, synthetic_matrix):
    path = tmp_path / "feature_matrix.parquet"
    synthetic_matrix.to_parquet(path, index=False)

    result = load_matrix(path)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == synthetic_matrix.shape


# ── Tests: fit_kmeans ─────────────────────────────────────────────────────────

def test_fit_kmeans_adds_cluster_columns(train_matrix):
    t, v, te = split_by_users(train_matrix, random_state=42)
    t_out, v_out, te_out, _ = fit_kmeans(t, v, te, random_state=42)

    for df in (t_out, v_out, te_out):
        assert "user_cluster" in df.columns
        assert "product_cluster" in df.columns


def test_fit_kmeans_cluster_values_in_valid_range(train_matrix):
    t, v, te = split_by_users(train_matrix, random_state=42)
    t_out, v_out, te_out, _ = fit_kmeans(t, v, te, random_state=42)

    for df in (t_out, v_out, te_out):
        assert df["user_cluster"].isin(range(-1, 6)).all()
        assert df["product_cluster"].isin(range(-1, 6)).all()


def test_fit_kmeans_returns_cluster_models_dict(train_matrix):
    t, v, te = split_by_users(train_matrix, random_state=42)
    _, _, _, cluster_models = fit_kmeans(t, v, te, random_state=42)

    assert "kmeans_user" in cluster_models
    assert "kmeans_product" in cluster_models
    assert "scaler_user" in cluster_models
    assert "scaler_product" in cluster_models


def test_fit_kmeans_preserves_row_count(train_matrix):
    t, v, te = split_by_users(train_matrix, random_state=42)
    n_t, n_v, n_te = len(t), len(v), len(te)
    t_out, v_out, te_out, _ = fit_kmeans(t, v, te, random_state=42)

    assert len(t_out) == n_t
    assert len(v_out) == n_v
    assert len(te_out) == n_te


# ── Tests: get_Xy ─────────────────────────────────────────────────────────────

def test_get_xy_excludes_id_and_label_cols(train_matrix):
    t, v, te = split_by_users(train_matrix, random_state=42)
    t, v, te, _ = fit_kmeans(t, v, te, random_state=42)
    _, _, _, _, _, _, feature_cols = get_Xy(t, v, te)

    assert "user_key" not in feature_cols
    assert "product_key" not in feature_cols
    assert "label" not in feature_cols


def test_get_xy_label_is_series(train_matrix):
    t, v, te = split_by_users(train_matrix, random_state=42)
    t, v, te, _ = fit_kmeans(t, v, te, random_state=42)
    _, y_tr, _, y_v, _, y_te, _ = get_Xy(t, v, te)

    assert isinstance(y_tr, pd.Series)
    assert isinstance(y_v, pd.Series)
    assert isinstance(y_te, pd.Series)


def test_get_xy_shapes_consistent(train_matrix):
    t, v, te = split_by_users(train_matrix, random_state=42)
    t, v, te, _ = fit_kmeans(t, v, te, random_state=42)
    x_tr, y_tr, x_v, y_v, x_te, y_te, feature_cols = get_Xy(t, v, te)

    assert x_tr.shape[0] == len(y_tr)
    assert x_v.shape[0] == len(y_v)
    assert x_te.shape[0] == len(y_te)
    assert x_tr.shape[1] == len(feature_cols)


# ── Tests: train() — flujo completo sin Optuna ───────────────────────────────

def test_train_no_optuna_returns_expected_keys(tmp_path, train_matrix, mock_mlflow, mock_joblib, mock_lgbm):
    result = train(matrix=train_matrix, models_dir=tmp_path, run_optuna_flag=False)

    assert set(result.keys()) >= {"model", "cluster_models", "feature_cols", "metrics", "best_params", "importance"}


def test_train_no_optuna_metrics_are_floats(tmp_path, train_matrix, mock_mlflow, mock_joblib, mock_lgbm):
    result = train(matrix=train_matrix, models_dir=tmp_path, run_optuna_flag=False)

    for key in ("precision", "recall", "f1", "auc"):
        assert key in result["metrics"]
        assert isinstance(result["metrics"][key], float)


def test_train_no_optuna_feature_cols_excludes_ids(tmp_path, train_matrix, mock_mlflow, mock_joblib, mock_lgbm):
    result = train(matrix=train_matrix, models_dir=tmp_path, run_optuna_flag=False)

    assert "user_key" not in result["feature_cols"]
    assert "product_key" not in result["feature_cols"]
    assert "label" not in result["feature_cols"]


def test_train_no_optuna_calls_joblib_dump_twice(tmp_path, train_matrix, mock_mlflow, mock_joblib, mock_lgbm):
    train(matrix=train_matrix, models_dir=tmp_path, run_optuna_flag=False)

    assert mock_joblib.dump.call_count == 2


def test_train_no_optuna_dumps_model_and_clusters(tmp_path, train_matrix, mock_mlflow, mock_joblib, mock_lgbm):
    train(matrix=train_matrix, models_dir=tmp_path, run_optuna_flag=False)

    dumped_paths = [str(c.args[1]) for c in mock_joblib.dump.call_args_list]
    assert any("model.pkl" in p for p in dumped_paths)
    assert any("cluster_models.pkl" in p for p in dumped_paths)


def test_train_no_optuna_writes_model_log_json(tmp_path, train_matrix, mock_mlflow, mock_joblib, mock_lgbm):
    train(matrix=train_matrix, models_dir=tmp_path, run_optuna_flag=False)

    log_path = tmp_path / "model_log.json"
    assert log_path.exists()


def test_train_model_log_json_has_expected_keys(tmp_path, train_matrix, mock_mlflow, mock_joblib, mock_lgbm):
    train(matrix=train_matrix, models_dir=tmp_path, run_optuna_flag=False)

    log = json.loads((tmp_path / "model_log.json").read_text())
    for key in ("timestamp", "model_name", "metrics_test", "best_params", "feature_cols", "split"):
        assert key in log, f"Clave '{key}' ausente en model_log.json"


def test_train_no_optuna_best_params_has_defaults(tmp_path, train_matrix, mock_mlflow, mock_joblib, mock_lgbm):
    result = train(matrix=train_matrix, models_dir=tmp_path, run_optuna_flag=False)

    params = result["best_params"]
    assert params["n_estimators"] == 500
    assert params["learning_rate"] == pytest.approx(0.05)
    assert params["num_leaves"] == 63


def test_train_calls_mlflow_start_run(tmp_path, train_matrix, mock_mlflow, mock_joblib, mock_lgbm):
    train(matrix=train_matrix, models_dir=tmp_path, run_optuna_flag=False)

    mock_mlflow.start_run.assert_called_once_with(run_name="lgbm_kmeans_pipeline")


def test_train_logs_mlflow_metrics(tmp_path, train_matrix, mock_mlflow, mock_joblib, mock_lgbm):
    result = train(matrix=train_matrix, models_dir=tmp_path, run_optuna_flag=False)

    mock_mlflow.log_metrics.assert_called_once_with(result["metrics"])


# ── Tests: train() — rollback ─────────────────────────────────────────────────

def test_train_rollback_raises_runtime_error(tmp_path, train_matrix, mock_mlflow, mock_joblib, mock_lgbm):
    """F1 anterior=0.99, F1 nuevo≈0.0 → degradación > 5% → RuntimeError."""
    old_log = {"metrics_test": {"f1": 0.99, "precision": 0.99, "recall": 0.99, "auc": 0.99}}
    (tmp_path / "model_log.json").write_text(json.dumps(old_log), encoding="utf-8")

    with pytest.raises(RuntimeError, match="Rollback"):
        train(matrix=train_matrix, models_dir=tmp_path, run_optuna_flag=False)


def test_train_rollback_error_message_contains_f1_values(tmp_path, train_matrix, mock_mlflow, mock_joblib, mock_lgbm):
    old_log = {"metrics_test": {"f1": 0.99}}
    (tmp_path / "model_log.json").write_text(json.dumps(old_log), encoding="utf-8")

    with pytest.raises(RuntimeError) as exc_info:
        train(matrix=train_matrix, models_dir=tmp_path, run_optuna_flag=False)

    assert "0.9900" in str(exc_info.value)  # old F1
    assert "Umbral" in str(exc_info.value)


def test_train_rollback_does_not_call_joblib_dump(tmp_path, train_matrix, mock_mlflow, mock_joblib, mock_lgbm):
    """Cuando el rollback se activa, los artefactos NO deben ser sobreescritos."""
    old_log = {"metrics_test": {"f1": 0.99}}
    (tmp_path / "model_log.json").write_text(json.dumps(old_log), encoding="utf-8")

    with pytest.raises(RuntimeError):
        train(matrix=train_matrix, models_dir=tmp_path, run_optuna_flag=False)

    mock_joblib.dump.assert_not_called()


def test_train_rollback_preserves_old_model_log(tmp_path, train_matrix, mock_mlflow, mock_joblib, mock_lgbm):
    """El JSON original no debe ser sobreescrito si el rollback se activa."""
    old_log = {"metrics_test": {"f1": 0.99}, "model_name": "old_model"}
    (tmp_path / "model_log.json").write_text(json.dumps(old_log), encoding="utf-8")

    with pytest.raises(RuntimeError):
        train(matrix=train_matrix, models_dir=tmp_path, run_optuna_flag=False)

    surviving_log = json.loads((tmp_path / "model_log.json").read_text())
    assert surviving_log["model_name"] == "old_model"


def test_train_no_rollback_when_f1_improves(tmp_path, train_matrix, mock_mlflow, mock_joblib, mock_lgbm):
    """Si F1 nuevo ≥ F1 anterior × 0.95, el entrenamiento debe completarse."""
    # F1 anterior muy bajo (0.0) → cualquier F1 nuevo pasa el umbral
    old_log = {"metrics_test": {"f1": 0.0}}
    (tmp_path / "model_log.json").write_text(json.dumps(old_log), encoding="utf-8")

    result = train(matrix=train_matrix, models_dir=tmp_path, run_optuna_flag=False)

    assert "model" in result


def test_train_rollback_invalid_json_continues(tmp_path, train_matrix, mock_mlflow, mock_joblib, mock_lgbm):
    """Si model_log.json tiene JSON inválido, se captura el error y el train continúa."""
    (tmp_path / "model_log.json").write_text("{ esto no es json !!!}", encoding="utf-8")

    result = train(matrix=train_matrix, models_dir=tmp_path, run_optuna_flag=False)

    assert "model" in result


# ── Tests: train() — con Optuna (mockeado) ───────────────────────────────────

def test_train_with_optuna_calls_create_study(tmp_path, train_matrix, mock_mlflow, mock_joblib, mock_lgbm):
    """run_optuna_flag=True debe llamar a optuna.create_study."""
    import src.models.train as train_module

    study = MagicMock()
    study.best_params = {
        "n_estimators": 100, "learning_rate": 0.05, "num_leaves": 31,
        "max_depth": 6, "min_child_samples": 20,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_alpha": 0.1, "reg_lambda": 0.1,
    }
    study.best_value = 0.75
    train_module.optuna.create_study.return_value = study

    train(matrix=train_matrix, models_dir=tmp_path, run_optuna_flag=True, n_optuna_trials=1)

    train_module.optuna.create_study.assert_called_once()


def test_train_with_optuna_returns_best_params_from_study(tmp_path, train_matrix, mock_mlflow, mock_joblib, mock_lgbm):
    """Los best_params del resultado deben incluir los hiperparámetros de Optuna."""
    import src.models.train as train_module

    study = MagicMock()
    study.best_params = {
        "n_estimators": 123, "learning_rate": 0.07, "num_leaves": 45,
        "max_depth": 5, "min_child_samples": 30,
        "subsample": 0.9, "colsample_bytree": 0.7,
        "reg_alpha": 0.01, "reg_lambda": 0.01,
    }
    study.best_value = 0.80
    train_module.optuna.create_study.return_value = study

    result = train(matrix=train_matrix, models_dir=tmp_path, run_optuna_flag=True, n_optuna_trials=1)

    assert result["best_params"]["n_estimators"] == 123
    assert result["best_params"]["learning_rate"] == pytest.approx(0.07)


def test_train_with_optuna_study_optimize_called(tmp_path, train_matrix, mock_mlflow, mock_joblib, mock_lgbm):
    """study.optimize debe invocarse con el número de trials indicado."""
    import src.models.train as train_module

    study = MagicMock()
    study.best_params = {
        "n_estimators": 100, "learning_rate": 0.05, "num_leaves": 31,
        "max_depth": 4, "min_child_samples": 15,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_alpha": 0.05, "reg_lambda": 0.05,
    }
    study.best_value = 0.70
    train_module.optuna.create_study.return_value = study

    train(matrix=train_matrix, models_dir=tmp_path, run_optuna_flag=True, n_optuna_trials=3)

    study.optimize.assert_called_once()
    call_kwargs = study.optimize.call_args
    assert call_kwargs.kwargs.get("n_trials") == 3 or call_kwargs.args[1] == 3
