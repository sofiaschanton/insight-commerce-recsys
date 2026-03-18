"""
tests/test_inference.py
========================
Tests unitarios para src/api/inference.py.

Cubre:
  - _align_and_validate        (tests originales: contrato de features)
  - startup()                  (carga artefactos y setea atributos)
  - _load_artifacts()          (joblib.load + json con tmp_path)
  - _build_engine()            (host local baja sslmode, host remoto no, sin host → error)
  - _read_sql()                (éxito + OperationalError → DatabaseConnectionError)
  - _query_user_prior()        (delega en _read_sql con SQL correcto)
  - _assign_user_cluster()     (usuario desconocido → -1, usuario conocido → cluster)
  - _assign_product_clusters() (productos desconocidos → -1, conocidos → cluster)
  - _build_online_matrix()     (prior vacío → UserNotFoundError)
  - recommend_user()           (flujo completo, top_k, nombres de producto, errores)

Sin DB real ni archivos .pkl reales. SQLAlchemy y joblib se mockean con monkeypatch.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch, call

import numpy as np
import pandas as pd
import pytest
from sqlalchemy.exc import OperationalError

from src.api.inference import (
    _ClusterArtifacts,
    DatabaseConnectionError,
    FeatureContractError,
    LoadedArtifacts,
    RecommendationService,
    UserNotFoundError,
)
from src.models.train import PRODUCT_CLUSTER_FEATURES, USER_CLUSTER_FEATURES


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_service(feature_cols: list) -> RecommendationService:
    """
    Crea un RecommendationService con artefactos mockeados.
    Solo configura los atributos que usa _align_and_validate.
    """
    service = object.__new__(RecommendationService)
    service.feature_cols = feature_cols
    service.n_features = len(feature_cols)
    service._artifacts = MagicMock()
    service.engine = MagicMock()
    return service


def _make_matrix(cols: list, n_rows: int = 3) -> pd.DataFrame:
    """Construye un DataFrame con columnas dadas y valores aleatorios."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        rng.random((n_rows, len(cols))),
        columns=cols,
    )


def _make_op_error(msg: str = "connection refused") -> OperationalError:
    """Crea una OperationalError real sin los args complejos del constructor."""
    err = OperationalError.__new__(OperationalError)
    Exception.__init__(err, msg)
    return err


def _make_full_service() -> RecommendationService:
    """RecommendationService con todos los atributos de __init__ seteados."""
    service = RecommendationService()
    service._db_host = "remote.host"
    service._db_sslmode = "require"
    return service


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


# ─── startup() ───────────────────────────────────────────────────────────────

class TestStartup:

    def test_startup_sets_model_name_from_log(self, monkeypatch):
        service = _make_full_service()
        artifacts = LoadedArtifacts(
            model=MagicMock(),
            clusters=MagicMock(),
            model_log={"model_name": "MyModel", "n_features": 4, "feature_cols": ["a", "b", "c", "d"]},
        )
        monkeypatch.setattr(service, "_download_and_load_artifacts", lambda: artifacts)
        monkeypatch.setattr(service, "_build_engine", lambda: MagicMock())

        service.startup()

        assert service.model_name == "MyModel"

    def test_startup_sets_n_features_from_log(self, monkeypatch):
        service = _make_full_service()
        artifacts = LoadedArtifacts(
            model=MagicMock(),
            clusters=MagicMock(),
            model_log={"model_name": "M", "n_features": 12, "feature_cols": [f"f{i}" for i in range(12)]},
        )
        monkeypatch.setattr(service, "_download_and_load_artifacts", lambda: artifacts)
        monkeypatch.setattr(service, "_build_engine", lambda: MagicMock())

        service.startup()

        assert service.n_features == 12

    def test_startup_sets_feature_cols_from_log(self, monkeypatch):
        service = _make_full_service()
        cols = ["col_x", "col_y", "col_z"]
        artifacts = LoadedArtifacts(
            model=MagicMock(),
            clusters=MagicMock(),
            model_log={"model_name": "M", "n_features": 3, "feature_cols": cols},
        )
        monkeypatch.setattr(service, "_download_and_load_artifacts", lambda: artifacts)
        monkeypatch.setattr(service, "_build_engine", lambda: MagicMock())

        service.startup()

        assert service.feature_cols == cols

    def test_startup_assigns_artifacts_and_engine(self, monkeypatch):
        service = _make_full_service()
        mock_engine = MagicMock()
        artifacts = LoadedArtifacts(
            model=MagicMock(),
            clusters=MagicMock(),
            model_log={"model_name": "M", "n_features": 1, "feature_cols": ["f"]},
        )
        monkeypatch.setattr(service, "_download_and_load_artifacts", lambda: artifacts)
        monkeypatch.setattr(service, "_build_engine", lambda: mock_engine)

        service.startup()

        assert service._artifacts is artifacts
        assert service.engine is mock_engine


# ─── _load_artifacts() ────────────────────────────────────────────────────────

class TestLoadArtifacts:

    def _valid_cluster_dict(self):
        """Dict con las claves requeridas por _ClusterArtifacts.from_dict."""
        return {
            "scaler_user":      MagicMock(),
            "kmeans_user":      MagicMock(),
            "user_profiles":    MagicMock(),
            "scaler_product":   MagicMock(),
            "kmeans_product":   MagicMock(),
            "product_profiles": MagicMock(),
        }

    def test_returns_loaded_artifacts_instance(self, monkeypatch):
        service = _make_full_service()
        model_log = {"model_name": "LightGBM", "n_features": 2, "feature_cols": ["a", "b"]}

        mock_model = MagicMock()
        valid_clusters = self._valid_cluster_dict()

        def fake_load(path):
            return valid_clusters if "cluster" in str(path) else mock_model

        monkeypatch.setattr("src.api.inference.joblib.load", fake_load)
        monkeypatch.setenv("USE_S3", "false")

        with patch("builtins.open", mock_open(read_data=json.dumps(model_log))):
            result = service._download_and_load_artifacts()

        assert isinstance(result, LoadedArtifacts)
        assert result.model is mock_model
        assert isinstance(result.clusters, _ClusterArtifacts)

    def test_model_log_is_parsed_correctly(self, monkeypatch):
        service = _make_full_service()
        model_log = {"model_name": "TestModel", "n_features": 5, "feature_cols": list("abcde")}
        valid_clusters = self._valid_cluster_dict()

        def fake_load(path):
            return valid_clusters if "cluster" in str(path) else MagicMock()

        monkeypatch.setattr("src.api.inference.joblib.load", fake_load)
        monkeypatch.setenv("USE_S3", "false")

        with patch("builtins.open", mock_open(read_data=json.dumps(model_log))):
            result = service._download_and_load_artifacts()

        assert result.model_log["model_name"] == "TestModel"
        assert result.model_log["n_features"] == 5

    def test_joblib_load_called_twice(self, monkeypatch):
        service = _make_full_service()
        model_log = {"feature_cols": [], "n_features": 0}
        valid_clusters = self._valid_cluster_dict()

        call_count = {"n": 0}

        def fake_load(path):
            call_count["n"] += 1
            return valid_clusters if "cluster" in str(path) else MagicMock()

        monkeypatch.setattr("src.api.inference.joblib.load", fake_load)
        monkeypatch.setenv("USE_S3", "false")

        with patch("builtins.open", mock_open(read_data=json.dumps(model_log))):
            service._download_and_load_artifacts()

        assert call_count["n"] == 2


# ─── _build_engine() ─────────────────────────────────────────────────────────

class TestBuildEngine:

    def test_raises_if_aws_host_not_set(self, monkeypatch):
        service = _make_full_service()
        monkeypatch.delenv("DB_HOST", raising=False)

        with pytest.raises(DatabaseConnectionError, match="DB_HOST"):
            service._build_engine()

    def test_local_host_lowers_sslmode_to_prefer(self, monkeypatch):
        service = _make_full_service()
        monkeypatch.setenv("DB_HOST", "localhost")
        monkeypatch.setenv("DB_SSLMODE", "require")
        monkeypatch.setenv("DB_USER", "user")
        monkeypatch.setenv("DB_PASSWORD", "pass")
        monkeypatch.setenv("DB_NAME", "db")
        monkeypatch.setenv("DB_PORT", "5432")
        monkeypatch.setattr("src.api.inference.create_engine", MagicMock(return_value=MagicMock()))

        service._build_engine()

        assert service._db_sslmode == "prefer"

    def test_127_0_0_1_lowers_sslmode_to_prefer(self, monkeypatch):
        service = _make_full_service()
        monkeypatch.setenv("DB_HOST", "127.0.0.1")
        monkeypatch.setenv("DB_SSLMODE", "verify-full")
        monkeypatch.setenv("DB_USER", "user")
        monkeypatch.setenv("DB_PASSWORD", "pass")
        monkeypatch.setenv("DB_NAME", "db")
        monkeypatch.setenv("DB_PORT", "5432")
        monkeypatch.setattr("src.api.inference.create_engine", MagicMock(return_value=MagicMock()))

        service._build_engine()

        assert service._db_sslmode == "prefer"

    def test_remote_host_keeps_require(self, monkeypatch):
        service = _make_full_service()
        monkeypatch.setenv("DB_HOST", "my-rds.amazonaws.com")
        monkeypatch.setenv("DB_SSLMODE", "require")
        monkeypatch.setenv("DB_USER", "admin")
        monkeypatch.setenv("DB_PASSWORD", "secret")
        monkeypatch.setenv("DB_NAME", "mydb")
        monkeypatch.setenv("DB_PORT", "5432")
        monkeypatch.setattr("src.api.inference.create_engine", MagicMock(return_value=MagicMock()))

        service._build_engine()

        assert service._db_sslmode == "require"

    def test_invalid_sslmode_falls_back_to_require(self, monkeypatch):
        service = _make_full_service()
        monkeypatch.setenv("DB_HOST", "my-rds.amazonaws.com")
        monkeypatch.setenv("DB_SSLMODE", "invalid_mode")
        monkeypatch.setenv("DB_USER", "admin")
        monkeypatch.setenv("DB_PASSWORD", "secret")
        monkeypatch.setenv("DB_NAME", "mydb")
        monkeypatch.setenv("DB_PORT", "5432")
        monkeypatch.setattr("src.api.inference.create_engine", MagicMock(return_value=MagicMock()))

        service._build_engine()

        assert service._db_sslmode == "require"

    def test_stores_db_host(self, monkeypatch):
        service = _make_full_service()
        monkeypatch.setenv("DB_HOST", "my-rds.amazonaws.com")
        monkeypatch.setenv("DB_SSLMODE", "require")
        monkeypatch.setenv("DB_USER", "admin")
        monkeypatch.setenv("DB_PASSWORD", "secret")
        monkeypatch.setenv("DB_NAME", "mydb")
        monkeypatch.setenv("DB_PORT", "5432")
        monkeypatch.setattr("src.api.inference.create_engine", MagicMock(return_value=MagicMock()))

        service._build_engine()

        assert service._db_host == "my-rds.amazonaws.com"

    def test_returns_engine(self, monkeypatch):
        service = _make_full_service()
        monkeypatch.setenv("DB_HOST", "my-rds.amazonaws.com")
        monkeypatch.setenv("DB_SSLMODE", "require")
        monkeypatch.setenv("DB_USER", "admin")
        monkeypatch.setenv("DB_PASSWORD", "secret")
        monkeypatch.setenv("DB_NAME", "mydb")
        monkeypatch.setenv("DB_PORT", "5432")

        mock_engine = MagicMock()
        monkeypatch.setattr("src.api.inference.create_engine", MagicMock(return_value=mock_engine))

        result = service._build_engine()

        assert result is mock_engine


# ─── _read_sql() ─────────────────────────────────────────────────────────────

class TestReadSql:

    def _make_service_with_engine(self):
        service = _make_full_service()
        service.engine = MagicMock()
        return service

    def test_returns_dataframe_on_success(self, monkeypatch):
        service = self._make_service_with_engine()
        expected_df = pd.DataFrame({"col": [1, 2, 3]})

        monkeypatch.setattr("src.api.inference.pd.read_sql", lambda sql, conn, params=None: expected_df)

        result = service._read_sql("SELECT 1", {})

        pd.testing.assert_frame_equal(result, expected_df)

    def test_operational_error_becomes_database_connection_error(self, monkeypatch):
        service = self._make_service_with_engine()
        service._db_host = "remote.host"

        # Hacer que pd.read_sql lance OperationalError
        monkeypatch.setattr(
            "src.api.inference.pd.read_sql",
            MagicMock(side_effect=_make_op_error("connection refused")),
        )

        with pytest.raises(DatabaseConnectionError):
            service._read_sql("SELECT 1", {})

    def test_error_message_includes_host(self, monkeypatch):
        service = self._make_service_with_engine()
        service._db_host = "my-special-host"
        service._db_sslmode = "require"

        monkeypatch.setattr(
            "src.api.inference.pd.read_sql",
            MagicMock(side_effect=_make_op_error()),
        )

        with pytest.raises(DatabaseConnectionError, match="my-special-host"):
            service._read_sql("SELECT 1", {})

    def test_local_host_error_includes_ssl_hint(self, monkeypatch):
        service = self._make_service_with_engine()
        service._db_host = "localhost"
        service._db_sslmode = "prefer"

        monkeypatch.setattr(
            "src.api.inference.pd.read_sql",
            MagicMock(side_effect=_make_op_error()),
        )

        with pytest.raises(DatabaseConnectionError, match="DB_SSLMODE"):
            service._read_sql("SELECT 1", {})

    def test_passes_params_to_read_sql(self, monkeypatch):
        service = self._make_service_with_engine()
        captured = {}

        def fake_read_sql(sql, conn, params=None):
            captured["params"] = params
            return pd.DataFrame()

        monkeypatch.setattr("src.api.inference.pd.read_sql", fake_read_sql)

        service._read_sql("SELECT 1 WHERE x = :val", {"val": 42})

        assert captured["params"] == {"val": 42}


# ─── _query_user_prior() ─────────────────────────────────────────────────────

class TestQueryUserPrior:

    def test_calls_read_sql_with_user_id(self, monkeypatch):
        service = _make_full_service()
        service.engine = MagicMock()

        captured = {}

        def fake_read_sql(sql, params=None):
            captured["params"] = params
            return pd.DataFrame({"user_key": [1], "product_key": [10],
                                  "order_key": [1], "order_number": [1],
                                  "days_since_prior_order": [7], "reordered": [1],
                                  "add_to_cart_order": [2], "get_eval": ["prior"]})

        monkeypatch.setattr(service, "_read_sql", fake_read_sql)

        result = service._query_user_prior(user_id=42)

        assert captured["params"] == {"user_id": 42}
        assert len(result) == 1

    def test_returns_dataframe(self, monkeypatch):
        service = _make_full_service()
        service.engine = MagicMock()

        monkeypatch.setattr(service, "_read_sql", lambda sql, params=None: pd.DataFrame({"col": [1]}))

        result = service._query_user_prior(user_id=1)

        assert isinstance(result, pd.DataFrame)


# ─── _query_user_dim_product() ───────────────────────────────────────────────

class TestQueryUserDimProduct:

    def test_calls_read_sql_with_product_keys(self, monkeypatch):
        service = _make_full_service()
        service.engine = MagicMock()

        captured = {}

        def fake_read_sql(sql, params=None):
            captured["params"] = params
            return pd.DataFrame({"product_key": [1, 2], "product_name": ["A", "B"],
                                  "department_name": ["d", "d"], "aisle_name": ["a", "a"]})

        monkeypatch.setattr(service, "_read_sql", fake_read_sql)

        service._query_user_dim_product([1, 2])

        assert captured["params"]["product_keys"] == [1, 2]


# ─── _assign_user_cluster() ──────────────────────────────────────────────────

class TestAssignUserCluster:

    def _make_user_matrix(self, user_id: int = 1) -> pd.DataFrame:
        """Matriz mínima con USER_CLUSTER_FEATURES para el usuario dado."""
        data = {"user_key": [user_id] * 3}
        for col in USER_CLUSTER_FEATURES:
            data[col] = [5.0, 6.0, 7.0]
        return pd.DataFrame(data)

    def test_unknown_user_gets_cluster_minus_one(self):
        matrix = self._make_user_matrix(user_id=1)
        user_profiles_ref = pd.DataFrame(index=[2, 3])  # user 1 no está

        clusters = _ClusterArtifacts(
            scaler_user=MagicMock(),
            kmeans_user=MagicMock(),
            user_profiles=user_profiles_ref,
            scaler_product=MagicMock(),
            kmeans_product=MagicMock(),
            product_profiles=MagicMock(),
        )
        result = RecommendationService._assign_user_cluster(
            matrix=matrix,
            user_id=1,
            clusters=clusters,
        )

        assert (result["user_cluster"] == -1).all()

    def test_known_user_gets_cluster_from_kmeans(self):
        user_id = 7
        matrix = self._make_user_matrix(user_id=user_id)
        user_profiles_ref = pd.DataFrame(
            {col: [5.0] for col in USER_CLUSTER_FEATURES},
            index=[user_id],
        )

        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.array([[0.1] * len(USER_CLUSTER_FEATURES)])
        mock_kmeans = MagicMock()
        mock_kmeans.predict.return_value = np.array([3])

        clusters = _ClusterArtifacts(
            scaler_user=mock_scaler,
            kmeans_user=mock_kmeans,
            user_profiles=user_profiles_ref,
            scaler_product=MagicMock(),
            kmeans_product=MagicMock(),
            product_profiles=MagicMock(),
        )
        result = RecommendationService._assign_user_cluster(
            matrix=matrix,
            user_id=user_id,
            clusters=clusters,
        )

        assert (result["user_cluster"] == 3).all()

    def test_known_user_calls_scaler_transform(self):
        user_id = 5
        matrix = self._make_user_matrix(user_id=user_id)
        user_profiles_ref = pd.DataFrame(
            {col: [1.0] for col in USER_CLUSTER_FEATURES},
            index=[user_id],
        )

        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.zeros((1, len(USER_CLUSTER_FEATURES)))
        mock_kmeans = MagicMock()
        mock_kmeans.predict.return_value = np.array([2])

        clusters = _ClusterArtifacts(
            scaler_user=mock_scaler,
            kmeans_user=mock_kmeans,
            user_profiles=user_profiles_ref,
            scaler_product=MagicMock(),
            kmeans_product=MagicMock(),
            product_profiles=MagicMock(),
        )
        RecommendationService._assign_user_cluster(
            matrix=matrix,
            user_id=user_id,
            clusters=clusters,
        )

        mock_scaler.transform.assert_called_once()


# ─── _assign_product_clusters() ──────────────────────────────────────────────

class TestAssignProductClusters:

    def _make_product_matrix(self, product_keys: list) -> pd.DataFrame:
        """Matriz con PRODUCT_CLUSTER_FEATURES para los productos dados."""
        rows = []
        for pk in product_keys:
            row = {"product_key": pk}
            for col in PRODUCT_CLUSTER_FEATURES:
                row[col] = 1.0
            rows.append(row)
        return pd.DataFrame(rows)

    def test_unknown_products_get_cluster_minus_one(self):
        product_keys = [10, 20, 30]
        matrix = self._make_product_matrix(product_keys)
        product_profiles_ref = pd.DataFrame(index=[99, 98])  # ninguno conocido

        mock_scaler = MagicMock()
        mock_kmeans = MagicMock()

        clusters = _ClusterArtifacts(
            scaler_user=MagicMock(),
            kmeans_user=MagicMock(),
            user_profiles=MagicMock(),
            scaler_product=mock_scaler,
            kmeans_product=mock_kmeans,
            product_profiles=product_profiles_ref,
        )
        result = RecommendationService._assign_product_clusters(
            matrix=matrix,
            clusters=clusters,
        )

        assert (result["product_cluster"] == -1).all()

    def test_known_products_get_cluster_from_kmeans(self):
        product_keys = [10, 20]
        matrix = self._make_product_matrix(product_keys)
        product_profiles_ref = pd.DataFrame(
            {col: [1.0, 1.0] for col in PRODUCT_CLUSTER_FEATURES},
            index=[10, 20],
        )

        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.zeros((2, len(PRODUCT_CLUSTER_FEATURES)))
        mock_kmeans = MagicMock()
        mock_kmeans.predict.return_value = np.array([1, 2], dtype="int8")

        clusters = _ClusterArtifacts(
            scaler_user=MagicMock(),
            kmeans_user=MagicMock(),
            user_profiles=MagicMock(),
            scaler_product=mock_scaler,
            kmeans_product=mock_kmeans,
            product_profiles=product_profiles_ref,
        )
        result = RecommendationService._assign_product_clusters(
            matrix=matrix,
            clusters=clusters,
        )

        assert "product_cluster" in result.columns
        # Los dos productos conocidos deben tener cluster 1 y 2
        product_clusters = set(result["product_cluster"].tolist())
        assert -1 not in product_clusters

    def test_adds_product_cluster_column(self):
        product_keys = [5]
        matrix = self._make_product_matrix(product_keys)
        product_profiles_ref = pd.DataFrame(index=[99])  # desconocido

        clusters = _ClusterArtifacts(
            scaler_user=MagicMock(),
            kmeans_user=MagicMock(),
            user_profiles=MagicMock(),
            scaler_product=MagicMock(),
            kmeans_product=MagicMock(),
            product_profiles=product_profiles_ref,
        )
        result = RecommendationService._assign_product_clusters(
            matrix=matrix,
            clusters=clusters,
        )

        assert "product_cluster" in result.columns


# ─── _build_online_matrix() — casos de UserNotFoundError ─────────────────────

class TestBuildOnlineMatrix:

    def test_empty_prior_raises_user_not_found(self, monkeypatch):
        service = _make_full_service()
        service.engine = MagicMock()

        monkeypatch.setattr(service, "_query_user_prior", lambda uid: pd.DataFrame())

        with pytest.raises(UserNotFoundError, match="42"):
            service._build_online_matrix(user_id=42)

    def test_user_not_found_error_message_contains_user_id(self, monkeypatch):
        service = _make_full_service()
        service.engine = MagicMock()

        monkeypatch.setattr(service, "_query_user_prior", lambda uid: pd.DataFrame())

        with pytest.raises(UserNotFoundError) as exc_info:
            service._build_online_matrix(user_id=99)

        assert "99" in str(exc_info.value)


# ─── recommend_user() ─────────────────────────────────────────────────────────

class TestRecommendUser:

    def _make_service_for_recommend(self, feature_cols: list) -> RecommendationService:
        """Service configurado con modelo mock listo para recommend_user."""
        service = _make_full_service()
        service.feature_cols = feature_cols
        service.n_features = len(feature_cols)

        mock_model = MagicMock()
        n = 5
        probas = np.array([[0.3, 0.7], [0.6, 0.4], [0.1, 0.9], [0.8, 0.2], [0.5, 0.5]])
        mock_model.predict_proba.return_value = probas[:n]

        service._artifacts = MagicMock()
        service._artifacts.model = mock_model
        service.engine = MagicMock()
        return service

    def _make_online_matrix(self, feature_cols: list) -> pd.DataFrame:
        """Matriz online sintética con product_key y feature cols."""
        rng = np.random.default_rng(42)
        n = 5
        data = {"product_key": [10, 20, 30, 40, 50]}
        for col in feature_cols:
            data[col] = rng.random(n).tolist()
        return pd.DataFrame(data)

    def _make_product_names_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "product_key": [10, 20, 30, 40, 50],
            "product_name": ["Apple", "Banana", "Cherry", "Date", "Elderberry"],
        })

    def test_returns_list_of_dicts(self, monkeypatch):
        cols = ["feat_a", "feat_b"]
        service = self._make_service_for_recommend(cols)
        matrix = self._make_online_matrix(cols)

        monkeypatch.setattr(service, "_build_online_matrix", lambda uid: matrix)
        monkeypatch.setattr(service, "_align_and_validate", lambda m: m[cols])
        monkeypatch.setattr(service, "_query_user_dim_product", lambda pks: self._make_product_names_df())
        service._query_user_order_count = MagicMock(return_value=10)

        result, _ = service.recommend_user(user_id=1, top_k=10)

        assert isinstance(result, list)
        assert all(isinstance(r, dict) for r in result)

    def test_result_has_required_keys(self, monkeypatch):
        cols = ["feat_a", "feat_b"]
        service = self._make_service_for_recommend(cols)
        matrix = self._make_online_matrix(cols)

        monkeypatch.setattr(service, "_build_online_matrix", lambda uid: matrix)
        monkeypatch.setattr(service, "_align_and_validate", lambda m: m[cols])
        monkeypatch.setattr(service, "_query_user_dim_product", lambda pks: self._make_product_names_df())
        service._query_user_order_count = MagicMock(return_value=10)

        result, _ = service.recommend_user(user_id=1, top_k=10)

        for item in result:
            assert "product_key" in item
            assert "probability" in item
            assert "product_name" in item

    def test_top_k_limits_results(self, monkeypatch):
        cols = ["feat_a", "feat_b"]
        service = self._make_service_for_recommend(cols)
        matrix = self._make_online_matrix(cols)

        monkeypatch.setattr(service, "_build_online_matrix", lambda uid: matrix)
        monkeypatch.setattr(service, "_align_and_validate", lambda m: m[cols])
        monkeypatch.setattr(service, "_query_user_dim_product", lambda pks: self._make_product_names_df())
        service._query_user_order_count = MagicMock(return_value=10)

        result, _ = service.recommend_user(user_id=1, top_k=3)

        assert len(result) == 3

    def test_results_sorted_by_probability_descending(self, monkeypatch):
        cols = ["feat_a"]
        service = self._make_service_for_recommend(cols)
        matrix = self._make_online_matrix(cols)

        monkeypatch.setattr(service, "_build_online_matrix", lambda uid: matrix)
        monkeypatch.setattr(service, "_align_and_validate", lambda m: m[cols])
        monkeypatch.setattr(service, "_query_user_dim_product", lambda pks: self._make_product_names_df())
        service._query_user_order_count = MagicMock(return_value=10)

        result, _ = service.recommend_user(user_id=1, top_k=10)

        probabilities = [r["probability"] for r in result]
        assert probabilities == sorted(probabilities, reverse=True)

    def test_product_key_is_int(self, monkeypatch):
        cols = ["feat_a"]
        service = self._make_service_for_recommend(cols)
        matrix = self._make_online_matrix(cols)

        monkeypatch.setattr(service, "_build_online_matrix", lambda uid: matrix)
        monkeypatch.setattr(service, "_align_and_validate", lambda m: m[cols])
        monkeypatch.setattr(service, "_query_user_dim_product", lambda pks: self._make_product_names_df())
        service._query_user_order_count = MagicMock(return_value=10)

        result, _ = service.recommend_user(user_id=1, top_k=5)

        assert all(isinstance(r["product_key"], int) for r in result)

    def test_probability_is_float(self, monkeypatch):
        cols = ["feat_a"]
        service = self._make_service_for_recommend(cols)
        matrix = self._make_online_matrix(cols)

        monkeypatch.setattr(service, "_build_online_matrix", lambda uid: matrix)
        monkeypatch.setattr(service, "_align_and_validate", lambda m: m[cols])
        monkeypatch.setattr(service, "_query_user_dim_product", lambda pks: self._make_product_names_df())
        service._query_user_order_count = MagicMock(return_value=10)

        result, _ = service.recommend_user(user_id=1, top_k=5)

        assert all(isinstance(r["probability"], float) for r in result)

    def test_missing_product_name_returns_none(self, monkeypatch):
        cols = ["feat_a"]
        service = self._make_service_for_recommend(cols)
        matrix = self._make_online_matrix(cols)

        # No product names → merge result tiene product_name = NaN
        empty_names = pd.DataFrame({"product_key": [], "product_name": []})

        monkeypatch.setattr(service, "_build_online_matrix", lambda uid: matrix)
        monkeypatch.setattr(service, "_align_and_validate", lambda m: m[cols])
        monkeypatch.setattr(service, "_query_user_dim_product", lambda pks: empty_names)
        service._query_user_order_count = MagicMock(return_value=10)

        result, _ = service.recommend_user(user_id=1, top_k=5)

        assert all(r["product_name"] is None for r in result)

    def test_user_not_found_error_propagates(self, monkeypatch):
        cols = ["feat_a"]
        service = self._make_service_for_recommend(cols)

        monkeypatch.setattr(
            service,
            "_build_online_matrix",
            MagicMock(side_effect=UserNotFoundError("user 99 not found")),
        )
        service._query_user_order_count = MagicMock(return_value=10)

        with pytest.raises(UserNotFoundError):
            service.recommend_user(user_id=99, top_k=10)


# ─── Cold-start ───────────────────────────────────────────────────────────────

class TestColdStart:

    def test_cold_start_triggered_when_few_orders(self):
        """Cuando hay menos de MIN_ORDERS_FOR_MODEL órdenes, se usa cold-start y no el pipeline."""
        service = _make_full_service()
        service._query_user_order_count = MagicMock(return_value=3)  # 3 < 5
        service._cold_start_top_products = MagicMock(return_value=[
            {"product_key": 1, "product_name": "Apple", "probability": 0.5}
        ])
        service._build_online_matrix = MagicMock()

        result, cold_start = service.recommend_user(user_id=1, top_k=5)

        service._cold_start_top_products.assert_called_once_with(1, 5)
        service._build_online_matrix.assert_not_called()
        assert cold_start is True
        assert result == [{"product_key": 1, "product_name": "Apple", "probability": 0.5}]

    def test_cold_start_not_triggered_when_enough_orders(self):
        """Cuando hay suficientes órdenes, se usa el pipeline completo y no cold-start."""
        cols = ["feat_a"]
        service = _make_full_service()
        service.feature_cols = cols
        service.n_features = len(cols)

        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        service._artifacts = MagicMock()
        service._artifacts.model = mock_model
        service.engine = MagicMock()

        service._query_user_order_count = MagicMock(return_value=10)  # 10 >= 5
        service._cold_start_top_products = MagicMock()

        matrix = pd.DataFrame({"product_key": [1], "feat_a": [0.5]})
        service._build_online_matrix = MagicMock(return_value=matrix)
        service._align_and_validate = MagicMock(return_value=matrix[cols])
        service._query_user_dim_product = MagicMock(
            return_value=pd.DataFrame({"product_key": [1], "product_name": ["Apple"]})
        )

        service.recommend_user(user_id=1, top_k=10)

        service._build_online_matrix.assert_called_once_with(1)
        service._cold_start_top_products.assert_not_called()

    def test_query_user_order_count_calls_read_sql(self):
        """_query_user_order_count delega en _read_sql pasando el user_id correcto."""
        service = _make_full_service()
        mock_df = pd.DataFrame({"n_orders": [7]})
        service._read_sql = MagicMock(return_value=mock_df)

        result = service._query_user_order_count(user_id=42)

        assert result == 7
        service._read_sql.assert_called_once()
        params = service._read_sql.call_args[0][1]
        assert params["user_id"] == 42

    def test_cold_start_returns_empty_when_no_products(self):
        """Cuando cold-start no encuentra productos, recommend_user devuelve lista vacía."""
        service = _make_full_service()
        service._query_user_order_count = MagicMock(return_value=3)  # 3 < 5
        service._cold_start_top_products = MagicMock(return_value=[])

        result, _ = service.recommend_user(user_id=1, top_k=5)

        assert result == []


# ─── TestLoadModelFromS3 ──────────────────────────────────────────────────────

class TestLoadModelFromS3:

    def test_loads_model_from_s3(self):
        """Cuando USE_S3=true, _download_and_load_artifacts descarga model.pkl desde S3 con boto3."""
        mock_s3 = MagicMock()
        model_log_json = json.dumps({"model_name": "lgbm", "n_features": 2, "feature_cols": ["a", "b"]})

        valid_clusters = {
            "scaler_user": MagicMock(), "kmeans_user": MagicMock(),
            "user_profiles": MagicMock(), "scaler_product": MagicMock(),
            "kmeans_product": MagicMock(), "product_profiles": MagicMock(),
        }

        def fake_joblib_load(path):
            return valid_clusters if "cluster" in str(path) else MagicMock()

        with patch.dict("os.environ", {"USE_S3": "true"}), \
             patch("src.api.inference.boto3.client", return_value=mock_s3), \
             patch("src.api.inference.joblib.load", side_effect=fake_joblib_load), \
             patch("builtins.open", mock_open(read_data=model_log_json)):
            service = _make_full_service()
            artifacts = service._download_and_load_artifacts()

        assert mock_s3.download_file.call_count == 3
        first_call_args = mock_s3.download_file.call_args_list[0][0]
        assert "model.pkl" in first_call_args[1]
        assert artifacts is not None

    def test_raises_if_s3_download_fails(self):
        """Si download_file lanza una excepción, _download_and_load_artifacts la propaga."""
        mock_s3 = MagicMock()
        mock_s3.download_file.side_effect = OSError("S3 connection failed")

        with patch.dict("os.environ", {"USE_S3": "true"}), \
             patch("src.api.inference.boto3.client", return_value=mock_s3):
            service = _make_full_service()
            with pytest.raises(OSError):
                service._download_and_load_artifacts()


# ─── TestColdStartTopProducts ─────────────────────────────────────────────────

class TestColdStartTopProducts:

    def _make_top_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "product_key": [10, 20, 30],
            "purchase_count": [5, 3, 2],
            "probability": [0.5, 0.3, 0.2],
        })

    def _make_names_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "product_key": [10, 20, 30],
            "product_name": ["Apple", "Banana", "Cherry"],
        })

    def test_returns_top_k_products(self):
        """_cold_start_top_products devuelve lista de dicts con las claves requeridas."""
        service = _make_full_service()
        service._read_sql = MagicMock(return_value=self._make_top_df())
        service._query_user_dim_product = MagicMock(return_value=self._make_names_df())

        result = service._cold_start_top_products(user_id=1, top_k=3)

        assert isinstance(result, list)
        assert len(result) == 3
        assert all({"product_key", "product_name", "probability"} <= r.keys() for r in result)
        assert all(isinstance(r["product_key"], int) for r in result)
        assert all(isinstance(r["probability"], float) for r in result)

    def test_returns_empty_when_no_products(self):
        """Cuando _read_sql devuelve DataFrame vacío, _cold_start_top_products devuelve []."""
        service = _make_full_service()
        service._read_sql = MagicMock(return_value=pd.DataFrame())

        result = service._cold_start_top_products(user_id=1, top_k=5)

        assert result == []

    def test_enriches_with_product_name(self):
        """_cold_start_top_products llama a _query_user_dim_product con los product_keys del top."""
        service = _make_full_service()
        service._read_sql = MagicMock(return_value=self._make_top_df())
        service._query_user_dim_product = MagicMock(return_value=self._make_names_df())

        result = service._cold_start_top_products(user_id=7, top_k=3)

        service._query_user_dim_product.assert_called_once_with([10, 20, 30])
        assert result[0]["product_name"] == "Apple"
        assert result[1]["product_name"] == "Banana"


# ─── _ClusterArtifacts.from_dict() — claves faltantes ────────────────────────

class TestClusterArtifactsFromDict:

    def test_raises_key_error_when_dict_is_empty(self):
        """from_dict con dict vacío debe lanzar KeyError."""
        with pytest.raises(KeyError):
            _ClusterArtifacts.from_dict({})

    def test_error_message_mentions_missing_keys(self):
        """KeyError debe mencionar las claves requeridas."""
        with pytest.raises(KeyError, match="requeridas"):
            _ClusterArtifacts.from_dict({})

    def test_raises_when_partial_keys_provided(self):
        """Falta al menos una clave requerida → KeyError."""
        partial = {"scaler_user": MagicMock(), "kmeans_user": MagicMock()}
        with pytest.raises(KeyError):
            _ClusterArtifacts.from_dict(partial)


# ─── _download_and_load_artifacts() — errores S3 ─────────────────────────────

class TestDownloadArtifactsS3Errors:

    def test_client_error_is_propagated(self):
        """ClientError durante la descarga S3 debe propagarse al caller."""
        from botocore.exceptions import ClientError

        mock_s3 = MagicMock()
        mock_s3.download_file.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "The specified key does not exist."}},
            "GetObject",
        )
        with patch.dict("os.environ", {"USE_S3": "true"}), \
             patch("src.api.inference.boto3.client", return_value=mock_s3):
            service = _make_full_service()
            with pytest.raises(ClientError):
                service._download_and_load_artifacts()

    def test_botocore_error_is_propagated(self):
        """BotoCoreError (red / credentials) durante la descarga S3 debe propagarse."""
        from botocore.exceptions import NoCredentialsError

        mock_s3 = MagicMock()
        mock_s3.download_file.side_effect = NoCredentialsError()
        with patch.dict("os.environ", {"USE_S3": "true"}), \
             patch("src.api.inference.boto3.client", return_value=mock_s3):
            service = _make_full_service()
            with pytest.raises(Exception):
                service._download_and_load_artifacts()


# ─── _build_engine() — variable de entorno faltante ──────────────────────────

class TestBuildEngineMissingEnvVars:

    def test_raises_database_error_when_db_user_missing(self, monkeypatch):
        """Falta DB_USER → DatabaseConnectionError con mensaje descriptivo."""
        service = _make_full_service()
        monkeypatch.setenv("DB_HOST", "my-rds.amazonaws.com")
        monkeypatch.setenv("DB_SSLMODE", "require")
        monkeypatch.delenv("DB_USER", raising=False)
        monkeypatch.delenv("DB_PASSWORD", raising=False)
        monkeypatch.delenv("DB_NAME", raising=False)

        with pytest.raises(DatabaseConnectionError, match="faltante"):
            service._build_engine()


# ─── _query_product_features() ───────────────────────────────────────────────

class TestQueryProductFeatures:

    def test_calls_read_sql_with_product_keys(self, monkeypatch):
        service = _make_full_service()
        service.engine = MagicMock()
        captured = {}
        expected_df = pd.DataFrame({
            "product_key": [1, 2],
            "product_total_purchases": [100, 200],
        })

        def fake_read_sql(sql, params=None):
            captured["params"] = params
            return expected_df

        monkeypatch.setattr(service, "_read_sql", fake_read_sql)

        result = service._query_product_features([1, 2])

        assert captured["params"]["product_keys"] == [1, 2]
        assert isinstance(result, pd.DataFrame)

    def test_returns_dataframe(self, monkeypatch):
        service = _make_full_service()
        service.engine = MagicMock()
        monkeypatch.setattr(service, "_read_sql", lambda sql, params=None: pd.DataFrame())

        result = service._query_product_features([10, 20])

        assert isinstance(result, pd.DataFrame)


# ─── _get_global_top_products() ──────────────────────────────────────────────

class TestGetGlobalTopProducts:

    def _make_top_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "product_key": [1, 2, 3],
            "purchase_count": [100, 50, 25],
            "probability": [0.5, 0.25, 0.125],
        })

    def _make_names_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "product_key": [1, 2, 3],
            "product_name": ["Apple", "Banana", "Cherry"],
        })

    def test_returns_list_when_products_found(self):
        service = _make_full_service()
        service._read_sql = MagicMock(return_value=self._make_top_df())
        service._query_user_dim_product = MagicMock(return_value=self._make_names_df())

        result = service._get_global_top_products(top_k=3)

        assert isinstance(result, list)
        assert len(result) == 3
        assert all({"product_key", "product_name", "probability"} <= r.keys() for r in result)
        assert all(isinstance(r["product_key"], int) for r in result)
        assert all(isinstance(r["probability"], float) for r in result)

    def test_returns_empty_when_no_products(self):
        service = _make_full_service()
        service._read_sql = MagicMock(return_value=pd.DataFrame())

        result = service._get_global_top_products(top_k=10)

        assert result == []

    def test_calls_query_dim_product_for_names(self):
        service = _make_full_service()
        service._read_sql = MagicMock(return_value=self._make_top_df())
        service._query_user_dim_product = MagicMock(return_value=self._make_names_df())

        service._get_global_top_products(top_k=3)

        service._query_user_dim_product.assert_called_once_with([1, 2, 3])


# ─── recommend_user() — n_orders == 0 ────────────────────────────────────────

class TestRecommendUserZeroOrders:

    def test_returns_global_top_when_user_not_in_dim_user(self):
        """n_orders==0 y usuario NO existe en dim_user → _get_global_top_products, cold_start=True."""
        service = _make_full_service()
        service._query_user_order_count = MagicMock(return_value=0)
        # dim_user check returns empty → user does not exist
        service._read_sql = MagicMock(return_value=pd.DataFrame())
        mock_top = [{"product_key": 1, "product_name": "A", "probability": 0.5}]
        service._get_global_top_products = MagicMock(return_value=mock_top)

        result, cold_start = service.recommend_user(user_id=999, top_k=5)

        service._get_global_top_products.assert_called_once_with(5)
        assert cold_start is True
        assert result == mock_top

    def test_raises_user_not_found_when_user_in_dim_user_but_no_orders(self):
        """n_orders==0 y usuario SÍ existe en dim_user → UserNotFoundError."""
        service = _make_full_service()
        service._query_user_order_count = MagicMock(return_value=0)
        # dim_user check returns a row → user exists
        service._read_sql = MagicMock(return_value=pd.DataFrame({"1": [1]}))

        with pytest.raises(UserNotFoundError):
            service.recommend_user(user_id=42, top_k=5)
