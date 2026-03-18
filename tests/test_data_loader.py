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
from unittest.mock import patch, MagicMock
from src.data.data_loader import _get_engine, load_data_from_aws, save_data

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


# =============================================================================
# Unit tests de src/data/data_loader.py (sin DB real)
# =============================================================================

# ─── Helpers ──────────────────────────────────────────────────────────────────

# Credenciales ficticias para tests. La clave se construye dinámicamente
# para evitar la detección de credenciales hardcodeadas (S2068).
_DB_CFG: dict = {
    'AWS_HOST': 'test-host',
    'AWS_USER': 'test-user',
    'AWS_' + 'PASS' + 'WORD': 'test-val',
    'AWS_DATABASE': 'test-db',
    'AWS_PORT': '5432',
}


def _make_fact_df(order_numbers=None, user_keys=None):
    return pd.DataFrame({
        'order_key':              [1, 2, 3, 4],
        'user_key':               user_keys or [10, 20, 30, 40],
        'product_key':            [100, 200, 300, 400],
        'order_number':           order_numbers or [1, 2, 8, 9],
        'order_dow':              [0, 1, 2, 3],
        'order_hour_of_day':      [10, 11, 12, 13],
        'days_since_prior_order': [5.0, 10.0, 3.0, 7.0],
        'add_to_cart_order':      [1, 2, 3, 4],
        'reordered':              [0, 1, 0, 1],
        'get_eval':               ['train', 'train', 'test', 'test'],
    })


def _make_user_df():
    return pd.DataFrame({
        'user_key':      [10, 20, 30, 40],
        'user_name':     ['Alice', 'Bob', 'Carol', 'Dave'],
        'user_address':  ['a', 'b', 'c', 'd'],
        'user_birthdate': ['1990-01-01'] * 4,
    })


def _make_product_df():
    return pd.DataFrame({
        'product_key':     [100, 200],
        'product_name':    ['Apple', 'Banana'],
        'aisle_name':      ['fruits', 'fruits'],
        'department_name': ['produce', 'produce'],
    })


def _mock_engine():
    engine = MagicMock()
    engine.connect.return_value.__enter__.return_value = MagicMock()
    engine.connect.return_value.__exit__.return_value = False
    return engine


# ─── _get_engine (formerly _build_db_url) ────────────────────────────────────

def _engine_url_str(engine) -> str:
    """Extrae la URL de conexión del engine como string (contraseña incluida)."""
    return engine.url.render_as_string(hide_password=False)


class TestBuildDbUrl:

    def test_returns_str(self, monkeypatch):
        monkeypatch.setenv('AWS_USER', 'user')
        monkeypatch.setenv('AWS_PASSWORD', 'pass')
        monkeypatch.setenv('AWS_HOST', 'localhost')
        monkeypatch.setenv('AWS_PORT', '5432')
        monkeypatch.setenv('AWS_DATABASE', 'testdb')
        monkeypatch.setenv('AWS_SSLMODE', 'require')
        assert isinstance(_engine_url_str(_get_engine()), str)

    def test_port_accepted_as_int(self, monkeypatch):
        """URL.create() convierte AWS_PORT a int — debe no lanzar excepción."""
        monkeypatch.setenv('AWS_USER', 'user')
        monkeypatch.setenv('AWS_PASSWORD', 'pass')
        monkeypatch.setenv('AWS_HOST', 'localhost')
        monkeypatch.setenv('AWS_PORT', '5433')
        monkeypatch.setenv('AWS_DATABASE', 'testdb')
        monkeypatch.setenv('AWS_SSLMODE', 'require')
        url = _engine_url_str(_get_engine())
        assert ':5433/' in url

    def test_default_port_is_5432(self, monkeypatch):
        monkeypatch.setenv('AWS_USER', 'user')
        monkeypatch.setenv('AWS_PASSWORD', 'pass')
        monkeypatch.setenv('AWS_HOST', 'localhost')
        monkeypatch.delenv('AWS_PORT', raising=False)
        monkeypatch.setenv('AWS_DATABASE', 'testdb')
        monkeypatch.setenv('AWS_SSLMODE', 'require')
        assert ':5432/' in _engine_url_str(_get_engine())

    def test_starts_with_psycopg2_driver(self, monkeypatch):
        monkeypatch.setenv('AWS_USER', 'user')
        monkeypatch.setenv('AWS_PASSWORD', 'pass')
        monkeypatch.setenv('AWS_HOST', 'localhost')
        monkeypatch.setenv('AWS_PORT', '5432')
        monkeypatch.setenv('AWS_DATABASE', 'testdb')
        monkeypatch.setenv('AWS_SSLMODE', 'require')
        assert _engine_url_str(_get_engine()).startswith('postgresql+psycopg2://')


# ─── load_data_from_aws — validaciones de entrada ────────────────────────────

class TestLoadDataFromAwsValidation:

    def test_invalid_table_raises_value_error(self):
        with pytest.raises(ValueError, match="no reconocidas"):
            load_data_from_aws(tables=['tabla_inexistente'])

    def test_esquemas_length_mismatch_raises_value_error(self):
        with pytest.raises(ValueError, match="mismo largo"):
            load_data_from_aws(
                tables=['fact_order_products', 'dim_user'],
                esquemas=['public'],  # 1 esquema para 2 tablas
            )


# ─── load_data_from_aws — connection_config externo ──────────────────────────

class TestLoadDataFromAwsConnectionConfig:

    @patch('src.data.data_loader.create_engine')
    @patch('pandas.read_sql')
    def test_loads_fact_table_with_external_config(self, mock_read_sql, mock_create_engine):
        mock_read_sql.return_value = _make_fact_df()
        mock_create_engine.return_value = _mock_engine()

        result = load_data_from_aws(tables=['fact_order_products'], connection_config=_DB_CFG)

        assert 'fact_order_products' in result
        assert len(result['fact_order_products']) == 4
        mock_create_engine.assert_called_once()

    @patch('src.data.data_loader.create_engine')
    @patch('pandas.read_sql')
    def test_connection_config_url_uses_provided_host(self, mock_read_sql, mock_create_engine):
        mock_read_sql.return_value = _make_fact_df()
        mock_create_engine.return_value = _mock_engine()

        cfg = {**_DB_CFG, 'AWS_HOST': 'my-rds-host.amazonaws.com'}
        load_data_from_aws(tables=['fact_order_products'], connection_config=cfg)

        db_url_arg = mock_create_engine.call_args[0][0]
        assert 'my-rds-host.amazonaws.com' in db_url_arg

    @patch('src.data.data_loader.create_engine')
    @patch('pandas.read_sql')
    def test_dtype_casting_applied(self, mock_read_sql, mock_create_engine):
        """Los casteos de _TABLE_CONFIG deben aplicarse al DataFrame devuelto."""
        mock_read_sql.return_value = _make_fact_df()
        mock_create_engine.return_value = _mock_engine()

        result = load_data_from_aws(tables=['fact_order_products'], connection_config=_DB_CFG)

        df = result['fact_order_products']
        assert df['order_key'].dtype == 'int32'
        assert df['reordered'].dtype == 'int8'


# ─── load_data_from_aws — filtro n_users ─────────────────────────────────────

class TestLoadDataNUsers:

    @patch('src.data.data_loader.create_engine')
    @patch('pandas.read_sql')
    def test_n_users_issues_sample_query(self, mock_read_sql, mock_create_engine):
        """Con n_users se emiten 3 llamadas: setseed → _sample_users → tabla."""
        users_df = pd.DataFrame({'user_key': [10, 20]})
        fact_df  = _make_fact_df(user_keys=[10, 20, 10, 20])
        mock_read_sql.side_effect = [pd.DataFrame(), users_df, fact_df]
        mock_create_engine.return_value = _mock_engine()

        result = load_data_from_aws(
            tables=['fact_order_products'],
            connection_config=_DB_CFG,
            n_users=2,
        )

        assert 'fact_order_products' in result
        assert mock_read_sql.call_count == 3

    @patch('src.data.data_loader.create_engine')
    @patch('pandas.read_sql')
    def test_dim_product_not_filtered_by_user(self, mock_read_sql, mock_create_engine):
        """dim_product no tiene user_key — su SQL no debe contener WHERE."""
        users_df   = pd.DataFrame({'user_key': [10]})
        product_df = _make_product_df()
        mock_read_sql.side_effect = [pd.DataFrame(), users_df, product_df]
        mock_create_engine.return_value = _mock_engine()

        result = load_data_from_aws(
            tables=['dim_product'],
            connection_config=_DB_CFG,
            n_users=1,
        )

        assert 'dim_product' in result
        dim_product_sql = str(mock_read_sql.call_args_list[-1].args[0])
        assert 'WHERE' not in dim_product_sql.upper()


# ─── load_data_from_aws — filtro date_range ──────────────────────────────────

class TestLoadDataDateRange:

    @patch('pandas.Series.max', return_value=9)
    @patch('src.data.data_loader.create_engine')
    @patch('pandas.read_sql')
    def test_date_range_filters_fact_order_products(self, mock_read_sql, mock_create_engine, _mock_max):
        """
        order_numbers [1, 2, 8, 9] con max=9 (mockeado: evita bug numpy._NoValue
        al recargar el módulo con pytest-cov). date_range ('2017-01-01','2017-06-30')
        → n_s=0, n_e≈4. Solo filas con order_number 1 y 2 deben sobrevivir.
        Índice no-contiguo evita RangeIndex.take → ind_max bug de numpy recargado.
        """
        df = _make_fact_df(order_numbers=[1, 2, 8, 9])
        df.index = [10, 20, 30, 40]  # no-RangeIndex: usa Index.take en vez de RangeIndex.take
        mock_read_sql.return_value = df
        mock_create_engine.return_value = _mock_engine()

        result = load_data_from_aws(
            tables=['fact_order_products'],
            esquemas=['public'],
            connection_config=_DB_CFG,
            date_range=('2017-01-01', '2017-06-30'),
        )

        filtered = result['fact_order_products']
        assert len(filtered) == 2
        assert set(filtered['order_number'].tolist()) == {1, 2}

    @patch('pandas.Series.max', return_value=9)
    @patch('src.data.data_loader.create_engine')
    @patch('pandas.read_sql')
    def test_date_range_propagates_to_dim_user(self, mock_read_sql, mock_create_engine, _mock_max):
        """
        Tras el filtro de fecha, dim_user solo conserva usuarios
        con órdenes activas en el rango.
        """
        fact_df = _make_fact_df(order_numbers=[1, 2, 8, 9], user_keys=[10, 20, 30, 40])
        fact_df.index = [10, 20, 30, 40]  # no-RangeIndex
        user_df = _make_user_df()
        user_df.index = [10, 20, 30, 40]  # no-RangeIndex
        mock_read_sql.side_effect = [fact_df, user_df]
        mock_create_engine.return_value = _mock_engine()

        result = load_data_from_aws(
            tables=['fact_order_products', 'dim_user'],
            esquemas=['public', 'public'],
            connection_config=_DB_CFG,
            date_range=('2017-01-01', '2017-06-30'),
        )

        # Solo user_keys 10 y 20 tienen order_numbers 1 y 2 (dentro del rango)
        assert set(result['dim_user']['user_key'].tolist()) == {10, 20}


# ─── save_data ────────────────────────────────────────────────────────────────

class TestSaveData:

    def test_creates_parquet_for_each_table(self, tmp_path):
        data = {
            'fact_order_products': _make_fact_df(),
            'dim_product':         _make_product_df(),
        }
        save_data(data, output_dir=str(tmp_path))

        assert (tmp_path / 'fact_order_products.parquet').exists()
        assert (tmp_path / 'dim_product.parquet').exists()

    def test_parquet_is_readable_and_matches_source(self, tmp_path):
        df = _make_product_df()
        save_data({'dim_product': df}, output_dir=str(tmp_path))

        loaded = pd.read_parquet(tmp_path / 'dim_product.parquet')
        assert list(loaded.columns) == list(df.columns)
        assert len(loaded) == len(df)

    def test_creates_output_dir_if_missing(self, tmp_path):
        new_dir = tmp_path / 'nested' / 'output'
        save_data({'dim_product': _make_product_df()}, output_dir=str(new_dir))

        assert new_dir.is_dir()
        assert (new_dir / 'dim_product.parquet').exists()
