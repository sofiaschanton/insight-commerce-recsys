"""
tests/test_etl_dimensional.py
==============================
Tests unitarios para src/data/etl_dimensional.py

Estrategia de cobertura (objetivo: ≥70 %):
  - __init__          → test directo, sin mocks
  - connect()         → patch psycopg2.connect — éxito y falla
  - close()           → mock connections — con y sin conexiones abiertas
  - transfer_data()   → mock local_conn / aws_conn — chunk único, vacío, error, multi-chunk
  - generate_report() → lógica pura sobre report_stats — sin DB, sin archivos
  - populate_dim_user() → patch psycopg2.connect — sin usuarios y con usuarios
  - run_pipeline()    → patch métodos internos — aborto por falla de conexión y ejecución normal

Sin DB real. Sin archivos reales.
psycopg2 y faker ya están stubeados globalmente en conftest.py.
"""

import logging
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from src.data.etl_dimensional import DimensionalETL


# ── Fixtures de configuración ──────────────────────────────────────────────────
LOCAL_CFG = {
    "host": "localhost",
    "database": "test_db",
    "user": "user",
    "password": "pass",
    "port": "5432",
}
AWS_CFG = {
    "host": "aws-host",
    "database": "aws_db",
    "user": "user",
    "password": "pass",
    "port": "5432",
    "sslmode": "require",
}


# ── __init__ ───────────────────────────────────────────────────────────────────
class TestInit:
    def test_attributes_are_set_correctly(self):
        etl = DimensionalETL(LOCAL_CFG, AWS_CFG)

        assert etl.local_config is LOCAL_CFG
        assert etl.aws_config is AWS_CFG
        assert etl.local_conn is None
        assert etl.aws_conn is None
        assert isinstance(etl.batch_id, int)
        assert etl.report_stats == {}
        assert etl.pipeline_start_time is None

    def test_batch_id_is_unix_timestamp(self):
        before = int(datetime.now().timestamp())
        etl = DimensionalETL(LOCAL_CFG, AWS_CFG)
        after = int(datetime.now().timestamp())

        assert before <= etl.batch_id <= after


# ── connect() ─────────────────────────────────────────────────────────────────
class TestConnect:
    def test_returns_true_and_assigns_connections_on_success(self):
        etl = DimensionalETL(LOCAL_CFG, AWS_CFG)
        mock_conn = MagicMock()

        with patch("src.data.etl_dimensional.psycopg2.connect", return_value=mock_conn):
            result = etl.connect()

        assert result is True
        assert etl.local_conn is mock_conn
        assert etl.aws_conn is mock_conn

    def test_returns_false_on_exception(self):
        etl = DimensionalETL(LOCAL_CFG, AWS_CFG)

        with patch(
            "src.data.etl_dimensional.psycopg2.connect",
            side_effect=Exception("connection refused"),
        ):
            result = etl.connect()

        assert result is False
        assert etl.local_conn is None


# ── close() ───────────────────────────────────────────────────────────────────
class TestClose:
    def test_closes_both_open_connections(self):
        etl = DimensionalETL(LOCAL_CFG, AWS_CFG)
        etl.local_conn = MagicMock()
        etl.aws_conn = MagicMock()

        etl.close()

        etl.local_conn.close.assert_called_once()
        etl.aws_conn.close.assert_called_once()

    def test_does_not_raise_when_connections_are_none(self):
        etl = DimensionalETL(LOCAL_CFG, AWS_CFG)
        # Both remain None from __init__
        etl.close()  # must not raise

    def test_closes_partial_connections(self):
        """Solo local_conn está abierta; aws_conn es None."""
        etl = DimensionalETL(LOCAL_CFG, AWS_CFG)
        etl.local_conn = MagicMock()

        etl.close()

        etl.local_conn.close.assert_called_once()


# ── transfer_data() ────────────────────────────────────────────────────────────
class TestTransferData:
    def _etl_with_mock_conns(self):
        etl = DimensionalETL(LOCAL_CFG, AWS_CFG)
        etl.local_conn = MagicMock()
        etl.aws_conn = MagicMock()
        return etl

    def test_single_chunk_records_are_counted(self):
        etl = self._etl_with_mock_conns()

        local_cursor = MagicMock()
        local_cursor.fetchmany.side_effect = [[(1, "a"), (2, "b")], []]
        etl.local_conn.cursor.return_value = local_cursor
        etl.aws_conn.cursor.return_value = MagicMock()

        with patch("src.data.etl_dimensional.execute_values"):
            etl.transfer_data("SELECT 1", "dim_user", "INSERT INTO dim_user VALUES %s")

        stats = etl.report_stats["dim_user"]
        assert stats["estado"] == "Completado"
        assert stats["filas_procesadas"] == 2
        assert stats["error"] == "Ninguno"
        assert "duracion" in stats

    def test_empty_result_set(self):
        etl = self._etl_with_mock_conns()

        local_cursor = MagicMock()
        local_cursor.fetchmany.return_value = []
        etl.local_conn.cursor.return_value = local_cursor
        etl.aws_conn.cursor.return_value = MagicMock()

        with patch("src.data.etl_dimensional.execute_values"):
            etl.transfer_data("SELECT 1", "dim_product", "INSERT INTO dim_product VALUES %s")

        assert etl.report_stats["dim_product"]["filas_procesadas"] == 0
        assert etl.report_stats["dim_product"]["estado"] == "Completado"

    def test_exception_triggers_rollback_and_marks_failed(self):
        etl = self._etl_with_mock_conns()

        local_cursor = MagicMock()
        local_cursor.fetchmany.side_effect = Exception("DB timeout")
        etl.local_conn.cursor.return_value = local_cursor
        etl.aws_conn.cursor.return_value = MagicMock()

        etl.transfer_data("SELECT 1", "fact_order_products", "INSERT INTO fact VALUES %s")

        stats = etl.report_stats["fact_order_products"]
        assert stats["estado"] == "Fallido"
        assert "DB timeout" in stats["error"]
        etl.aws_conn.rollback.assert_called_once()

    def test_multiple_chunks_accumulate_count(self):
        etl = self._etl_with_mock_conns()

        chunk = [(i,) for i in range(5)]
        local_cursor = MagicMock()
        local_cursor.fetchmany.side_effect = [chunk, chunk, []]
        etl.local_conn.cursor.return_value = local_cursor
        etl.aws_conn.cursor.return_value = MagicMock()

        with patch("src.data.etl_dimensional.execute_values"):
            etl.transfer_data("SELECT 1", "dim_user", "INSERT VALUES %s", chunk_size=5)

        assert etl.report_stats["dim_user"]["filas_procesadas"] == 10

    def test_aws_commit_called_per_chunk(self):
        etl = self._etl_with_mock_conns()

        chunk = [(1,), (2,)]
        local_cursor = MagicMock()
        local_cursor.fetchmany.side_effect = [chunk, chunk, []]
        etl.local_conn.cursor.return_value = local_cursor
        etl.aws_conn.cursor.return_value = MagicMock()

        with patch("src.data.etl_dimensional.execute_values"):
            etl.transfer_data("SELECT 1", "dim_user", "INSERT VALUES %s")

        assert etl.aws_conn.commit.call_count == 2


# ── generate_report() ──────────────────────────────────────────────────────────
class TestGenerateReport:
    def test_warns_when_pipeline_was_not_started(self, caplog):
        etl = DimensionalETL(LOCAL_CFG, AWS_CFG)

        with caplog.at_level(logging.WARNING):
            etl.generate_report()

        assert "iniciado" in caplog.text

    def test_completed_table_appears_in_report(self, caplog):
        etl = DimensionalETL(LOCAL_CFG, AWS_CFG)
        etl.pipeline_start_time = datetime.now() - timedelta(seconds=10)
        etl.report_stats = {
            "dim_user": {
                "estado": "Completado",
                "filas_procesadas": 1_000,
                "duracion": timedelta(seconds=5),
                "error": "Ninguno",
            }
        }

        with caplog.at_level(logging.INFO):
            etl.generate_report()

        assert "dim_user" in caplog.text
        assert "Completado" in caplog.text

    def test_failed_table_shows_error_message(self, caplog):
        etl = DimensionalETL(LOCAL_CFG, AWS_CFG)
        etl.pipeline_start_time = datetime.now() - timedelta(seconds=3)
        etl.report_stats = {
            "fact_order_products": {
                "estado": "Fallido",
                "filas_procesadas": 0,
                "duracion": timedelta(seconds=1),
                "error": "timeout error",
            }
        }

        with caplog.at_level(logging.INFO):
            etl.generate_report()

        assert "Fallido" in caplog.text
        assert "timeout error" in caplog.text

    def test_total_rows_is_sum_of_all_tables(self, caplog):
        etl = DimensionalETL(LOCAL_CFG, AWS_CFG)
        etl.pipeline_start_time = datetime.now() - timedelta(seconds=5)
        etl.report_stats = {
            "dim_user": {
                "estado": "Completado",
                "filas_procesadas": 500,
                "duracion": timedelta(seconds=2),
                "error": "Ninguno",
            },
            "dim_product": {
                "estado": "Completado",
                "filas_procesadas": 300,
                "duracion": timedelta(seconds=1),
                "error": "Ninguno",
            },
        }

        with caplog.at_level(logging.INFO):
            etl.generate_report()

        # "800" should appear somewhere in the report (total rows)
        assert "800" in caplog.text

    def test_batch_id_appears_in_report(self, caplog):
        etl = DimensionalETL(LOCAL_CFG, AWS_CFG)
        etl.pipeline_start_time = datetime.now() - timedelta(seconds=1)
        etl.report_stats = {}

        with caplog.at_level(logging.INFO):
            etl.generate_report()

        assert str(etl.batch_id) in caplog.text


# ── populate_dim_user() ────────────────────────────────────────────────────────
class TestPopulateDimUser:
    def test_returns_early_when_no_null_users(self):
        etl = DimensionalETL(LOCAL_CFG, AWS_CFG)
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = []  # no users with NULL name
        mock_conn.cursor.return_value = mock_cur

        with patch("src.data.etl_dimensional.psycopg2.connect", return_value=mock_conn):
            etl.populate_dim_user()

        # No commit should happen if there's nothing to populate
        mock_conn.commit.assert_not_called()

    def test_populates_users_in_batches(self):
        etl = DimensionalETL(LOCAL_CFG, AWS_CFG)
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = [(1,), (2,), (3,)]
        mock_conn.cursor.return_value = mock_cur

        with patch("src.data.etl_dimensional.psycopg2.connect", return_value=mock_conn), \
             patch("src.data.etl_dimensional.execute_values"):
            etl.populate_dim_user()

        mock_conn.commit.assert_called()

    def test_raises_on_db_error(self):
        etl = DimensionalETL(LOCAL_CFG, AWS_CFG)

        with patch(
            "src.data.etl_dimensional.psycopg2.connect",
            side_effect=Exception("AWS unreachable"),
        ):
            with pytest.raises(Exception, match="AWS unreachable"):
                etl.populate_dim_user()


# ── run_pipeline() ─────────────────────────────────────────────────────────────
class TestRunPipeline:
    def test_aborts_and_leaves_empty_stats_when_connect_fails(self):
        etl = DimensionalETL(LOCAL_CFG, AWS_CFG)

        with patch.object(etl, "connect", return_value=False):
            etl.run_pipeline()

        assert etl.report_stats == {}

    def test_sets_pipeline_start_time(self):
        etl = DimensionalETL(LOCAL_CFG, AWS_CFG)

        # Stub all methods that require real DB or file I/O
        aws_cursor = MagicMock()
        aws_cursor.fetchall.return_value = [(1,), (2,)]

        etl.local_conn = MagicMock()
        etl.aws_conn = MagicMock()
        etl.aws_conn.cursor.return_value = aws_cursor

        before = datetime.now()
        with patch.object(etl, "connect", return_value=True), \
             patch.object(etl, "transfer_data"), \
             patch.object(etl, "populate_dim_user"), \
             patch.object(etl, "generate_report"), \
             patch.object(etl, "close"):
            etl.run_pipeline()
        after = datetime.now()

        assert etl.pipeline_start_time is not None
        assert before <= etl.pipeline_start_time <= after

    def test_calls_transfer_data_three_times(self):
        etl = DimensionalETL(LOCAL_CFG, AWS_CFG)

        aws_cursor = MagicMock()
        aws_cursor.fetchall.return_value = [(10,), (20,)]
        etl.local_conn = MagicMock()
        etl.aws_conn = MagicMock()
        etl.aws_conn.cursor.return_value = aws_cursor

        with patch.object(etl, "connect", return_value=True), \
             patch.object(etl, "transfer_data") as mock_transfer, \
             patch.object(etl, "populate_dim_user"), \
             patch.object(etl, "generate_report"), \
             patch.object(etl, "close"):
            etl.run_pipeline()

        assert mock_transfer.call_count == 3

    def test_close_is_called_after_pipeline(self):
        etl = DimensionalETL(LOCAL_CFG, AWS_CFG)

        aws_cursor = MagicMock()
        aws_cursor.fetchall.return_value = []
        etl.local_conn = MagicMock()
        etl.aws_conn = MagicMock()
        etl.aws_conn.cursor.return_value = aws_cursor

        with patch.object(etl, "connect", return_value=True), \
             patch.object(etl, "transfer_data"), \
             patch.object(etl, "populate_dim_user"), \
             patch.object(etl, "generate_report"), \
             patch.object(etl, "close") as mock_close:
            etl.run_pipeline()

        mock_close.assert_called_once()
