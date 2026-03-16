"""
tests/test_api.py
=================
Tests unitarios para src/api/main.py.

Usa TestClient de FastAPI con RecommendationService completamente mockeado
para evitar carga de modelo, artefactos o conexiones a base de datos.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.inference import (
    DatabaseConnectionError,
    FeatureContractError,
    UserNotFoundError,
)

# Recomendaciones sintéticas que devuelve el mock
_SAMPLE_RECS = [
    {"product_key": 1, "product_name": "Milk", "probability": 0.92},
    {"product_key": 2, "product_name": "Bread", "probability": 0.75},
    {"product_key": 3, "product_name": "Eggs", "probability": 0.61},
]


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_service():
    """Service mock con atributos y comportamiento por defecto."""
    svc = MagicMock()
    svc.model_name = "LightGBM optimizado"
    svc.n_features = 26
    svc.recommend_user.return_value = _SAMPLE_RECS
    return svc


@pytest.fixture
def client(mock_service):
    """TestClient con service reemplazado por el mock."""
    with patch("src.api.main.service", mock_service):
        with TestClient(app) as c:
            yield c


# ── GET /health ───────────────────────────────────────────────────────────────

def test_health_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_response_schema(client):
    data = client.get("/health").json()
    assert data["status"] == "ok"
    assert data["model"] == "LightGBM optimizado"
    assert data["n_features"] == 26
    assert isinstance(data["artefactos"], list)
    assert len(data["artefactos"]) > 0


# ── POST /recommend/{user_id} — éxito ────────────────────────────────────────

def test_recommend_user_returns_200(client):
    resp = client.post("/recommend/42")
    assert resp.status_code == 200


def test_recommend_user_response_schema(client):
    data = client.post("/recommend/42").json()
    assert data["user_id"] == 42
    assert isinstance(data["recommendations"], list)
    assert len(data["recommendations"]) == len(_SAMPLE_RECS)


def test_recommend_user_calls_service(client, mock_service):
    client.post("/recommend/42")
    mock_service.recommend_user.assert_called_once_with(user_id=42, top_k=10)


def test_recommend_user_recommendation_fields(client):
    data = client.post("/recommend/42").json()
    first = data["recommendations"][0]
    assert "product_key" in first
    assert "probability" in first


# ── POST /recommend/{user_id} — errores ──────────────────────────────────────

def test_recommend_user_not_found_returns_404(client, mock_service):
    mock_service.recommend_user.side_effect = UserNotFoundError("user 99 no encontrado")
    resp = client.post("/recommend/99")
    assert resp.status_code == 404


def test_recommend_user_not_found_detail(client, mock_service):
    mock_service.recommend_user.side_effect = UserNotFoundError("user 99 no encontrado")
    data = client.post("/recommend/99").json()
    assert "detail" in data


def test_recommend_feature_contract_error_returns_400(client, mock_service):
    mock_service.recommend_user.side_effect = FeatureContractError("columna faltante")
    resp = client.post("/recommend/1")
    assert resp.status_code == 400


def test_recommend_database_error_returns_503(client, mock_service):
    mock_service.recommend_user.side_effect = DatabaseConnectionError("DB no disponible")
    resp = client.post("/recommend/1")
    assert resp.status_code == 503


# ── POST /recommend/batch — éxito ────────────────────────────────────────────

def test_recommend_batch_returns_200(client):
    resp = client.post("/recommend/batch", json={"user_ids": [1, 2, 3]})
    assert resp.status_code == 200


def test_recommend_batch_response_schema(client):
    data = client.post("/recommend/batch", json={"user_ids": [1, 2]}).json()
    assert "results" in data
    assert len(data["results"]) == 2


def test_recommend_batch_each_result_has_user_id(client):
    data = client.post("/recommend/batch", json={"user_ids": [10, 20]}).json()
    user_ids = {r["user_id"] for r in data["results"]}
    assert user_ids == {10, 20}


def test_recommend_batch_calls_service_per_user(client, mock_service):
    client.post("/recommend/batch", json={"user_ids": [1, 2, 3]})
    assert mock_service.recommend_user.call_count == 3


# ── POST /recommend/batch — usuario no encontrado (error parcial) ─────────────

def test_recommend_batch_user_not_found_is_partial_error(client, mock_service):
    def side_effect(user_id, top_k):
        if user_id == 99:
            raise UserNotFoundError("no encontrado")
        return _SAMPLE_RECS

    mock_service.recommend_user.side_effect = side_effect
    data = client.post("/recommend/batch", json={"user_ids": [1, 99]}).json()

    assert len(data["results"]) == 2
    ok = next(r for r in data["results"] if r["user_id"] == 1)
    err = next(r for r in data["results"] if r["user_id"] == 99)
    assert ok["error"] is None
    assert err["error"] is not None
    assert err["recommendations"] is None


# ── POST /recommend/batch — errores que cortan todo el batch ─────────────────

def test_recommend_batch_feature_contract_error_returns_400(client, mock_service):
    mock_service.recommend_user.side_effect = FeatureContractError("contrato roto")
    resp = client.post("/recommend/batch", json={"user_ids": [1, 2]})
    assert resp.status_code == 400


def test_recommend_batch_database_error_returns_503(client, mock_service):
    mock_service.recommend_user.side_effect = DatabaseConnectionError("sin conexión")
    resp = client.post("/recommend/batch", json={"user_ids": [1, 2]})
    assert resp.status_code == 503


# ── POST /recommend/batch — validación de payload ────────────────────────────

def test_recommend_batch_empty_list_returns_422(client):
    resp = client.post("/recommend/batch", json={"user_ids": []})
    assert resp.status_code == 422


def test_recommend_batch_101_users_returns_422(client):
    resp = client.post("/recommend/batch", json={"user_ids": list(range(1, 102))})
    assert resp.status_code == 422


def test_recommend_batch_missing_body_returns_422(client):
    resp = client.post("/recommend/batch")
    assert resp.status_code == 422


def test_recommend_batch_exactly_100_users_returns_200(client):
    resp = client.post("/recommend/batch", json={"user_ids": list(range(1, 101))})
    assert resp.status_code == 200
