"""
tests/test_schemas.py
=====================
Tests unitarios para src/api/schemas.py.

Verifica validación de Pydantic: tipos correctos, límites de BatchRequest,
campos opcionales y construcción de respuestas.
"""

import pytest
from pydantic import ValidationError

from src.api.schemas import (
    BatchItemResult,
    BatchRequest,
    BatchResponse,
    HealthResponse,
    RecommendationItem,
    RecommendResponse,
)


# ── RecommendationItem ────────────────────────────────────────────────────────

def test_recommendation_item_valid():
    item = RecommendationItem(product_key=42, product_name="Milk", probability=0.85)
    assert item.product_key == 42
    assert item.product_name == "Milk"
    assert item.probability == 0.85


def test_recommendation_item_without_name():
    item = RecommendationItem(product_key=7, probability=0.5)
    assert item.product_name is None


def test_recommendation_item_requires_product_key():
    with pytest.raises(ValidationError):
        RecommendationItem(probability=0.5)


def test_recommendation_item_requires_probability():
    with pytest.raises(ValidationError):
        RecommendationItem(product_key=1)


# ── RecommendResponse ─────────────────────────────────────────────────────────

def test_recommend_response_valid():
    items = [RecommendationItem(product_key=i, probability=0.9 - i * 0.1) for i in range(3)]
    resp = RecommendResponse(user_id=100, recommendations=items)
    assert resp.user_id == 100
    assert len(resp.recommendations) == 3


def test_recommend_response_empty_recommendations():
    resp = RecommendResponse(user_id=1, recommendations=[])
    assert resp.recommendations == []


def test_recommend_response_requires_user_id():
    with pytest.raises(ValidationError):
        RecommendResponse(recommendations=[])


# ── BatchRequest ──────────────────────────────────────────────────────────────

def test_batch_request_single_user():
    req = BatchRequest(user_ids=[42])
    assert req.user_ids == [42]


def test_batch_request_max_users():
    req = BatchRequest(user_ids=list(range(1, 101)))
    assert len(req.user_ids) == 100


def test_batch_request_empty_list_fails():
    with pytest.raises(ValidationError) as exc_info:
        BatchRequest(user_ids=[])
    assert "min_length" in str(exc_info.value) or "too_short" in str(exc_info.value)


def test_batch_request_exceeds_max_fails():
    with pytest.raises(ValidationError) as exc_info:
        BatchRequest(user_ids=list(range(1, 102)))  # 101 usuarios
    assert "max_length" in str(exc_info.value) or "too_long" in str(exc_info.value)


def test_batch_request_exactly_100_users():
    req = BatchRequest(user_ids=list(range(100, 200)))
    assert len(req.user_ids) == 100


def test_batch_request_invalid_type_fails():
    with pytest.raises(ValidationError):
        BatchRequest(user_ids=["not_an_int"])


# ── BatchItemResult ───────────────────────────────────────────────────────────

def test_batch_item_result_with_recommendations():
    recs = [RecommendationItem(product_key=1, probability=0.9)]
    result = BatchItemResult(user_id=5, recommendations=recs)
    assert result.user_id == 5
    assert result.error is None
    assert len(result.recommendations) == 1


def test_batch_item_result_with_error():
    result = BatchItemResult(user_id=99, error="Usuario no encontrado")
    assert result.user_id == 99
    assert result.recommendations is None
    assert result.error == "Usuario no encontrado"


def test_batch_item_result_defaults_are_none():
    result = BatchItemResult(user_id=1)
    assert result.recommendations is None
    assert result.error is None


# ── BatchResponse ─────────────────────────────────────────────────────────────

def test_batch_response_valid():
    results = [
        BatchItemResult(
            user_id=i,
            recommendations=[RecommendationItem(product_key=i * 10, probability=0.8)],
        )
        for i in range(1, 4)
    ]
    resp = BatchResponse(results=results)
    assert len(resp.results) == 3


def test_batch_response_empty_results():
    resp = BatchResponse(results=[])
    assert resp.results == []


def test_batch_response_mixed_success_and_error():
    results = [
        BatchItemResult(
            user_id=1,
            recommendations=[RecommendationItem(product_key=10, probability=0.9)],
        ),
        BatchItemResult(user_id=2, error="no historial"),
    ]
    resp = BatchResponse(results=results)
    assert resp.results[0].error is None
    assert resp.results[1].recommendations is None


# ── HealthResponse ────────────────────────────────────────────────────────────

def test_health_response_valid():
    resp = HealthResponse(
        status="ok",
        model="LightGBM optimizado",
        n_features=26,
        artefactos=["model.pkl", "cluster_models.pkl", "model_log.json"],
    )
    assert resp.status == "ok"
    assert resp.n_features == 26
    assert len(resp.artefactos) == 3


def test_health_response_requires_all_fields():
    with pytest.raises(ValidationError):
        HealthResponse(status="ok", model="LightGBM")  # falta n_features y artefactos
