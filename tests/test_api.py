"""API tests for the SHL Assessment Recommender."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def assert_chat_response_schema(payload: dict) -> None:
    """Validate the required assignment response schema."""
    assert isinstance(payload, dict)

    # Required top-level keys.
    assert set(payload.keys()) == {
        "reply",
        "recommendations",
        "end_of_conversation",
    }

    # reply
    assert isinstance(payload["reply"], str)
    assert payload["reply"].strip()

    # recommendations
    recommendations = payload["recommendations"]
    assert isinstance(recommendations, list)
    assert len(recommendations) == 0 or 1 <= len(recommendations) <= 10

    for recommendation in recommendations:
        assert set(recommendation.keys()) == {"name", "url", "test_type"}

        assert isinstance(recommendation["name"], str)
        assert recommendation["name"].strip()

        assert isinstance(recommendation["url"], str)
        assert recommendation["url"].startswith("https://www.shl.com/")

        assert isinstance(recommendation["test_type"], str)
        assert recommendation["test_type"].strip()

    # end_of_conversation
    assert isinstance(payload["end_of_conversation"], bool)


def test_health_endpoint() -> None:
    """GET /health should return HTTP 200 and {'status': 'ok'}."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_schema_compliance() -> None:
    """POST /chat should always return the required schema."""
    response = client.post(
        "/chat",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "I am hiring a Java developer.",
                }
            ]
        },
    )

    assert response.status_code == 200
    assert_chat_response_schema(response.json())


def test_vague_input_triggers_clarification() -> None:
    """Vague first-turn requests should not produce recommendations."""
    response = client.post(
        "/chat",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "I need an assessment.",
                }
            ]
        },
    )

    assert response.status_code == 200

    payload = response.json()
    assert_chat_response_schema(payload)

    assert payload["recommendations"] == []
    assert payload["end_of_conversation"] is False


def test_off_topic_input_is_refused() -> None:
    """Off-topic requests should be refused with empty recommendations."""
    response = client.post(
        "/chat",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "What salary should I offer a software engineer?",
                }
            ]
        },
    )

    assert response.status_code == 200

    payload = response.json()
    assert_chat_response_schema(payload)

    assert payload["recommendations"] == []
    assert payload["end_of_conversation"] is False

    # The reply should indicate scope limitation.
    assert "only help with selecting shl assessments" in payload["reply"].lower()


def test_response_structure_for_greeting() -> None:
    """Even greeting responses must match the assignment schema."""
    response = client.post(
        "/chat",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "Hello",
                }
            ]
        },
    )

    assert response.status_code == 200
    assert_chat_response_schema(response.json())


def test_invalid_request_returns_validation_error() -> None:
    """Malformed requests should be rejected by FastAPI validation."""
    response = client.post("/chat", json={})

    # FastAPI/Pydantic validation error.
    assert response.status_code == 422