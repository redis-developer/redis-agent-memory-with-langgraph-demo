"""API endpoint tests using FastAPI TestClient.

get_service() is called directly (not via Depends), so we patch it at the
module level. agent_memory_client() is also patched so no real Redis
connection is attempted.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.app import app
from backend.memory import TurnResult


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_service():
    return MagicMock()


@pytest.fixture
def mock_agent_memory():
    """A MagicMock that works correctly as a context manager."""
    m = MagicMock()
    m.__enter__ = MagicMock(return_value=m)
    m.__exit__ = MagicMock(return_value=False)
    return m


@pytest.fixture
def client(mock_service, mock_agent_memory):
    with patch("backend.app.get_service", return_value=mock_service), \
         patch("backend.app.agent_memory_client", return_value=mock_agent_memory):
        yield TestClient(app)


# ---------------------------------------------------------------------------
# GET /api/health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_returns_200(self, client):
        assert client.get("/api/health").status_code == 200

    def test_returns_ok_status(self, client):
        assert client.get("/api/health").json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# GET /api/ready
# ---------------------------------------------------------------------------

class TestReadinessEndpoint:
    def test_happy_path_returns_200(self, mock_service, mock_agent_memory):
        health_response = MagicMock()
        health_response.model_dump.return_value = {"status": "ok"}
        mock_agent_memory.health.return_value = health_response

        with patch("backend.app.get_service", return_value=mock_service), \
             patch("backend.app.agent_memory_client", return_value=mock_agent_memory):
            response = TestClient(app).get("/api/ready")

        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert response.json()["agent_memory"]["status"] == "ok"

    def test_returns_503_when_memory_unreachable(self, mock_service, mock_agent_memory):
        mock_agent_memory.health.side_effect = RuntimeError("connection refused")

        with patch("backend.app.get_service", return_value=mock_service), \
             patch("backend.app.agent_memory_client", return_value=mock_agent_memory):
            response = TestClient(app).get("/api/ready")

        assert response.status_code == 503


# ---------------------------------------------------------------------------
# POST /api/sessions
# ---------------------------------------------------------------------------

class TestCreateSessionEndpoint:
    def test_returns_200(self, client):
        assert client.post("/api/sessions").status_code == 200

    def test_returns_session_id(self, client):
        data = client.post("/api/sessions").json()
        assert "session_id" in data
        assert data["session_id"].startswith("session-")

    def test_session_ids_are_unique(self, client):
        id1 = client.post("/api/sessions").json()["session_id"]
        id2 = client.post("/api/sessions").json()["session_id"]
        assert id1 != id2


# ---------------------------------------------------------------------------
# GET /api/sessions/{session_id}/memory
# ---------------------------------------------------------------------------

class TestGetSessionMemoryEndpoint:
    def test_returns_200(self, client, mock_service):
        mock_service.read_session_context.return_value = []
        assert client.get("/api/sessions/session-abc12345/memory").status_code == 200

    def test_returns_memory_list(self, client, mock_service):
        mock_service.read_session_context.return_value = ["user: hello", "assistant: hi"]
        response = client.get("/api/sessions/session-abc12345/memory")
        data = response.json()
        assert data["session_id"] == "session-abc12345"
        assert data["short_term_memory"] == ["user: hello", "assistant: hi"]

    def test_returns_502_on_service_error(self, client, mock_service):
        mock_service.read_session_context.side_effect = RuntimeError("Redis down")
        assert client.get("/api/sessions/session-abc12345/memory").status_code == 502


# ---------------------------------------------------------------------------
# DELETE /api/sessions/{session_id}/memory
# ---------------------------------------------------------------------------

class TestDeleteSessionMemoryEndpoint:
    def test_returns_200(self, client, mock_service):
        mock_service.delete_session_memory.return_value = None
        assert client.delete("/api/sessions/session-abc12345/memory").status_code == 200

    def test_returns_empty_memory_list(self, client, mock_service):
        mock_service.delete_session_memory.return_value = None
        data = client.delete("/api/sessions/session-abc12345/memory").json()
        assert data["session_id"] == "session-abc12345"
        assert data["short_term_memory"] == []

    def test_returns_502_on_service_error(self, client, mock_service):
        mock_service.delete_session_memory.side_effect = RuntimeError("Redis down")
        assert client.delete("/api/sessions/session-abc12345/memory").status_code == 502


# ---------------------------------------------------------------------------
# POST /api/chat
# ---------------------------------------------------------------------------

def _make_turn_result(**kwargs) -> TurnResult:
    defaults = dict(
        session_id="session-abc12345",
        user_text="Hello",
        assistant_text="Hi there!",
        session_context=["user: Hello"],
        long_term_memories=["Prefers Delta"],
        extracted_memories=["User said hello"],
    )
    return TurnResult(**{**defaults, **kwargs})


class TestChatEndpoint:
    def test_returns_200(self, client, mock_service):
        mock_service.run_turn.return_value = _make_turn_result()
        assert client.post("/api/chat", json={"message": "Hello", "session_id": "session-abc12345"}).status_code == 200

    def test_response_shape(self, client, mock_service):
        mock_service.run_turn.return_value = _make_turn_result()
        data = client.post("/api/chat", json={"message": "Hello", "session_id": "session-abc12345"}).json()
        assert data["session_id"] == "session-abc12345"
        assert data["user_message"] == "Hello"
        assert data["assistant_message"] == "Hi there!"
        assert data["short_term_memory"] == ["user: Hello"]
        assert data["long_term_memory"] == ["Prefers Delta"]
        assert data["extracted_long_term_memory"] == ["User said hello"]

    def test_generates_session_id_when_omitted(self, client, mock_service):
        mock_service.run_turn.return_value = _make_turn_result(session_id="session-generated")
        response = client.post("/api/chat", json={"message": "Hi"})
        assert response.status_code == 200
        # run_turn is called with positional args: (agent_memory, session_id, message)
        positional_args = mock_service.run_turn.call_args.args
        assert positional_args[1].startswith("session-")

    def test_rejects_empty_message(self, client):
        assert client.post("/api/chat", json={"message": ""}).status_code == 422

    def test_rejects_missing_message(self, client):
        assert client.post("/api/chat", json={}).status_code == 422

    def test_returns_502_on_service_error(self, client, mock_service):
        mock_service.run_turn.side_effect = RuntimeError("LLM failure")
        assert client.post("/api/chat", json={"message": "Hello"}).status_code == 502
