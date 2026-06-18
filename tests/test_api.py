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
from backend.memory import ThreadSummary, TurnResult


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


def _thread_summary(thread_id="thread-abc12345", title="Affordability", preview="hi") -> ThreadSummary:
    return ThreadSummary(
        thread_id=thread_id,
        title=title,
        preview=preview,
        created_at=1000.0,
        last_active=2000.0,
    )


# ---------------------------------------------------------------------------
# GET /api/threads
# ---------------------------------------------------------------------------

class TestListThreadsEndpoint:
    def test_returns_200(self, client, mock_service):
        mock_service.list_threads.return_value = []
        assert client.get("/api/threads").status_code == 200

    def test_returns_thread_summaries(self, client, mock_service):
        mock_service.list_threads.return_value = [_thread_summary()]
        data = client.get("/api/threads").json()
        assert data["threads"][0]["thread_id"] == "thread-abc12345"
        assert data["threads"][0]["title"] == "Affordability"

    def test_returns_502_on_service_error(self, client, mock_service):
        mock_service.list_threads.side_effect = RuntimeError("Redis down")
        assert client.get("/api/threads").status_code == 502


# ---------------------------------------------------------------------------
# POST /api/threads
# ---------------------------------------------------------------------------

class TestCreateThreadEndpoint:
    def test_returns_200(self, client, mock_service):
        mock_service.create_thread.return_value = _thread_summary()
        assert client.post("/api/threads").status_code == 200

    def test_returns_thread_id(self, client, mock_service):
        mock_service.create_thread.return_value = _thread_summary()
        data = client.post("/api/threads").json()
        assert data["thread_id"].startswith("thread-")


# ---------------------------------------------------------------------------
# GET /api/threads/{thread_id}/messages
# ---------------------------------------------------------------------------

class TestThreadMessagesEndpoint:
    def test_returns_history(self, client, mock_service):
        mock_service.read_thread_messages.return_value = [
            {"role": "user", "text": "hi"},
            {"role": "assistant", "text": "hello"},
        ]
        data = client.get("/api/threads/thread-abc12345/messages").json()
        assert data["thread_id"] == "thread-abc12345"
        assert data["messages"] == [
            {"role": "user", "text": "hi"},
            {"role": "assistant", "text": "hello"},
        ]

    def test_returns_502_on_service_error(self, client, mock_service):
        mock_service.read_thread_messages.side_effect = RuntimeError("Redis down")
        assert client.get("/api/threads/thread-abc12345/messages").status_code == 502


# ---------------------------------------------------------------------------
# GET /api/threads/{thread_id}/memory
# ---------------------------------------------------------------------------

class TestGetThreadMemoryEndpoint:
    def test_returns_200(self, client, mock_service):
        mock_service.read_session_context.return_value = []
        assert client.get("/api/threads/thread-abc12345/memory").status_code == 200

    def test_returns_memory_list(self, client, mock_service):
        mock_service.read_session_context.return_value = ["user: hello", "assistant: hi"]
        data = client.get("/api/threads/thread-abc12345/memory").json()
        assert data["thread_id"] == "thread-abc12345"
        assert data["short_term_memory"] == ["user: hello", "assistant: hi"]

    def test_returns_502_on_service_error(self, client, mock_service):
        mock_service.read_session_context.side_effect = RuntimeError("Redis down")
        assert client.get("/api/threads/thread-abc12345/memory").status_code == 502


# ---------------------------------------------------------------------------
# DELETE /api/threads/{thread_id}/memory
# ---------------------------------------------------------------------------

class TestDeleteThreadMemoryEndpoint:
    def test_returns_200(self, client, mock_service):
        mock_service.delete_session_memory.return_value = None
        assert client.delete("/api/threads/thread-abc12345/memory").status_code == 200

    def test_returns_empty_memory_list(self, client, mock_service):
        mock_service.delete_session_memory.return_value = None
        data = client.delete("/api/threads/thread-abc12345/memory").json()
        assert data["thread_id"] == "thread-abc12345"
        assert data["short_term_memory"] == []

    def test_returns_502_on_service_error(self, client, mock_service):
        mock_service.delete_session_memory.side_effect = RuntimeError("Redis down")
        assert client.delete("/api/threads/thread-abc12345/memory").status_code == 502


# ---------------------------------------------------------------------------
# POST /api/chat
# ---------------------------------------------------------------------------

def _make_turn_result(**kwargs) -> TurnResult:
    defaults = dict(
        thread_id="thread-abc12345",
        title="Affordability",
        user_text="Hello",
        assistant_text="Hi there!",
        session_context=["user: Hello"],
        long_term_memories=["Prefers a 5-year fixed rate"],
        extracted_memories=["User said hello"],
    )
    return TurnResult(**{**defaults, **kwargs})


class TestChatEndpoint:
    def test_returns_200(self, client, mock_service):
        mock_service.run_turn.return_value = _make_turn_result()
        assert client.post("/api/chat", json={"message": "Hello", "thread_id": "thread-abc12345"}).status_code == 200

    def test_response_shape(self, client, mock_service):
        mock_service.run_turn.return_value = _make_turn_result()
        data = client.post("/api/chat", json={"message": "Hello", "thread_id": "thread-abc12345"}).json()
        assert data["thread_id"] == "thread-abc12345"
        assert data["title"] == "Affordability"
        assert data["user_message"] == "Hello"
        assert data["assistant_message"] == "Hi there!"
        assert data["short_term_memory"] == ["user: Hello"]
        assert data["long_term_memory"] == ["Prefers a 5-year fixed rate"]
        assert data["extracted_long_term_memory"] == ["User said hello"]

    def test_creates_thread_when_omitted(self, client, mock_service):
        mock_service.create_thread.return_value = _thread_summary(thread_id="thread-generated")
        mock_service.run_turn.return_value = _make_turn_result(thread_id="thread-generated")
        response = client.post("/api/chat", json={"message": "Hi"})
        assert response.status_code == 200
        # A new thread is created and its id is passed positionally to run_turn:
        # (agent_memory, thread_id, message)
        positional_args = mock_service.run_turn.call_args.args
        assert positional_args[1] == "thread-generated"

    def test_rejects_empty_message(self, client):
        assert client.post("/api/chat", json={"message": ""}).status_code == 422

    def test_rejects_missing_message(self, client):
        assert client.post("/api/chat", json={}).status_code == 422

    def test_returns_502_on_service_error(self, client, mock_service):
        mock_service.run_turn.side_effect = RuntimeError("LLM failure")
        assert client.post("/api/chat", json={"message": "Hello", "thread_id": "thread-x"}).status_code == 502
