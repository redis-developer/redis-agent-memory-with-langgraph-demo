"""Tests for RedisAgentMemoryService methods.

The LLM and AgentMemory client are always mocked — no real API calls are made.
"""
from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest
from langchain_core.messages import AIMessage

from backend.memory import (
    DemoConfig,
    MemoryCandidate,
    MemoryExtraction,
    RedisAgentMemoryService,
)
from tests.conftest import make_not_found_error


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEMO_CONFIG = DemoConfig(
    openai_model="gpt-4.1-mini",
    agent_memory_server_url="https://memory.example.com",
    agent_memory_store_id="store-test",
    agent_memory_api_key="key-test",
    owner_id="testuser",
    namespace="test-ns",
    agent_id="test-agent",
)


@pytest.fixture
def service():
    """Service with LLM replaced by mocks — no OpenAI connection."""
    with patch("backend.memory.ChatOpenAI"):
        svc = RedisAgentMemoryService(DEMO_CONFIG)
    svc.llm = MagicMock()
    svc.extractor = MagicMock()
    return svc


@pytest.fixture
def mock_agent_memory():
    return MagicMock()


# ---------------------------------------------------------------------------
# read_session_context
# ---------------------------------------------------------------------------

class TestReadSessionContext:
    def test_returns_empty_list_on_not_found_error(self, service, mock_agent_memory):
        mock_agent_memory.get_session_memory.side_effect = make_not_found_error()
        assert service.read_session_context(mock_agent_memory, "session-123") == []

    def test_returns_empty_list_on_404_status_code(self, service, mock_agent_memory):
        exc = RuntimeError("not found")
        exc.status_code = 404  # type: ignore[attr-defined]
        mock_agent_memory.get_session_memory.side_effect = exc
        assert service.read_session_context(mock_agent_memory, "session-123") == []

    def test_raises_wrapped_error_on_other_exceptions(self, service, mock_agent_memory):
        mock_agent_memory.get_session_memory.side_effect = RuntimeError("connection refused")
        with pytest.raises(RuntimeError, match="AGENT_MEMORY_SERVER_URL"):
            service.read_session_context(mock_agent_memory, "session-123")

    def test_returns_formatted_context_lines(self, service, mock_agent_memory):
        mock_agent_memory.get_session_memory.return_value = {
            "events": [
                {"role": "user", "content": [{"text": "Hello"}]},
                {"role": "assistant", "content": [{"text": "Hi!"}]},
            ]
        }
        result = service.read_session_context(mock_agent_memory, "session-123")
        assert result == ["user: Hello", "assistant: Hi!"]

    def test_skips_events_with_empty_text(self, service, mock_agent_memory):
        mock_agent_memory.get_session_memory.return_value = {
            "events": [{"role": "user", "content": [{"text": ""}]}]
        }
        assert service.read_session_context(mock_agent_memory, "session-123") == []

    def test_truncates_to_session_context_limit(self, service, mock_agent_memory):
        events = [
            {"role": "user", "content": [{"text": f"message {i}"}]}
            for i in range(20)
        ]
        mock_agent_memory.get_session_memory.return_value = {"events": events}
        result = service.read_session_context(mock_agent_memory, "session-123")
        assert len(result) == 12  # SESSION_CONTEXT_LIMIT


# ---------------------------------------------------------------------------
# delete_session_memory
# ---------------------------------------------------------------------------

class TestDeleteSessionMemory:
    def test_silently_ignores_not_found_error(self, service, mock_agent_memory):
        mock_agent_memory.delete_session_memory.side_effect = make_not_found_error()
        # Should not raise
        service.delete_session_memory(mock_agent_memory, "session-123")

    def test_silently_ignores_404_status_code(self, service, mock_agent_memory):
        exc = RuntimeError("not found")
        exc.status_code = 404  # type: ignore[attr-defined]
        mock_agent_memory.delete_session_memory.side_effect = exc
        service.delete_session_memory(mock_agent_memory, "session-123")

    def test_raises_wrapped_error_on_other_exceptions(self, service, mock_agent_memory):
        mock_agent_memory.delete_session_memory.side_effect = RuntimeError("Redis down")
        with pytest.raises(RuntimeError, match="AGENT_MEMORY_SERVER_URL"):
            service.delete_session_memory(mock_agent_memory, "session-123")

    def test_returns_none_on_success(self, service, mock_agent_memory):
        mock_agent_memory.delete_session_memory.return_value = None
        assert service.delete_session_memory(mock_agent_memory, "session-123") is None


# ---------------------------------------------------------------------------
# run_turn — deduplication logic
# ---------------------------------------------------------------------------

class TestRunTurnDeduplication:
    """
    Verify that memories already present in retrieved long-term memory are
    NOT re-written, while genuinely new ones ARE written and returned.
    """

    def _setup_agent_memory(self, mock_agent_memory, existing_ltm_texts: list[str]):
        """Configure the mock so session memory is empty and LTM returns existing_ltm_texts."""
        mock_agent_memory.get_session_memory.return_value = {"events": []}
        mock_agent_memory.search_long_term_memory.return_value = {
            "memories": [{"text": t} for t in existing_ltm_texts]
        }
        mock_agent_memory.add_session_event.return_value = None
        mock_agent_memory.bulk_create_long_term_memories.return_value = None

    def test_new_memory_is_written(self, service, mock_agent_memory):
        self._setup_agent_memory(mock_agent_memory, existing_ltm_texts=[])
        service.llm.invoke.return_value = AIMessage(content="Sure!")
        service.extractor.invoke.return_value = MemoryExtraction(
            memories=[MemoryCandidate(text="User is vegetarian.")]
        )

        result = service.run_turn(mock_agent_memory, "session-123", "I am vegetarian.")

        mock_agent_memory.bulk_create_long_term_memories.assert_called_once()
        records = mock_agent_memory.bulk_create_long_term_memories.call_args.kwargs["memories"]
        assert len(records) == 1
        assert records[0]["text"] == "User is vegetarian."
        assert "User is vegetarian." in result.extracted_memories

    def test_duplicate_memory_is_not_rewritten(self, service, mock_agent_memory):
        existing = "User prefers Delta Airlines."
        self._setup_agent_memory(mock_agent_memory, existing_ltm_texts=[existing])
        service.llm.invoke.return_value = AIMessage(content="Got it!")
        # Extractor returns the same memory that's already in LTM
        service.extractor.invoke.return_value = MemoryExtraction(
            memories=[MemoryCandidate(text=existing)]
        )

        result = service.run_turn(mock_agent_memory, "session-123", "I prefer Delta Airlines.")

        mock_agent_memory.bulk_create_long_term_memories.assert_not_called()
        assert result.extracted_memories == []

    def test_only_new_memories_written_when_mixed(self, service, mock_agent_memory):
        existing = "User prefers Delta Airlines."
        self._setup_agent_memory(mock_agent_memory, existing_ltm_texts=[existing])
        service.llm.invoke.return_value = AIMessage(content="Noted!")
        service.extractor.invoke.return_value = MemoryExtraction(
            memories=[
                MemoryCandidate(text=existing),             # duplicate — should be skipped
                MemoryCandidate(text="User is vegetarian."), # new — should be written
            ]
        )

        result = service.run_turn(mock_agent_memory, "session-123", "I prefer Delta and I'm vegetarian.")

        mock_agent_memory.bulk_create_long_term_memories.assert_called_once()
        records = mock_agent_memory.bulk_create_long_term_memories.call_args.kwargs["memories"]
        written_texts = [r["text"] for r in records]
        assert existing not in written_texts
        assert "User is vegetarian." in written_texts
        assert result.extracted_memories == ["User is vegetarian."]

    def test_blank_memory_candidates_are_ignored(self, service, mock_agent_memory):
        self._setup_agent_memory(mock_agent_memory, existing_ltm_texts=[])
        service.llm.invoke.return_value = AIMessage(content="Ok!")
        service.extractor.invoke.return_value = MemoryExtraction(
            memories=[MemoryCandidate(text="   ")]  # blank after strip
        )

        result = service.run_turn(mock_agent_memory, "session-123", "Hello.")

        mock_agent_memory.bulk_create_long_term_memories.assert_not_called()
        assert result.extracted_memories == []

    def test_session_events_are_always_written(self, service, mock_agent_memory):
        self._setup_agent_memory(mock_agent_memory, existing_ltm_texts=[])
        service.llm.invoke.return_value = AIMessage(content="Hello!")
        service.extractor.invoke.return_value = MemoryExtraction(memories=[])

        service.run_turn(mock_agent_memory, "session-123", "Hi there.")

        assert mock_agent_memory.add_session_event.call_count == 2  # user + assistant
