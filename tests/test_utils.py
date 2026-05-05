"""Unit tests for pure utility functions in backend/memory.py.

These tests have no external dependencies — no LLM calls, no Redis connections.
"""
from __future__ import annotations

import hashlib
import re
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from backend.memory import (
    coerce_events,
    coerce_memories,
    explain_agent_memory_error,
    get_event_role,
    get_event_text,
    get_memory_text,
    is_not_found_error,
    memory_id,
    message_text,
    new_session_id,
    normalize_memory_text,
)
from redis_agent_memory import errors

from tests.conftest import make_not_found_error


# ---------------------------------------------------------------------------
# memory_id
# ---------------------------------------------------------------------------

class TestMemoryId:
    def test_format(self):
        result = memory_id("user1", "ns", "some text")
        assert result.startswith("demo-")
        assert len(result) == len("demo-") + 32

    def test_only_hex_chars_in_suffix(self):
        suffix = memory_id("u", "n", "t")[len("demo-"):]
        assert re.fullmatch(r"[0-9a-f]+", suffix)

    def test_deterministic(self):
        assert memory_id("user1", "ns", "text") == memory_id("user1", "ns", "text")

    def test_different_owner_differs(self):
        assert memory_id("user1", "ns", "text") != memory_id("user2", "ns", "text")

    def test_different_namespace_differs(self):
        assert memory_id("u", "ns1", "text") != memory_id("u", "ns2", "text")

    def test_different_text_differs(self):
        assert memory_id("u", "ns", "text1") != memory_id("u", "ns", "text2")

    def test_hash_matches_sha256(self):
        owner, ns, text = "u", "n", "t"
        expected = "demo-" + hashlib.sha256(f"{owner}:{ns}:{text}".encode()).hexdigest()[:32]
        assert memory_id(owner, ns, text) == expected


# ---------------------------------------------------------------------------
# normalize_memory_text
# ---------------------------------------------------------------------------

class TestNormalizeMemoryText:
    def test_lowercases(self):
        assert normalize_memory_text("HELLO") == "hello"

    def test_strips_punctuation(self):
        assert normalize_memory_text("Hello, World!") == "hello world"

    def test_collapses_whitespace(self):
        assert normalize_memory_text("  foo   bar  ") == "foo bar"

    def test_empty_string(self):
        assert normalize_memory_text("") == ""

    def test_only_special_chars(self):
        assert normalize_memory_text("!@#$%") == ""

    def test_realistic_sentence(self):
        assert normalize_memory_text("I prefer Delta Airlines.") == "i prefer delta airlines"

    def test_equivalent_sentences_normalize_equal(self):
        a = normalize_memory_text("User prefers Delta.")
        b = normalize_memory_text("user prefers delta")
        assert a == b


# ---------------------------------------------------------------------------
# new_session_id
# ---------------------------------------------------------------------------

class TestNewSessionId:
    def test_prefix(self):
        assert new_session_id().startswith("session-")

    def test_suffix_length(self):
        suffix = new_session_id()[len("session-"):]
        assert len(suffix) == 8

    def test_suffix_is_hex(self):
        suffix = new_session_id()[len("session-"):]
        assert re.fullmatch(r"[0-9a-f]+", suffix)

    def test_unique(self):
        assert new_session_id() != new_session_id()


# ---------------------------------------------------------------------------
# message_text
# ---------------------------------------------------------------------------

class TestMessageText:
    def test_string_content(self):
        assert message_text(HumanMessage(content="hello")) == "hello"

    def test_list_of_text_dicts(self):
        msg = HumanMessage(content=[{"text": "part1"}, {"text": "part2"}])
        assert message_text(msg) == "part1\npart2"

    def test_list_with_non_dict_item(self):
        msg = HumanMessage(content=[{"text": "part1"}, "raw"])
        assert message_text(msg) == "part1\nraw"

    def test_empty_list(self):
        assert message_text(HumanMessage(content=[])) == ""

    def test_ai_message(self):
        assert message_text(AIMessage(content="response")) == "response"


# ---------------------------------------------------------------------------
# coerce_memories
# ---------------------------------------------------------------------------

class TestCoerceMemories:
    def test_object_with_memories_attr(self):
        obj = SimpleNamespace(memories=["a", "b"])
        assert coerce_memories(obj) == ["a", "b"]

    def test_dict_with_memories_key(self):
        assert coerce_memories({"memories": ["x"]}) == ["x"]

    def test_object_missing_attr_returns_empty(self):
        assert coerce_memories(SimpleNamespace()) == []

    def test_dict_missing_key_returns_empty(self):
        assert coerce_memories({}) == []

    def test_none_memories_returns_empty(self):
        assert coerce_memories({"memories": None}) == []

    def test_attr_takes_precedence_over_dict_key(self):
        obj = SimpleNamespace(memories=["from-attr"])
        assert coerce_memories(obj) == ["from-attr"]


# ---------------------------------------------------------------------------
# coerce_events
# ---------------------------------------------------------------------------

class TestCoerceEvents:
    def test_object_with_events_attr(self):
        obj = SimpleNamespace(events=["e1", "e2"])
        assert coerce_events(obj) == ["e1", "e2"]

    def test_dict_with_events_key(self):
        assert coerce_events({"events": ["e"]}) == ["e"]

    def test_missing_returns_empty(self):
        assert coerce_events(SimpleNamespace()) == []

    def test_none_events_returns_empty(self):
        assert coerce_events({"events": None}) == []


# ---------------------------------------------------------------------------
# get_event_role
# ---------------------------------------------------------------------------

class TestGetEventRole:
    def test_dict_with_string_role_lowercased(self):
        assert get_event_role({"role": "User"}) == "user"

    def test_dict_with_enum_like_role(self):
        role = SimpleNamespace(value="Assistant")
        assert get_event_role({"role": role}) == "assistant"

    def test_object_with_string_role(self):
        assert get_event_role(SimpleNamespace(role="USER")) == "user"

    def test_object_with_enum_like_role(self):
        role = SimpleNamespace(value="System")
        assert get_event_role(SimpleNamespace(role=role)) == "system"


# ---------------------------------------------------------------------------
# get_event_text
# ---------------------------------------------------------------------------

class TestGetEventText:
    def test_dict_event_joins_parts(self):
        event = {"content": [{"text": "hello"}, {"text": "world"}]}
        assert get_event_text(event) == "hello\nworld"

    def test_object_event_joins_parts(self):
        content = [SimpleNamespace(text="hi"), SimpleNamespace(text="there")]
        assert get_event_text(SimpleNamespace(content=content)) == "hi\nthere"

    def test_empty_parts_are_filtered(self):
        event = {"content": [{"text": "hello"}, {"text": ""}]}
        assert get_event_text(event) == "hello"

    def test_empty_content_list(self):
        assert get_event_text({"content": []}) == ""

    def test_none_content(self):
        assert get_event_text({"content": None}) == ""


# ---------------------------------------------------------------------------
# get_memory_text
# ---------------------------------------------------------------------------

class TestGetMemoryText:
    def test_dict_returns_text_value(self):
        assert get_memory_text({"text": "remember this"}) == "remember this"

    def test_object_returns_text_attr(self):
        assert get_memory_text(SimpleNamespace(text="remember that")) == "remember that"

    def test_dict_missing_key_returns_empty(self):
        assert get_memory_text({}) == ""


# ---------------------------------------------------------------------------
# is_not_found_error
# ---------------------------------------------------------------------------

class TestIsNotFoundError:
    def test_status_code_404_returns_true(self):
        exc = Exception("not found")
        exc.status_code = 404  # type: ignore[attr-defined]
        assert is_not_found_error(exc) is True

    def test_status_code_500_returns_false(self):
        exc = Exception("server error")
        exc.status_code = 500  # type: ignore[attr-defined]
        assert is_not_found_error(exc) is False

    def test_generic_exception_returns_false(self):
        assert is_not_found_error(ValueError("oops")) is False

    def test_not_found_error_response_content_returns_true(self):
        exc = make_not_found_error()
        assert is_not_found_error(exc) is True


# ---------------------------------------------------------------------------
# explain_agent_memory_error
# ---------------------------------------------------------------------------

class TestExplainAgentMemoryError:
    def test_returns_runtime_error(self):
        result = explain_agent_memory_error("session read", ValueError("boom"))
        assert isinstance(result, RuntimeError)

    def test_message_contains_operation(self):
        result = explain_agent_memory_error("long-term memory search", Exception("x"))
        assert "long-term memory search" in str(result)

    def test_message_contains_original_error(self):
        result = explain_agent_memory_error("op", Exception("original problem"))
        assert "original problem" in str(result)

    def test_message_contains_env_var_hint(self):
        result = explain_agent_memory_error("op", Exception("x"))
        assert "AGENT_MEMORY_SERVER_URL" in str(result)
