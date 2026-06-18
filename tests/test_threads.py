"""Tests for the ThreadIndex registry that backs the chat selector.

Uses fakeredis so no real Redis connection is needed.
"""
from __future__ import annotations

import fakeredis
import pytest

from backend.memory import DEFAULT_THREAD_TITLE, ThreadIndex


@pytest.fixture
def index():
    client = fakeredis.FakeRedis(decode_responses=True)
    return ThreadIndex(client)


class TestCreate:
    def test_create_returns_summary_with_thread_id(self, index):
        summary = index.create("owner-1")
        assert summary.thread_id.startswith("thread-")
        assert summary.title == DEFAULT_THREAD_TITLE

    def test_created_thread_is_listed(self, index):
        summary = index.create("owner-1")
        listed = index.list("owner-1")
        assert [t.thread_id for t in listed] == [summary.thread_id]


class TestTouchAndList:
    def test_touch_creates_metadata_when_missing(self, index):
        index.touch("owner-1", "thread-aaa", preview="hello", title="Affordability")
        summary = index.get("thread-aaa")
        assert summary is not None
        assert summary.title == "Affordability"
        assert summary.preview == "hello"

    def test_list_orders_by_last_active_desc(self, index):
        index.touch("owner-1", "thread-old", preview="first")
        index.touch("owner-1", "thread-new", preview="second")
        ordered = [t.thread_id for t in index.list("owner-1")]
        assert ordered.index("thread-new") < ordered.index("thread-old")

    def test_touch_again_moves_thread_to_front(self, index):
        index.touch("owner-1", "thread-a", preview="a")
        index.touch("owner-1", "thread-b", preview="b")
        index.touch("owner-1", "thread-a", preview="a again")
        assert index.list("owner-1")[0].thread_id == "thread-a"

    def test_threads_are_scoped_per_owner(self, index):
        index.touch("owner-1", "thread-a")
        index.touch("owner-2", "thread-b")
        assert [t.thread_id for t in index.list("owner-1")] == ["thread-a"]
        assert [t.thread_id for t in index.list("owner-2")] == ["thread-b"]


class TestGet:
    def test_missing_thread_returns_none(self, index):
        assert index.get("thread-missing") is None


class TestSetTitle:
    def test_updates_title(self, index):
        index.touch("owner-1", "thread-a", title="Old")
        index.set_title("thread-a", "New title")
        assert index.get("thread-a").title == "New title"


class TestReset:
    def test_reset_clears_owner_threads(self, index):
        index.touch("owner-1", "thread-a")
        index.touch("owner-1", "thread-b")
        index.reset("owner-1")
        assert index.list("owner-1") == []
        assert index.get("thread-a") is None
