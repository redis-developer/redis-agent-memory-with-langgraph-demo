"""Shared fixtures and helpers for the test suite."""
from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest

from redis_agent_memory.errors.notfounderrorresponsecontent import (
    NotFoundErrorResponseContent,
    NotFoundErrorResponseContentData,
)
from redis_agent_memory.models.notfounderrortype import NotFoundErrorType


def make_not_found_error() -> NotFoundErrorResponseContent:
    """Build a real NotFoundErrorResponseContent with a mocked httpx.Response."""
    response = MagicMock(spec=httpx.Response)
    response.status_code = 404
    response.text = "not found"
    response.headers = httpx.Headers()
    data = NotFoundErrorResponseContentData(
        title="Not Found",
        type=NotFoundErrorType.ROOT_ERRORS_RESOURCE_NOT_FOUND,
    )
    return NotFoundErrorResponseContent(data=data, raw_response=response)
