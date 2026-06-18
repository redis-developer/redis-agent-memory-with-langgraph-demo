from __future__ import annotations

import logging
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .memory import RedisAgentMemoryService, build_service
from pydantic import BaseModel, Field
from redis_agent_memory import AgentMemory


logger = logging.getLogger("uvicorn.error")
app = FastAPI(title="Albion Home Loans · Mortgage Assistant (Redis Agent Memory + LangGraph)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    thread_id: str | None = None


class ThreadSummaryModel(BaseModel):
    thread_id: str
    title: str
    preview: str
    created_at: float
    last_active: float


class ThreadListResponse(BaseModel):
    threads: list[ThreadSummaryModel]


class ThreadMessage(BaseModel):
    role: str
    text: str


class ThreadMessagesResponse(BaseModel):
    thread_id: str
    messages: list[ThreadMessage]


class ChatResponse(BaseModel):
    thread_id: str
    title: str
    user_message: str
    assistant_message: str
    short_term_memory: list[str]
    long_term_memory: list[str]
    extracted_long_term_memory: list[str]


class ThreadMemoryResponse(BaseModel):
    thread_id: str
    short_term_memory: list[str]


class HealthResponse(BaseModel):
    status: str


class AgentMemoryHealthResponse(BaseModel):
    status: str


class ReadinessResponse(BaseModel):
    status: str
    agent_memory: AgentMemoryHealthResponse


@lru_cache
def get_service() -> RedisAgentMemoryService:
    # build_service connects to Redis Cloud and sets up the LangGraph RedisSaver
    # checkpointer + thread index that power the chat selector.
    return build_service()


def agent_memory_client(service: RedisAgentMemoryService) -> AgentMemory:
    config = service.config
    return AgentMemory(
        config.agent_memory_server_url,
        store_id=config.agent_memory_store_id,
        api_key=config.agent_memory_api_key,
    )


def _to_thread_model(summary) -> ThreadSummaryModel:
    return ThreadSummaryModel(
        thread_id=summary.thread_id,
        title=summary.title,
        preview=summary.preview,
        created_at=summary.created_at,
        last_active=summary.last_active,
    )


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/api/ready", response_model=ReadinessResponse)
def ready() -> ReadinessResponse:
    service = get_service()
    try:
        with agent_memory_client(service) as agent_memory:
            agent_memory_health = agent_memory.health(timeout_ms=3000)
    except Exception as exc:
        logger.warning("Redis Agent Memory readiness check failed", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail="Redis Agent Memory is not ready",
        ) from exc

    agent_memory_payload = AgentMemoryHealthResponse.model_validate(agent_memory_health.model_dump())
    logger.debug("Redis Agent Memory readiness check succeeded: %s", agent_memory_payload.model_dump())

    return ReadinessResponse(status="ok", agent_memory=agent_memory_payload)


@app.get("/api/threads", response_model=ThreadListResponse)
def list_threads() -> ThreadListResponse:
    service = get_service()
    try:
        summaries = service.list_threads()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return ThreadListResponse(threads=[_to_thread_model(s) for s in summaries])


@app.post("/api/threads", response_model=ThreadSummaryModel)
def create_thread() -> ThreadSummaryModel:
    service = get_service()
    try:
        summary = service.create_thread()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return _to_thread_model(summary)


@app.get("/api/threads/{thread_id}/messages", response_model=ThreadMessagesResponse)
def get_thread_messages(thread_id: str) -> ThreadMessagesResponse:
    service = get_service()
    try:
        with agent_memory_client(service) as agent_memory:
            messages = service.read_thread_messages(agent_memory, thread_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return ThreadMessagesResponse(
        thread_id=thread_id,
        messages=[ThreadMessage(role=m["role"], text=m["text"]) for m in messages],
    )


@app.get("/api/threads/{thread_id}/memory", response_model=ThreadMemoryResponse)
def get_thread_memory(thread_id: str) -> ThreadMemoryResponse:
    service = get_service()
    try:
        with agent_memory_client(service) as agent_memory:
            memory = service.read_session_context(agent_memory, thread_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return ThreadMemoryResponse(thread_id=thread_id, short_term_memory=memory)


@app.delete("/api/threads/{thread_id}/memory", response_model=ThreadMemoryResponse)
def delete_thread_memory(thread_id: str) -> ThreadMemoryResponse:
    service = get_service()
    try:
        with agent_memory_client(service) as agent_memory:
            service.delete_session_memory(agent_memory, thread_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return ThreadMemoryResponse(thread_id=thread_id, short_term_memory=[])


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    service = get_service()
    thread_id = request.thread_id or service.create_thread().thread_id
    try:
        with agent_memory_client(service) as agent_memory:
            result = service.run_turn(agent_memory, thread_id, request.message.strip())
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return ChatResponse(
        thread_id=result.thread_id,
        title=result.title,
        user_message=result.user_text,
        assistant_message=result.assistant_text,
        short_term_memory=result.session_context,
        long_term_memory=result.long_term_memories,
        extracted_long_term_memory=result.extracted_memories,
    )
