from __future__ import annotations

import logging
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from memory_service import RedisAgentMemoryService, load_config, new_session_id
from pydantic import BaseModel, Field
from redis_agent_memory import AgentMemory


logger = logging.getLogger("uvicorn.error")
app = FastAPI(title="Redis Agent Memory with LangGraph Demo")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    session_id: str | None = None


class SessionResponse(BaseModel):
    session_id: str


class ChatResponse(BaseModel):
    session_id: str
    user_message: str
    assistant_message: str
    short_term_memory: list[str]
    long_term_memory: list[str]
    extracted_long_term_memory: list[str]


class SessionMemoryResponse(BaseModel):
    session_id: str
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
    return RedisAgentMemoryService(load_config())


def agent_memory_client(service: RedisAgentMemoryService) -> AgentMemory:
    config = service.config
    return AgentMemory(
        config.agent_memory_server_url,
        store_id=config.agent_memory_store_id,
        api_key=config.agent_memory_api_key,
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


@app.post("/api/sessions", response_model=SessionResponse)
def create_session() -> SessionResponse:
    return SessionResponse(session_id=new_session_id())


@app.get("/api/sessions/{session_id}/memory", response_model=SessionMemoryResponse)
def get_session_memory(session_id: str) -> SessionMemoryResponse:
    service = get_service()
    try:
        with agent_memory_client(service) as agent_memory:
            memory = service.read_session_context(agent_memory, session_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return SessionMemoryResponse(session_id=session_id, short_term_memory=memory)


@app.delete("/api/sessions/{session_id}/memory", response_model=SessionMemoryResponse)
def delete_session_memory(session_id: str) -> SessionMemoryResponse:
    service = get_service()
    try:
        with agent_memory_client(service) as agent_memory:
            service.delete_session_memory(agent_memory, session_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return SessionMemoryResponse(session_id=session_id, short_term_memory=[])


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    service = get_service()
    session_id = request.session_id or new_session_id()
    try:
        with agent_memory_client(service) as agent_memory:
            result = service.run_turn(agent_memory, session_id, request.message.strip())
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return ChatResponse(
        session_id=result.session_id,
        user_message=result.user_text,
        assistant_message=result.assistant_text,
        short_term_memory=result.session_context,
        long_term_memory=result.long_term_memories,
        extracted_long_term_memory=result.extracted_memories,
    )
