from __future__ import annotations

import hashlib
import os
import re
import time
import uuid
from dataclasses import dataclass
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from redis_agent_memory import AgentMemory, errors, models
from typing_extensions import TypedDict


DEMO_SOURCE = "langgraph-demo"
SESSION_CONTEXT_LIMIT = 12


class MemoryCandidate(BaseModel):
    text: str = Field(description="A durable memory written as one concise sentence.")
    topics: list[str] = Field(default_factory=list)
    memory_type: Literal["semantic", "episodic"] = "semantic"


class MemoryExtraction(BaseModel):
    memories: list[MemoryCandidate] = Field(default_factory=list)


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    owner_id: str
    session_id: str
    namespace: str
    session_context: list[str]
    recalled_memories: list[str]
    extracted_memories: list[str]


@dataclass(frozen=True)
class DemoConfig:
    openai_model: str
    agent_memory_server_url: str
    agent_memory_store_id: str
    agent_memory_api_key: str
    owner_id: str
    namespace: str
    agent_id: str


@dataclass(frozen=True)
class TurnResult:
    session_id: str
    user_text: str
    assistant_text: str
    session_context: list[str]
    long_term_memories: list[str]
    extracted_memories: list[str]


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def load_config() -> DemoConfig:
    load_dotenv()
    return DemoConfig(
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        agent_memory_server_url=require_env("AGENT_MEMORY_SERVER_URL"),
        agent_memory_store_id=require_env("AGENT_MEMORY_STORE_ID"),
        agent_memory_api_key=require_env("AGENT_MEMORY_API_KEY"),
        owner_id=os.getenv("DEMO_OWNER_ID", "riferrei"),
        namespace=os.getenv("DEMO_NAMESPACE", "langgraph-travel-demo"),
        agent_id=os.getenv("DEMO_AGENT_ID", "travel-agent"),
    )


def now_ms() -> int:
    return int(time.time() * 1000)


def message_text(message: AnyMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def memory_id(owner_id: str, namespace: str, text: str) -> str:
    digest = hashlib.sha256(f"{owner_id}:{namespace}:{text}".encode("utf-8")).hexdigest()
    return f"demo-{digest[:32]}"


def normalize_memory_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def new_session_id() -> str:
    return f"session-{uuid.uuid4().hex[:8]}"


def coerce_memories(response: object) -> list[object]:
    memories = getattr(response, "memories", None)
    if memories is None and isinstance(response, dict):
        memories = response.get("memories")
    return list(memories or [])


def get_memory_text(memory: object) -> str:
    if isinstance(memory, dict):
        return str(memory.get("text", ""))
    return str(getattr(memory, "text", ""))


def coerce_events(response: object) -> list[object]:
    events = getattr(response, "events", None)
    if events is None and isinstance(response, dict):
        events = response.get("events")
    return list(events or [])


def get_event_role(event: object) -> str:
    role = event.get("role") if isinstance(event, dict) else getattr(event, "role", "")
    return str(getattr(role, "value", role)).lower()


def get_event_text(event: object) -> str:
    content = event.get("content", []) if isinstance(event, dict) else getattr(event, "content", [])
    parts = []
    for item in content or []:
        if isinstance(item, dict):
            parts.append(str(item.get("text", "")))
        else:
            parts.append(str(getattr(item, "text", "")))
    return "\n".join(part for part in parts if part)


def is_not_found_error(exc: Exception) -> bool:
    return isinstance(exc, errors.NotFoundErrorResponseContent) or getattr(exc, "status_code", None) == 404


def explain_agent_memory_error(operation: str, exc: Exception) -> RuntimeError:
    hint = (
        f"Redis Agent Memory {operation} failed. Check AGENT_MEMORY_SERVER_URL, "
        "AGENT_MEMORY_STORE_ID, and AGENT_MEMORY_API_KEY. The server URL should be "
        "the Agent Memory data-plane base URL, not the PyPI/docs URL."
    )
    return RuntimeError(f"{hint}\n\nOriginal error: {exc}")


class RedisAgentMemoryService:
    def __init__(self, config: DemoConfig) -> None:
        self.config = config
        self.llm = ChatOpenAI(model=config.openai_model, temperature=0.2)
        self.extractor = self.llm.with_structured_output(MemoryExtraction)

    def build_graph(self, agent_memory: AgentMemory):
        def retrieve_session_context(state: AgentState) -> dict:
            try:
                response = agent_memory.get_session_memory(session_id=state["session_id"])
            except Exception as exc:
                if is_not_found_error(exc):
                    return {"session_context": []}
                raise explain_agent_memory_error("session memory read", exc)

            session_context = []
            for event in coerce_events(response)[-SESSION_CONTEXT_LIMIT:]:
                text = get_event_text(event).strip()
                if text:
                    session_context.append(f"{get_event_role(event)}: {text}")
            return {"session_context": session_context}

        def retrieve_long_term_memories(state: AgentState) -> dict:
            last_user_message = next(
                (message for message in reversed(state["messages"]) if isinstance(message, HumanMessage)),
                None,
            )
            query = message_text(last_user_message) if last_user_message else ""

            try:
                response = agent_memory.search_long_term_memory(
                    request={
                        "text": query,
                        "limit": 5,
                        "filter": {
                            "ownerId": {"eq": state["owner_id"]},
                            "namespace": {"eq": state["namespace"]},
                        },
                        "filterOp": models.FilterConjunction.ALL,
                    }
                )
            except Exception as exc:
                raise explain_agent_memory_error("long-term memory search", exc)
            recalled = [get_memory_text(memory) for memory in coerce_memories(response)]
            return {"recalled_memories": recalled}

        def call_model(state: AgentState) -> dict:
            session_context = "\n".join(f"- {event}" for event in state["session_context"])
            if not session_context:
                session_context = "- No previous turns in this session."

            long_term_context = "\n".join(f"- {memory}" for memory in state["recalled_memories"])
            if not long_term_context:
                long_term_context = "- No relevant long-term memories found."

            system_prompt = f"""You are a polished travel concierge.

Use short-term memory for continuity within the current session.
Use long-term memory for durable user facts, preferences, and constraints.
Do not mention implementation details.
Keep answers concise, specific, and naturally personalized.

Short-term memory from this session:
{session_context}

Relevant long-term memories:
{long_term_context}
"""
            response = self.llm.invoke([SystemMessage(content=system_prompt), *state["messages"]])
            return {"messages": [response]}

        def write_memory(state: AgentState) -> dict:
            user_message = next(
                (message for message in reversed(state["messages"]) if isinstance(message, HumanMessage)),
                None,
            )
            assistant_message = next(
                (message for message in reversed(state["messages"]) if isinstance(message, AIMessage)),
                None,
            )
            if user_message is None or assistant_message is None:
                return {"extracted_memories": []}

            user_text = message_text(user_message)
            assistant_text = message_text(assistant_message)

            try:
                agent_memory.add_session_event(
                    session_id=state["session_id"],
                    actor_id=state["owner_id"],
                    role=models.MessageRole.USER,
                    content=[{"text": user_text}],
                    created_at=now_ms(),
                    metadata={"source": DEMO_SOURCE},
                )
                agent_memory.add_session_event(
                    session_id=state["session_id"],
                    actor_id=self.config.agent_id,
                    role=models.MessageRole.ASSISTANT,
                    content=[{"text": assistant_text}],
                    created_at=now_ms(),
                    metadata={"source": DEMO_SOURCE},
                )
            except Exception as exc:
                raise explain_agent_memory_error("session event write", exc)

            extraction = self.extractor.invoke(
                [
                    SystemMessage(
                        content=(
                            "Extract only durable user facts, persistent preferences, and stable constraints "
                            "that the user explicitly states in the current message and that should help in future "
                            "unrelated sessions. Do not extract active task details, current itinerary details, "
                            "dates, destinations, booking requests, or other context that only matters for this "
                            "conversation unless the user explicitly asks to remember it for later. Do not extract "
                            "anything that is only mentioned by the assistant or already present in existing "
                            "long-term memories."
                        )
                    ),
                    HumanMessage(
                        content=(
                            f"Current user message:\n{user_text}\n\n"
                            "Existing long-term memories:\n"
                            + "\n".join(f"- {memory}" for memory in state["recalled_memories"])
                        )
                    ),
                ]
            )

            records = []
            extracted_texts = []
            known_memory_texts = {
                normalize_memory_text(memory)
                for memory in state["recalled_memories"]
            }
            accepted_memory_texts = set(known_memory_texts)
            for memory in extraction.memories:
                text = memory.text.strip()
                if not text:
                    continue
                normalized_text = normalize_memory_text(text)
                if not normalized_text or normalized_text in accepted_memory_texts:
                    continue
                accepted_memory_texts.add(normalized_text)
                record_id = memory_id(state["owner_id"], state["namespace"], text)
                extracted_texts.append(text)
                records.append(
                    {
                        "id": record_id,
                        "text": text,
                        "ownerId": state["owner_id"],
                        "namespace": state["namespace"],
                        "sessionId": state["session_id"],
                        "topics": memory.topics or ["travel"],
                        "memoryType": memory.memory_type,
                    }
                )

            if records:
                try:
                    agent_memory.bulk_create_long_term_memories(memories=records)
                except Exception as exc:
                    raise explain_agent_memory_error("long-term memory write", exc)
            return {"extracted_memories": extracted_texts}

        builder = StateGraph(AgentState)
        builder.add_node("retrieve_session_context", retrieve_session_context)
        builder.add_node("retrieve_long_term_memories", retrieve_long_term_memories)
        builder.add_node("call_model", call_model)
        builder.add_node("write_memory", write_memory)
        builder.add_edge(START, "retrieve_session_context")
        builder.add_edge("retrieve_session_context", "retrieve_long_term_memories")
        builder.add_edge("retrieve_long_term_memories", "call_model")
        builder.add_edge("call_model", "write_memory")
        builder.add_edge("write_memory", END)
        return builder.compile()

    def run_turn(self, agent_memory: AgentMemory, session_id: str, user_text: str) -> TurnResult:
        graph = self.build_graph(agent_memory)
        state = graph.invoke(
            {
                "messages": [HumanMessage(content=user_text)],
                "owner_id": self.config.owner_id,
                "session_id": session_id,
                "namespace": self.config.namespace,
                "session_context": [],
                "recalled_memories": [],
                "extracted_memories": [],
            }
        )
        assistant_message = next(
            (message for message in reversed(state["messages"]) if isinstance(message, AIMessage)),
            None,
        )
        return TurnResult(
            session_id=session_id,
            user_text=user_text,
            assistant_text=message_text(assistant_message) if assistant_message else "",
            session_context=state["session_context"],
            long_term_memories=state["recalled_memories"],
            extracted_memories=state["extracted_memories"],
        )

    def read_session_context(self, agent_memory: AgentMemory, session_id: str) -> list[str]:
        try:
            response = agent_memory.get_session_memory(session_id=session_id)
        except Exception as exc:
            if is_not_found_error(exc):
                return []
            raise explain_agent_memory_error("session memory read", exc)

        session_context = []
        for event in coerce_events(response)[-SESSION_CONTEXT_LIMIT:]:
            text = get_event_text(event).strip()
            if text:
                session_context.append(f"{get_event_role(event)}: {text}")
        return session_context

    def delete_session_memory(self, agent_memory: AgentMemory, session_id: str) -> None:
        try:
            agent_memory.delete_session_memory(session_id=session_id)
        except Exception as exc:
            if not is_not_found_error(exc):
                raise explain_agent_memory_error("session memory delete", exc)
