from __future__ import annotations

import hashlib
import os
import re
import time
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Annotated, Literal

import redis
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.redis import RedisSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from redis_agent_memory import AgentMemory, errors, models
from typing_extensions import TypedDict


DEMO_SOURCE = "langgraph-demo"
SESSION_CONTEXT_LIMIT = 12

# Prefix for every Redis key this demo owns (thread index + checkpointer share
# one Redis Cloud database, so namespacing keeps it tidy and resettable).
THREAD_INDEX_PREFIX = "demo:threads"
THREAD_META_PREFIX = "demo:thread"
DEFAULT_THREAD_TITLE = "New conversation"


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
    redis_url: str
    owner_id: str
    namespace: str
    agent_id: str


@dataclass(frozen=True)
class TurnResult:
    thread_id: str
    title: str
    user_text: str
    assistant_text: str
    session_context: list[str]
    long_term_memories: list[str]
    extracted_memories: list[str]


@dataclass(frozen=True)
class ThreadSummary:
    thread_id: str
    title: str
    preview: str
    created_at: float
    last_active: float


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
        redis_url=require_env("REDIS_URL"),
        owner_id=os.getenv("DEMO_OWNER_ID", "jordan-uk"),
        namespace=os.getenv("DEMO_NAMESPACE", "mortgage-demo"),
        agent_id=os.getenv("DEMO_AGENT_ID", "mortgage-adviser"),
    )


def now() -> datetime:
    return datetime.now(timezone.utc)


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


def new_thread_id() -> str:
    return f"thread-{uuid.uuid4().hex[:8]}"


class ThreadIndex:
    """Lightweight registry of chat threads, stored in the same Redis that backs
    the LangGraph checkpointer.

    ``RedisSaver`` persists each thread's message history but has no first-class
    "list every thread" API, so we keep a tiny index alongside it:

    - a sorted set ``demo:threads:{owner_id}`` ordered by last-active time, and
    - a hash ``demo:thread:{thread_id}`` holding the display metadata.
    """

    def __init__(self, client: "redis.Redis") -> None:
        self.client = client

    def _index_key(self, owner_id: str) -> str:
        return f"{THREAD_INDEX_PREFIX}:{owner_id}"

    def _meta_key(self, thread_id: str) -> str:
        return f"{THREAD_META_PREFIX}:{thread_id}"

    def create(self, owner_id: str, title: str = DEFAULT_THREAD_TITLE) -> ThreadSummary:
        thread_id = new_thread_id()
        timestamp = time.time()
        self.client.hset(
            self._meta_key(thread_id),
            mapping={
                "title": title,
                "preview": "",
                "created_at": timestamp,
                "last_active": timestamp,
            },
        )
        self.client.zadd(self._index_key(owner_id), {thread_id: timestamp})
        return ThreadSummary(thread_id, title, "", timestamp, timestamp)

    def touch(
        self,
        owner_id: str,
        thread_id: str,
        *,
        preview: str | None = None,
        title: str | None = None,
    ) -> None:
        """Record activity on a thread, creating its metadata if needed."""
        meta_key = self._meta_key(thread_id)
        timestamp = time.time()
        mapping: dict[str, object] = {"last_active": timestamp}
        if not self.client.exists(meta_key):
            mapping["created_at"] = timestamp
            mapping.setdefault("title", title or DEFAULT_THREAD_TITLE)
        if preview is not None:
            mapping["preview"] = preview[:140]
        if title is not None:
            mapping["title"] = title
        self.client.hset(meta_key, mapping=mapping)
        self.client.zadd(self._index_key(owner_id), {thread_id: timestamp})

    def set_title(self, thread_id: str, title: str) -> None:
        self.client.hset(self._meta_key(thread_id), "title", title)

    def get(self, thread_id: str) -> ThreadSummary | None:
        meta = self.client.hgetall(self._meta_key(thread_id))
        if not meta:
            return None
        return ThreadSummary(
            thread_id=thread_id,
            title=meta.get("title", DEFAULT_THREAD_TITLE),
            preview=meta.get("preview", ""),
            created_at=float(meta.get("created_at", 0) or 0),
            last_active=float(meta.get("last_active", 0) or 0),
        )

    def list(self, owner_id: str) -> list[ThreadSummary]:
        thread_ids = self.client.zrevrange(self._index_key(owner_id), 0, -1)
        summaries = []
        for thread_id in thread_ids:
            summary = self.get(thread_id)
            if summary is not None:
                summaries.append(summary)
        return summaries

    def reset(self, owner_id: str) -> None:
        """Delete every thread for an owner (used by the demo reseed script)."""
        for thread_id in self.client.zrange(self._index_key(owner_id), 0, -1):
            self.client.delete(self._meta_key(thread_id))
        self.client.delete(self._index_key(owner_id))


def coerce_memories(response: object) -> list[object]:
    if isinstance(response, dict):
        return list(response.get("items", response.get("memories", [])) or [])
    items = getattr(response, "items", None)
    if items is not None:
        return list(items)
    return []


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
    def __init__(
        self,
        config: DemoConfig,
        checkpointer: "RedisSaver | None" = None,
        thread_index: "ThreadIndex | None" = None,
    ) -> None:
        self.config = config
        self.llm = ChatOpenAI(model=config.openai_model, temperature=0.2)
        self.extractor = self.llm.with_structured_output(MemoryExtraction)
        # In production these are wired to Redis Cloud (see build_service). In
        # unit tests they stay None: the graph compiles without a checkpointer
        # and the thread index is simply skipped, so no Redis connection is made.
        self.checkpointer = checkpointer
        self.thread_index = thread_index

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
                        # Retrieve a broad slice so a synthesis question can draw
                        # on the full accumulated borrower profile, not just the
                        # top few semantically-closest facts.
                        "limit": 8,
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

            system_prompt = f"""You are a UK mortgage assistant for Albion Home Loans.

You help customers through their mortgage journey: how much they can borrow,
deposits and loan-to-value (LTV), Agreement in Principle, fixed-rate terms,
remortgaging and product transfers, stamp duty, and first-time-buyer options.

Style rules (important):
- Reply in 2-4 short sentences of plain conversational prose.
- Plain text ONLY. Do NOT use Markdown, asterisks for bold, bullet points, or
  numbered lists. Write naturally, as if speaking.
- Use UK terminology and pounds (£). Be warm and direct.
- Give general guidance; add a brief reminder that this is not regulated advice
  only when the customer asks for a recommendation, not on every reply.

Use short-term memory for continuity within the current conversation.
Use long-term memory for durable facts about the customer (income, deposit,
first-time-buyer status, target area and price, and product preferences) so you
never ask them to repeat themselves across conversations. Personalize naturally.

Short-term memory from this conversation:
{session_context}

Relevant long-term memories about this customer:
{long_term_context}
"""
            # The assistant's working memory of the conversation comes from the
            # short-term-memory block above (sourced from redis-agent-memory), not
            # from replaying the checkpointer's full message history. We therefore
            # send only the current user turn. This keeps the three memory layers
            # distinct and makes "Clear STM" actually make the agent forget the
            # recent conversation, while the thread transcript (langgraph-redis)
            # and long-term memory remain intact.
            current_user_message = next(
                (message for message in reversed(state["messages"]) if isinstance(message, HumanMessage)),
                None,
            )
            turn_messages = [current_user_message] if current_user_message is not None else []
            response = self.llm.invoke([SystemMessage(content=system_prompt), *turn_messages])
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
                    created_at=now(),
                    metadata={"source": DEMO_SOURCE},
                )
                agent_memory.add_session_event(
                    session_id=state["session_id"],
                    actor_id=self.config.agent_id,
                    role=models.MessageRole.ASSISTANT,
                    content=[{"text": assistant_text}],
                    created_at=now(),
                    metadata={"source": DEMO_SOURCE},
                )
            except Exception as exc:
                raise explain_agent_memory_error("session event write", exc)

            extraction = self.extractor.invoke(
                [
                    SystemMessage(
                        content=(
                            "Extract only durable facts about the mortgage customer, persistent preferences, "
                            "and stable financial circumstances that the user explicitly states in the current "
                            "message and that should help in future, separate conversations. Useful durable facts "
                            "include household income, deposit saved, first-time-buyer status, target property area "
                            "and price range, and product preferences (e.g. a 5-year fixed term). Do not extract "
                            "transient task details such as a specific property address they are browsing right now, "
                            "one-off questions, dates, or anything that only matters for this conversation unless the "
                            "user explicitly asks you to remember it. Do not extract anything that is only mentioned "
                            "by the assistant or is already present in existing long-term memories. "
                            "If the message is a short reply, a confirmation, a single word, a number, or only "
                            "makes sense in the context of the current conversation, return an empty list.\n\n"
                            "Examples of messages that should produce NO memories:\n"
                            "- 'yes' (a confirmation)\n"
                            "- 'no', 'ok', 'sure', 'sounds good' (short replies)\n"
                            "- '£340,000' (a number answering a question)\n"
                            "- 'next Tuesday' (a date answering a question)\n"
                            "- 'what about 14 Elm Road?' (a transient question about one property)\n\n"
                            "Examples of messages that SHOULD produce memories:\n"
                            "- 'My name is Priya' → 'The customer's name is Priya.'\n"
                            "- 'We're first-time buyers' → 'The customer is a first-time buyer.'\n"
                            "- 'Our household income is £92,000 a year' → 'The customer's household income is £92,000 per year.'\n"
                            "- 'We've saved £21,000 for a deposit' → 'The customer has a deposit of £21,000 saved.'\n"
                            "- 'We're looking at flats around £340k in Bristol' → 'The customer is looking for a property around £340,000 in Bristol.'\n"
                            "- 'I'd prefer a 5-year fixed rate' → 'The customer prefers a 5-year fixed-rate mortgage.'\n"
                            "- 'yes, remember I'm self-employed' → 'The customer is self-employed.' (extract the confirmed fact, not the confirmation itself)\n"
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
                        "topics": memory.topics or ["mortgage"],
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
        # The checkpointer (RedisSaver) is what persists each thread's message
        # history to Redis, keyed by thread_id, so threads can be listed and
        # resumed. Without one (unit tests) the graph still runs, just in-memory.
        return builder.compile(checkpointer=self.checkpointer)

    def run_turn(self, agent_memory: AgentMemory, thread_id: str, user_text: str) -> TurnResult:
        # Does this thread still need an auto-generated title? A thread may be
        # pre-created (via create_thread) with the default placeholder title, so
        # key off the title itself rather than mere existence in the index.
        existing = self.thread_index.get(thread_id) if self.thread_index is not None else None
        needs_title = self.thread_index is not None and (
            existing is None or not existing.title or existing.title == DEFAULT_THREAD_TITLE
        )

        graph = self.build_graph(agent_memory)
        # thread_id drives the RedisSaver checkpointer: prior messages for this
        # thread are loaded and the new one is appended via the add_messages
        # reducer. We only pass the new HumanMessage, not the whole history.
        state = graph.invoke(
            {
                "messages": [HumanMessage(content=user_text)],
                "owner_id": self.config.owner_id,
                "session_id": thread_id,
                "namespace": self.config.namespace,
                "session_context": [],
                "recalled_memories": [],
                "extracted_memories": [],
            },
            config={"configurable": {"thread_id": thread_id}},
        )
        assistant_message = next(
            (message for message in reversed(state["messages"]) if isinstance(message, AIMessage)),
            None,
        )
        assistant_text = message_text(assistant_message) if assistant_message else ""

        title = DEFAULT_THREAD_TITLE
        if self.thread_index is not None:
            title = self.generate_title(user_text) if needs_title else existing.title
            self.thread_index.touch(
                self.config.owner_id,
                thread_id,
                preview=user_text,
                title=title if needs_title else None,
            )

        return TurnResult(
            thread_id=thread_id,
            title=title,
            user_text=user_text,
            assistant_text=assistant_text,
            session_context=state["session_context"],
            long_term_memories=state["recalled_memories"],
            extracted_memories=state["extracted_memories"],
        )

    def generate_title(self, first_message: str) -> str:
        """Produce a short, human-readable thread title from the first message.

        Uses the existing LLM for a clean label; falls back to a truncated
        version of the message if the call fails.
        """
        try:
            response = self.llm.invoke(
                [
                    SystemMessage(
                        content=(
                            "Write a concise 3-6 word title (no quotes, no trailing "
                            "punctuation) summarising the topic of this mortgage "
                            "assistant message. Reply with the title only."
                        )
                    ),
                    HumanMessage(content=first_message),
                ]
            )
            title = message_text(response).strip().strip('"').strip()
            if title:
                return title[:60]
        except Exception:
            pass
        fallback = first_message.strip().splitlines()[0] if first_message.strip() else DEFAULT_THREAD_TITLE
        return (fallback[:48] + "…") if len(fallback) > 48 else fallback or DEFAULT_THREAD_TITLE

    def list_threads(self, owner_id: str | None = None) -> list[ThreadSummary]:
        if self.thread_index is None:
            return []
        return self.thread_index.list(owner_id or self.config.owner_id)

    def create_thread(self, owner_id: str | None = None) -> ThreadSummary:
        if self.thread_index is None:
            return ThreadSummary(new_thread_id(), DEFAULT_THREAD_TITLE, "", time.time(), time.time())
        return self.thread_index.create(owner_id or self.config.owner_id)

    def read_thread_messages(self, agent_memory: AgentMemory, thread_id: str) -> list[dict]:
        """Return the full persisted message history for a thread (for resuming
        it in the UI). Reads straight from the RedisSaver checkpoint."""
        if self.checkpointer is None:
            return []
        graph = self.build_graph(agent_memory)
        snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
        messages = (snapshot.values or {}).get("messages", []) if snapshot else []
        history = []
        for message in messages:
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            else:
                continue
            text = message_text(message).strip()
            if text:
                history.append({"role": role, "text": text})
        return history

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


def build_service(config: DemoConfig | None = None) -> RedisAgentMemoryService:
    """Construct a fully wired service for production use.

    Connects to Redis Cloud, sets up the LangGraph RedisSaver checkpointer
    (which creates its search indexes on first run), and builds the thread
    index that backs the chat selector.
    """
    config = config or load_config()
    redis_client = redis.Redis.from_url(config.redis_url, decode_responses=True)
    checkpointer = RedisSaver(redis_url=config.redis_url)
    checkpointer.setup()
    thread_index = ThreadIndex(redis_client)
    return RedisAgentMemoryService(config, checkpointer=checkpointer, thread_index=thread_index)
