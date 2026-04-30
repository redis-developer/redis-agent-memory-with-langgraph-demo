from __future__ import annotations

import argparse
import hashlib
import os
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
from redis_agent_memory import AgentMemory, models
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from typing_extensions import TypedDict


console = Console()
DEMO_SOURCE = "langgraph-demo"


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
    recalled_memories: list[str]


@dataclass(frozen=True)
class DemoConfig:
    openai_model: str
    agent_memory_server_url: str
    agent_memory_store_id: str
    agent_memory_api_key: str
    owner_id: str
    namespace: str
    agent_id: str


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


def new_session_id() -> str:
    return f"interactive-{uuid.uuid4().hex[:8]}"


def coerce_memories(response: object) -> list[object]:
    memories = getattr(response, "memories", None)
    if memories is None and isinstance(response, dict):
        memories = response.get("memories")
    return list(memories or [])


def get_memory_text(memory: object) -> str:
    if isinstance(memory, dict):
        return str(memory.get("text", ""))
    return str(getattr(memory, "text", ""))


def explain_agent_memory_error(operation: str, exc: Exception) -> RuntimeError:
    hint = (
        f"Redis Agent Memory {operation} failed. Check AGENT_MEMORY_SERVER_URL, "
        "AGENT_MEMORY_STORE_ID, and AGENT_MEMORY_API_KEY. The server URL should be "
        "the Agent Memory data-plane base URL, not the PyPI/docs URL."
    )
    return RuntimeError(f"{hint}\n\nOriginal error: {exc}")


class RedisAgentMemoryLangGraphDemo:
    def __init__(self, config: DemoConfig) -> None:
        self.config = config
        self.llm = ChatOpenAI(model=config.openai_model, temperature=0.2)
        self.extractor = self.llm.with_structured_output(MemoryExtraction)

    def build_graph(self, agent_memory: AgentMemory):
        def retrieve_memories(state: AgentState) -> dict:
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
            show_memories("Recalled from Redis Agent Memory", recalled)
            return {"recalled_memories": recalled}

        def call_model(state: AgentState) -> dict:
            memory_context = "\n".join(f"- {memory}" for memory in state["recalled_memories"])
            if not memory_context:
                memory_context = "- No relevant long-term memories found."

            system_prompt = f"""You are a polished travel concierge.

Use the relevant long-term memories when they help, but do not mention implementation details.
Keep answers concise, specific, and naturally personalized.

Relevant long-term memories:
{memory_context}
"""
            response = self.llm.invoke([SystemMessage(content=system_prompt), *state["messages"]])
            console.print(Panel(message_text(response), title="Assistant", border_style="green"))
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
                return {}

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
                            "Extract only durable user facts, preferences, constraints, "
                            "or stable travel context worth remembering for future sessions. "
                            "Skip transient requests and anything already implied by the assistant."
                        )
                    ),
                    HumanMessage(content=f"User: {user_text}\nAssistant: {assistant_text}"),
                ]
            )

            records = []
            extracted_texts = []
            for memory in extraction.memories:
                text = memory.text.strip()
                if not text:
                    continue
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
            show_memories("Written as Long-Term Memory", extracted_texts)
            return {}

        builder = StateGraph(AgentState)
        builder.add_node("retrieve_memories", retrieve_memories)
        builder.add_node("call_model", call_model)
        builder.add_node("write_memory", write_memory)
        builder.add_edge(START, "retrieve_memories")
        builder.add_edge("retrieve_memories", "call_model")
        builder.add_edge("call_model", "write_memory")
        builder.add_edge("write_memory", END)
        return builder.compile()

    def ask(self, graph, session_id: str, user_text: str) -> AgentState:
        console.print(Panel(user_text, title=f"User ({session_id})", border_style="cyan"))
        return graph.invoke(
            {
                "messages": [HumanMessage(content=user_text)],
                "owner_id": self.config.owner_id,
                "session_id": session_id,
                "namespace": self.config.namespace,
                "recalled_memories": [],
            }
        )

    def run_interactive(self, session_id: str | None) -> None:
        session = session_id or new_session_id()
        with AgentMemory(
            self.config.agent_memory_server_url,
            store_id=self.config.agent_memory_store_id,
            api_key=self.config.agent_memory_api_key,
        ) as agent_memory:
            graph = self.build_graph(agent_memory)
            console.rule("[bold]Redis Agent Memory + LangGraph Interactive Demo[/bold]")
            show_demo_context(self.config, session)
            console.print(
                "Type a message for the agent. Commands: [bold]/new[/bold] starts a fresh session, "
                "[bold]/where[/bold] shows Redis Insight lookup values, [bold]/quit[/bold] exits."
            )
            while True:
                try:
                    user_text = input("\nYou> ").strip()
                except EOFError:
                    break
                if user_text.lower() in {"quit", "exit", "/quit", "/exit"}:
                    break
                if user_text == "/new":
                    session = new_session_id()
                    console.print(Panel(f"Started fresh session: {session}", border_style="blue"))
                    show_demo_context(self.config, session)
                    continue
                if user_text == "/where":
                    show_demo_context(self.config, session)
                    continue
                if user_text:
                    self.ask(graph, session, user_text)


def show_memories(title: str, memories: list[str]) -> None:
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=4)
    table.add_column("Memory")
    if not memories:
        table.add_row("-", "None")
    else:
        for index, memory in enumerate(memories, start=1):
            table.add_row(str(index), memory)
    console.print(table)


def show_demo_context(config: DemoConfig, session_id: str) -> None:
    table = Table(title="Redis Insight Lookup Values", show_header=True, header_style="bold cyan")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("store_id", config.agent_memory_store_id)
    table.add_row("owner_id", config.owner_id)
    table.add_row("namespace", config.namespace)
    table.add_row("session_id", session_id)
    console.print(table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Redis Agent Memory + LangGraph demo.")
    parser.add_argument("--session-id", help="Session ID to use in interactive mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    demo = RedisAgentMemoryLangGraphDemo(config)
    demo.run_interactive(args.session_id)


if __name__ == "__main__":
    main()
