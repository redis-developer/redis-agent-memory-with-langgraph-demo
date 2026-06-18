# UK Mortgage Assistant — Redis Agent Memory + LangGraph Demo

## Overview

This demo is a **UK mortgage assistant** ("Albion Home Loans") that shows the value of agent memory: an assistant that remembers a customer across *separate conversations*, so they never have to re-explain their income, deposit, or preferences.

It combines two Redis-backed layers:

| Layer | Backed by | Role |
|:--|:--|:--|
| **Thread / session log** | [`langgraph-redis`](https://pypi.org/project/langgraph-checkpoint-redis/) `RedisSaver` checkpointer, keyed by `thread_id` | Durable per-conversation message history. Powers the **chat-thread selector** on the left — list, switch, and resume conversations. |
| **Agent memory** | [`redis-agent-memory`](https://pypi.org/project/redis-agent-memory/) | **STM** = working-memory context for the active turn. **LTM** = durable borrower facts shared across *every* thread. |

The punchline: switch to a different thread (a separate `langgraph-redis` log) and the assistant still knows the borrower (shared `redis-agent-memory` long-term memory).

## Table of Contents

- [Demo Objectives](#demo-objectives)
- [Setup](#setup)
- [Seeding the Demo](#seeding-the-demo)
- [Running the Demo](#running-the-demo)
- [Architecture](#architecture)
- [Running the Tests](#running-the-tests)
- [Known Issues](#known-issues)
- [Resources](#resources)
- [License](#license)

## Demo Objectives

- Show Redis as the memory and persistence layer for an agentic application.
- Use **`langgraph-redis`** as the durable thread log behind a multi-conversation chat selector.
- Use **`redis-agent-memory`** for short-term (working) memory and durable long-term memory.
- Demonstrate the core value of agent memory: **persistent, personalized experience across separate conversations** — state a fact once, and the assistant recalls it in a brand-new chat.
- Keep transient conversation details in STM while promoting only durable borrower facts (income, deposit, first-time-buyer status, target area/price, product preferences) into LTM.

## Setup

### Dependencies

- [Docker](https://docs.docker.com/get-docker/) for running the web UI
- A **Redis** database with the Query Engine + JSON modules (Redis 8 or Redis Cloud) for the `langgraph-redis` checkpointer
- A [Redis Agent Memory](https://redis.io/try-free) service (data-plane URL, store ID, API key)
- An [OpenAI API key](https://platform.openai.com/api-keys)

### Configuration

1. Clone the repository and create your environment file:

   ```sh
   cp .env.example .env
   ```

2. Edit `.env`:

| Variable | Required | Description |
|:--|:--:|:--|
| `OPENAI_API_KEY` | Yes | API key used by the LangGraph agent. |
| `AGENT_MEMORY_SERVER_URL` | Yes | Agent Memory Server data-plane base URL. |
| `AGENT_MEMORY_STORE_ID` | Yes | Store ID used by the Agent Memory Server API. |
| `AGENT_MEMORY_API_KEY` | Yes | API key used by the Agent Memory Server API. |
| `REDIS_URL` | Yes | Full Redis connection string for the `langgraph-redis` thread log, e.g. `redis://default:<password>@<host>:<port>` (use `rediss://` for TLS). |
| `OPENAI_MODEL` | No | OpenAI model used for responses and memory extraction. |
| `DEMO_OWNER_ID` | No | Stable borrower identifier for long-term memories (default `jordan-uk`). |
| `DEMO_NAMESPACE` | No | Logical namespace for this demo's memories (default `mortgage-demo`). |
| `DEMO_AGENT_ID` | No | Actor ID used when writing assistant session events (default `mortgage-adviser`). |

3. Build and run the app:

   ```sh
   docker compose up --build
   ```

The backend connects to `REDIS_URL` on startup and sets up the `RedisSaver` indexes automatically.

## Seeding the Demo

To open with a populated sidebar, run the seed script once after the services are up:

```sh
uv run python scripts/seed_demo.py
```

This replays **two short, realistic first-time-buyer conversations** that establish *part* of the borrower profile — who they are plus household income, then deposit and target property. The rest of the profile is added live during the demo (see the demo script below). Each conversation's durable facts are promoted to long-term memory.

To wipe and re-seed (handy between live runs):

```sh
uv run python scripts/seed_demo.py --reset      # reset then seed
uv run python scripts/seed_demo.py --reset-only # just wipe demo state
```

## Running the Demo

Open `http://localhost:8080`.

- **Left pane** — the chat-thread selector. Create a new conversation, or click any thread to resume its full history (loaded from `langgraph-redis`).
- **Centre** — the chat with the mortgage assistant.
- **Right pane** — the live Redis Agent Memory view: **STM** (working memory this turn), **LTM** (durable borrower facts retrieved), and newly **Extracted** long-term memory.

### Demo script (three-step arc)

The demo builds up a borrower profile across conversations: seeded history → a new chat that uses some of it and adds more → a final new chat that synthesises everything.

**Step 1 — Show the history (seeded; just click).** Click both sidebar threads and read them out:
- *First-time buyer affordability* — "We're first-time buyers in Bristol… combined income around £92,000 — how much could we borrow?"
- *Deposit size* — "We've saved about £21,000… viewing flats around £340,000. Is that a big enough deposit?"

These are two separate conversations (distinct `langgraph-redis` threads) whose facts were promoted into long-term memory. (The **LTM** panel is empty when viewing the seeded threads — those turns *created* the memories, so there was nothing to retrieve yet; it populates on the live turns below.)

**Step 2 — New chat: uses the history, then adds more.** Click **＋ New conversation**:
> "We're hoping to start making offers soon. What should we have sorted before we apply for a mortgage?"

The **LTM** panel fills and the answer is tailored to the borrower without restating anything. Then, in the same chat, add context:
> "We're both on permanent contracts with no other debts, and we'd want a five-year fixed rate. Does that change anything?"

Watch the **Extracted Long-Term Memory** panel promote the two new facts.

**Step 3 — Another new chat: the synthesis.** Click **＋ New conversation**:
> "What mortgage would you recommend for us, and roughly what would the monthly repayments be?"

A normal fresh-chat question — yet the **LTM** panel shows *all* the accumulated facts and the reply synthesises them (LTV, income, five-year fixed, a monthly estimate), none of it restated here. This is the "it already knows this customer, across every conversation" moment.

### Optional beats

- **STM vs LTM** — drop a transient detail ("the flat I'm looking at is on Oak Lane"), confirm the assistant recalls it, click **Clear STM**, then ask again: it forgets the in-chat detail but still knows the durable profile (LTM).
- **"Won't it get stuck on an old budget?"** — be precise here. Today the levers are: working memory expires on a TTL (the *Clear STM* beat); duplicate facts are prevented via content-hash deduplication; and long-term memories can be **updated, expired, or deleted via the API**. The full Agent Memory Server additionally offers policy-based **forgetting/decay** (age, inactivity, budget) and **recency/freshness-weighted retrieval**. What the product does **not** currently do on its own is automatically detect that a newer fact *semantically contradicts* an older one (e.g. a £500k budget superseding £340k) and retire the stale one — that's developer-managed (update/delete) or a roadmap question to confirm with Product, not an automatic behaviour. Do not stage the budget change as an automatic "watch it forget" moment.
- **Persistence** — reload the page; the threads are still there (server-side in Redis), proving they are not just client state.

After a live run, reset to the clean two-conversation baseline with `uv run python scripts/seed_demo.py --reset`.

### Backend endpoints

| Endpoint | Purpose |
|:--|:--|
| `GET /api/threads` | List conversations for the borrower (most recent first). |
| `POST /api/threads` | Create a new conversation. |
| `GET /api/threads/{id}/messages` | Full message history for a thread (from `langgraph-redis`). |
| `GET /api/threads/{id}/memory` | Read the thread's short-term (working) memory. |
| `DELETE /api/threads/{id}/memory` | Clear the thread's short-term memory. |
| `POST /api/chat` | Run one agent turn (`{thread_id, message}`). |
| `GET /api/health` | Backend liveness check. |
| `GET /api/ready` | Backend readiness check, including Redis Agent Memory. |

## Architecture

Each agent turn is a small LangGraph graph. `langgraph-redis` persists the conversation per `thread_id`; `redis-agent-memory` provides STM and LTM:

1. Load the thread's prior messages from the `RedisSaver` checkpointer (keyed by `thread_id`).
2. Retrieve the current session's short-term memory from Redis Agent Memory.
3. Retrieve relevant long-term memories for the borrower from Redis Agent Memory.
4. Inject both memory contexts into the OpenAI system prompt and generate the response.
5. Write the user and assistant messages as session events.
6. Extract durable borrower facts, filter out anything already in LTM, and write the rest back to Redis Agent Memory.
7. Update the thread index (title + last-active) so the sidebar stays current.

## Running the Tests

The test suite requires no external services — no Redis connection, no OpenAI key. Network calls are mocked and the thread index is tested against `fakeredis`.

```sh
uv sync
uv run pytest
```

Tests under `tests/`:

- `test_utils.py` — pure utility functions
- `test_api.py` — all FastAPI endpoints via `TestClient`
- `test_service.py` — `RedisAgentMemoryService` methods, including LTM deduplication
- `test_threads.py` — the `ThreadIndex` registry (via `fakeredis`)

## Known Issues

- Requires a reachable Agent Memory Server data-plane endpoint and a Redis database with the Query Engine + JSON modules.
- Memory extraction is performed by the LLM, so phrasing can vary between runs.
- The assistant provides general guidance only and is not a regulated financial-advice tool.
- `--reset` is scoped to this demo's own threads (it deletes only those threads' checkpoints, the thread index, and the borrower's long-term memories), so it is safe against a shared Redis database — though a dedicated database is still recommended.

## Resources

- [Redis Agent Memory](https://pypi.org/project/redis-agent-memory/)
- [langgraph-redis (langgraph-checkpoint-redis)](https://pypi.org/project/langgraph-checkpoint-redis/)
- [LangGraph documentation](https://langchain-ai.github.io/langgraph/)
- [OpenAI API documentation](https://platform.openai.com/docs)

## License

This project is licensed under the MIT License.
