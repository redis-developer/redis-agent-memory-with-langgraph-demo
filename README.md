# Redis Agent Memory with LangGraph Demo

## Overview

This demo demonstrates how [Redis Agent Memory](https://pypi.org/project/redis-agent-memory/) can add durable memory to a LangGraph agent. Built with Python, LangGraph, OpenAI, and the `redis-agent-memory` Python client, it shows how an agent can remember user facts and preferences across sessions while keeping the interaction simple enough for a user walkthrough.

The demo runs as an interactive terminal assistant. You type messages into the agent, observe how memories are stored and retrieved, inspect data using Redis Insight when useful, start a fresh session, and then ask follow-up questions that rely on long-term memory.

## Table of Contents

- [Demo Objectives](#demo-objectives)
- [Setup](#setup)
- [Running the Demo](#running-the-demo)
- [Architecture](#architecture)
- [Known Issues](#known-issues)
- [Resources](#resources)
- [Maintainers](#maintainers)
- [License](#license)

## Demo Objectives

- Demonstrate Redis as a memory persistence layer for agentic applications.
- Show how to integrate Redis Agent Memory through the Python client.
- Illustrate the LangGraph pattern of retrieving memory before an LLM call and writing memory after a turn.
- Show the difference between a session conversation and durable long-term memory.
- Provide a simple way to verify your Redis Agent Memory service endpoint.

## Setup

### Dependencies

- [Docker](https://docs.docker.com/get-docker/) for Docker-based runs
- [uv](https://docs.astral.sh/uv/) for local Python runs
- [Redis Agent Memory Server](https://redis.github.io/agent-memory-server)
- [Redis Insight](https://redis.io/insight/) for optional memory inspection
- [OpenAI API key](https://platform.openai.com/api-keys)

### Account Requirements

| Account                                          | Description                                                    |
|:-------------------------------------------------|:---------------------------------------------------------------|
| [OpenAI](https://auth.openai.com/create-account) | LLM used to generate assistant responses and extract memories. |
| [Redis Agent Memory](https://redis.io/try-free)  | Fully managed service for agent memory backed by Redis Cloud.  |

This demo does not deploy Redis Agent Memory Server. Before running the demo, make sure you have an Agent Memory Server data-plane URL, store ID, and API key.

### Configuration

#### Docker Setup

1. Clone the repository:

   ```sh
   git clone <repository-url>
   cd redis-agent-memory-with-langgraph-demo
   ```

2. Create your environment file:

   ```sh
   cp .env.example .env
   ```

3. Edit `.env` with your configuration:

| Variable                    | Required | Description                                                   |
|:----------------------------|:--------:|:--------------------------------------------------------------|
| `OPENAI_API_KEY`            | Yes      | API key used by the LangGraph agent.                          |
| `AGENT_MEMORY_SERVER_URL`   | Yes      | Agent Memory Server data-plane base URL.                      |
| `AGENT_MEMORY_STORE_ID`     | Yes      | Store ID used by the Agent Memory Server API.                 |
| `AGENT_MEMORY_API_KEY`      | Yes      | API key used by the Agent Memory Server API.                  |
| `OPENAI_MODEL`              | No       | OpenAI model used for responses and memory extraction.        |
| `DEMO_OWNER_ID`             | No       | Stable user identifier for long-term memories.                |
| `DEMO_NAMESPACE`            | No       | Logical namespace for this demo's memories.                   |
| `DEMO_AGENT_ID`             | No       | Actor ID used when writing assistant session events.          |

4. Build and run the interactive demo:

   ```sh
   docker compose run --rm demo
   ```

#### Local Python Setup

If you prefer to run the demo without Docker, use the checked-in lockfile:

```sh
cp .env.example .env
uv sync --locked
uv run python demo.py
```

Edit `.env` before starting the demo.

If you previously created `.venv` with a different Python version and see an error like `ModuleNotFoundError: No module named 'encodings'`, recreate the environment:

```sh
deactivate 2>/dev/null || true
rm -rf .venv
uv sync --locked
uv run python demo.py
```

## Running the Demo

Start the interactive assistant:

```sh
docker compose run --rm demo
```

If you installed the demo locally with Python, use:

```sh
uv run python demo.py
```

The demo opens an interactive prompt:

```text
You>
```

### Examples of Interactions

- "My name is Ricardo."
- "Remember that I prefer short answers."
- "I like vegetarian restaurants, but I do not like cilantro."
- "I am planning a trip to Lisbon next month."
- `/new`
- "Fresh session: what do you remember about me?"
- "Can you recommend a dinner plan for Lisbon?"

### Useful Commands

- `/new` starts a fresh session while keeping the same long-term memory owner.
- `/where` prints the `store_id`, `owner_id`, `namespace`, and current `session_id`.
- `/quit` exits the demo.

### Suggested Recording Flow

1. Start the demo with `uv run python demo.py`.
2. Tell the agent a durable fact or preference.
3. Inspect Redis Insight to show the new long-term memory.
4. Type `/new` to start a fresh session.
5. Ask the agent a question that requires the previous memory.
6. Inspect Redis Insight again to show the memory being reused.

## Architecture

The demo uses LangGraph to model one agent turn as a small graph:

1. Retrieve relevant long-term memories from Redis Agent Memory.
2. Inject those memories into the OpenAI system prompt.
3. Generate the assistant response.
4. Write the user and assistant messages as session events.
5. Extract durable facts and preferences from the turn.
6. Write extracted memories back to Redis Agent Memory.

![Redis Agent Memory with LangGraph architecture](images/architecture-diagram.png)

## Known Issues

- The demo requires a reachable Agent Memory Server data-plane endpoint.
- Memory extraction is performed by the LLM, so phrasing can vary between runs.
- Re-running the same durable fact may create the same deterministic memory ID and depend on server-side idempotency behavior.
- Redis Insight inspection depends on how your Agent Memory Server stores data internally.

## Resources

- [Redis Agent Memory](https://pypi.org/project/redis-agent-memory/)
- [LangGraph documentation](https://langchain-ai.github.io/langgraph/)
- [OpenAI API documentation](https://platform.openai.com/docs)
- [Redis Insight](https://redis.io/insight/)

## Maintainers

**Maintainers:**
- Ricardo Ferreira — [@riferrei](https://github.com/riferrei)

## License

This project is licensed under the MIT License.
