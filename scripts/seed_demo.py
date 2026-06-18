"""Seed (or reset) the UK mortgage assistant demo.

Replays a few canned conversations through the agent so the demo opens with a
populated thread sidebar (langgraph-redis) and a borrower long-term-memory
profile (redis-agent-memory). The whole point on stage: open a *new* thread and
watch the adviser already know the borrower's income, deposit and preferences.

Usage:
    uv run python scripts/seed_demo.py            # seed the demo
    uv run python scripts/seed_demo.py --reset    # wipe demo state first, then seed
    uv run python scripts/seed_demo.py --reset-only

Requires the same .env as the app (OpenAI + Agent Memory creds + REDIS_URL).
"""
from __future__ import annotations

import argparse
import sys

from backend.app import agent_memory_client
from backend.memory import build_service, coerce_memories, models


# Each thread is (title-is-auto-generated, [user messages in order]). The first
# thread establishes the borrower profile; later threads rely on it via LTM.
# A believable set of PAST conversations for one borrower (a first-time-buyer
# couple in Bristol) spread over time. Each conversation contributes a distinct
# slice of the profile to long-term memory:
#   1) who they are + household income
#   2) deposit saved + target property/area
#   3) the mortgage product they prefer
# In the live demo you walk through these, then open a NEW thread and ask a
# fresh question — the assistant answers using all of the above from LTM.
# Keep each conversation short so responses stay concise.
# Two short, REALISTIC historical conversations to seed: each is a genuine
# question a first-time buyer would ask, with the durable facts surfacing
# naturally as context (not stated in a vacuum). They establish part of the
# borrower profile (who they are + income, then deposit + target) — but
# deliberately NOT everything, leaving room in the live demo to add more
# context (e.g. employment, product preference) in a new chat. A final new
# chat then synthesises ALL of the accumulated long-term memories.
SEED_THREADS: list[list[str]] = [
    [
        "We're first-time buyers in Bristol trying to work out what we can realistically "
        "afford. Our combined income is around £92,000 a year — how much could we borrow?"
    ],
    [
        "We've saved about £21,000 for a deposit and we've been viewing flats around "
        "£340,000. Is that a big enough deposit, or should we hold off and save more?"
    ],
]


def reset(service) -> None:
    """Remove this demo's threads, checkpoints, and borrower long-term memories.

    Scoped to this demo's own thread ids — it does NOT wipe the whole Redis
    database, so it is safe to point at a shared instance.
    """
    owner_id = service.config.owner_id
    print(f"Resetting demo state for owner '{owner_id}'…")

    client = service.thread_index.client

    # 1. Delete the RedisSaver checkpoints for each of this owner's threads only
    #    (match the thread id anywhere in the key, across checkpoint variants).
    thread_ids = [t.thread_id for t in service.thread_index.list(owner_id)]
    deleted = 0
    for thread_id in thread_ids:
        for key in client.scan_iter(match=f"*{thread_id}*", count=500):
            client.delete(key)
            deleted += 1
    print(f"  deleted {deleted} checkpoint keys across {len(thread_ids)} threads")

    # 2. Thread index (sorted set + per-thread metadata hashes).
    service.thread_index.reset(owner_id)

    # 3. Borrower long-term memories (best-effort semantic sweep).
    try:
        with agent_memory_client(service) as agent_memory:
            response = agent_memory.search_long_term_memory(
                request={
                    "text": "mortgage customer profile income deposit preference",
                    "limit": 100,
                    "filter": {
                        "ownerId": {"eq": owner_id},
                        "namespace": {"eq": service.config.namespace},
                    },
                    "filterOp": models.FilterConjunction.ALL,
                }
            )
            memory_ids = []
            for memory in coerce_memories(response):
                mem_id = memory.get("id") if isinstance(memory, dict) else getattr(memory, "id", None)
                if mem_id:
                    memory_ids.append(mem_id)
            if memory_ids:
                agent_memory.bulk_delete_long_term_memories(memory_ids=memory_ids)
            print(f"  deleted {len(memory_ids)} long-term memories")
    except Exception as exc:  # noqa: BLE001 - reset is best-effort
        print(f"  (skipped long-term memory delete: {exc})")


def seed(service) -> None:
    print("Seeding demo conversations…")
    with agent_memory_client(service) as agent_memory:
        for index, messages in enumerate(SEED_THREADS, start=1):
            thread = service.create_thread()
            print(f"\nThread {index}: {thread.thread_id}")
            for message in messages:
                result = service.run_turn(agent_memory, thread.thread_id, message)
                extracted = (
                    f"  ↳ extracted: {result.extracted_memories}" if result.extracted_memories else ""
                )
                print(f"  • {message[:70]}{extracted}")
            print(f"  title → {result.title!r}")
    print("\nDone. Open the app and pick a conversation from the sidebar.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed or reset the mortgage demo.")
    parser.add_argument("--reset", action="store_true", help="Wipe demo state before seeding.")
    parser.add_argument("--reset-only", action="store_true", help="Wipe demo state and exit.")
    args = parser.parse_args()

    service = build_service()
    if args.reset or args.reset_only:
        reset(service)
    if not args.reset_only:
        seed(service)
    return 0


if __name__ == "__main__":
    sys.exit(main())
