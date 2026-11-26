"""Public entry points for the multi-agent research system."""

from __future__ import annotations

import uuid
from typing import Any

from rag_agent.agents.graph import build_graph
from rag_agent.citations import CONFIG_KEY, CitationRegistry


def build_agent(checkpointer=None):
    """Return the compiled four-agent LangGraph (kept for API compatibility)."""
    return build_graph(checkpointer=checkpointer)


def run_config(thread_id: str | None = None, run_id: str | None = None) -> dict[str, Any]:
    """Config for one turn: a fresh evidence registry plus thread/run identity."""
    return {
        "configurable": {
            "thread_id": thread_id or str(uuid.uuid4()),
            CONFIG_KEY: CitationRegistry(),
        },
        "run_id": run_id or str(uuid.uuid4()),
        "recursion_limit": 40,
    }


def run_research(question: str, graph=None) -> dict[str, Any]:
    """Answer one question synchronously (used by evals and scripts).

    Returns the answer, the evidence texts (RAG "contexts"), the sources,
    the plan, the verifier's verdict, and an ``outcome``:
    answered | no_answer | blocked | declined | extractive.
    """
    graph = graph or build_graph()
    config = run_config()
    state = graph.invoke({"messages": [{"role": "user", "content": question}]}, config)
    registry: CitationRegistry = config["configurable"][CONFIG_KEY]
    verification = state.get("verification") or {}
    blocked = (state.get("blocked") or {}).get("category")
    if blocked:
        outcome = "declined" if blocked == "policy" else "blocked"
    elif verification.get("confidence") == "none":
        outcome = "no_answer"
    elif state.get("degraded"):
        outcome = "extractive"
    else:
        outcome = "answered"
    return {
        "answer": state.get("answer", ""),
        "contexts": [registry.text(i) for i in registry.indices()],
        "sources": registry.sources,
        "plan": state.get("plan"),
        "verification": verification,
        "outcome": outcome,
    }
