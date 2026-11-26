"""Helpers shared by every tool: evidence registration, streaming, formatting."""

from __future__ import annotations

import difflib
import functools
import inspect

from langgraph.config import get_stream_writer

from rag_agent.citations import CitationRegistry, registry_from_config
from rag_agent.retrieval import RetrievedChunk
from rag_agent.vectorstore import list_sources

SNIPPET_CHARS = 300


def emit(payload: dict) -> None:
    """Best-effort push to LangGraph's custom stream channel (no-op outside a run)."""
    try:
        writer = get_stream_writer()
    except Exception:  # noqa: BLE001 — no active run (e.g. a direct tool call in a test)
        return
    if writer is not None:
        writer(payload)


def emit_sources(tool_name: str, sources: list[dict]) -> None:
    if sources:
        emit({"kind": "sources", "tool": tool_name, "sources": sources})


def register_chunk(registry: CitationRegistry, hit: RetrievedChunk) -> dict:
    md = hit.metadata
    title = hit.source + (f" (p. {md['page']})" if md.get("page") else "")
    return registry.register(
        key=f"doc:{hit.id}",
        source={
            "kind": "document",
            "title": title,
            "snippet": hit.text[:SNIPPET_CHARS],
            "score": round(hit.relevance, 3),
            "retrieval": hit.signals(),
        },
        text=hit.text,
        ref={"chunk_id": hit.id, "source": hit.source, "chunk_index": md.get("chunk_index")},
    )


def format_evidence(entries: list[dict], bodies: list[str]) -> str:
    """Evidence block for the LLM: each passage labelled with its citation number."""
    parts = []
    for e, body in zip(entries, bodies, strict=True):
        url = f" ({e['url']})" if e.get("url") else ""
        parts.append(f"[{e['index']}] {e['title']}{url}\n{body}")
    return "\n\n".join(parts)


def resolve_source(name: str) -> str | None:
    """Map a possibly-inexact document name from the LLM to an indexed source.

    LLMs often say "the error codes doc" or drop the extension; matching
    leniently lets the agent recover instead of failing the tool call.
    """
    sources = list_sources()
    if name in sources:
        return name
    lowered = name.lower().strip()
    for s in sources:
        if lowered in s.lower() or s.lower().rsplit(".", 1)[0] in lowered:
            return s
    close = difflib.get_close_matches(name, sources, n=1, cutoff=0.5)
    return close[0] if close else None


def unknown_source_message(name: str) -> str:
    available = ", ".join(list_sources()) or "(no documents indexed)"
    return f"UNKNOWN_DOCUMENT: no document matches '{name}'. Available documents: {available}"


# --- search-attempt logging ----------------------------------------------------------

_OUTCOME_PREFIXES = {
    "NO_RELEVANT_DOCUMENTS": "no_results",
    "NO_WEB_RESULTS": "no_results",
    "NO_WIKIPEDIA_ARTICLE": "no_results",
    "WEB_SEARCH_UNAVAILABLE": "unavailable",
    "WIKIPEDIA_UNAVAILABLE": "unavailable",
    "FETCH_FAILED": "unavailable",
    "FETCH_BLOCKED": "blocked",
    "UNKNOWN_DOCUMENT": "error",
}


def outcome_of(result: str) -> str:
    for prefix, outcome in _OUTCOME_PREFIXES.items():
        if result.startswith(prefix):
            return outcome
    return "ok"


def logs_attempt(query_arg: str):
    """Decorator for search tools: record ``(tool, query, outcome)`` in the turn's
    registry, so a "nothing found" answer can say exactly what was tried and
    whether a source was empty or merely unreachable."""

    def decorate(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            result = fn(*args, **kwargs)
            bound = inspect.signature(fn).bind_partial(*args, **kwargs)
            config = bound.arguments.get("config")
            query = str(bound.arguments.get(query_arg, ""))
            outcome = outcome_of(result)
            detail = result[:200] if outcome != "ok" else ""
            registry_from_config(config).log_attempt(fn.__name__, query, outcome, detail)
            return result

        return wrapper

    return decorate
