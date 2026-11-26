"""What the user gets when neither their documents nor the web can answer.

Principle: **say so, specifically, and help them move forward** — never fill
the gap with an invented answer. The response is assembled deterministically
(no LLM), so it cannot hallucinate. It tells the user:

1. *What was searched* — which sources, with which queries, and whether each
   source was empty or merely unreachable (web search down != no results).
2. *Why nothing matched* — e.g. no documents uploaded, or the question is
   outside what the documents cover (and what they *do* cover).
3. *The closest passages* — near-misses that weren't relevant enough to
   answer from, so the user can judge for themselves.
4. *Concrete next steps* — upload a document, rephrase with specific terms,
   retry later if a source was down.

It is reached two ways: research found no evidence at all, or the Fact
Verifier judged that the evidence found doesn't actually answer the question.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig

from rag_agent.agents import events
from rag_agent.agents.state import ResearchState
from rag_agent.citations import CitationRegistry, registry_from_config
from rag_agent.config import settings
from rag_agent.retrieval import hybrid_search
from rag_agent.tools.common import emit
from rag_agent.vectorstore import source_stats

log = logging.getLogger(__name__)

DOC_TOOLS = {"search_documents", "search_within_document", "compare_documents"}
WEB_TOOLS = {"web_search", "fetch_webpage", "wikipedia_lookup"}


# --- graph plumbing ------------------------------------------------------------------


def assess_node(state: ResearchState, config: RunnableConfig) -> dict:
    """Join point after research: count the evidence before deciding what to do."""
    return {"evidence_count": len(registry_from_config(config))}


def route_after_assess(state: ResearchState) -> str:
    if state.get("evidence_count", 0) > 0:
        return "compose"
    return "compose" if settings.answer_without_evidence else "no_answer"


def no_answer_node(state: ResearchState, config: RunnableConfig) -> dict:
    registry = registry_from_config(config)
    if not state.get("unanswered"):  # came straight from research: verifier wasn't started
        events.agent_start("verifier", "Checking why nothing was found")
    result = build_no_answer(
        question=state.get("question", ""),
        registry=registry,
        findings=state.get("findings", []),
        unanswered=bool(state.get("unanswered")),
    )
    emit({"kind": "no_answer", **result})
    verification = {
        "checked": bool(state.get("unanswered")),
        "confidence": "none",
        "coverage": "none",
        "evidence_count": len(registry),
    }
    emit({"kind": "verification", **verification})
    events.agent_end("verifier", "No answer found — explained what was searched.")
    return {
        "answer": result["markdown"],
        "verification": verification,
    }


# --- the response itself --------------------------------------------------------------


def _source_status(attempts: list[dict], tools: set[str]) -> str:
    """not_searched | unavailable | no_results | irrelevant"""
    mine = [a for a in attempts if a["tool"] in tools]
    if not mine:
        return "not_searched"
    outcomes = {a["outcome"] for a in mine}
    if outcomes <= {"unavailable", "blocked", "error"}:
        return "unavailable"
    if "ok" in outcomes:
        return "irrelevant"  # something came back, but it didn't answer the question
    return "no_results"


def _near_misses(question: str, registry: CitationRegistry, unanswered: bool) -> list[dict]:
    """Closest document passages, shown as "possibly related" — never as an answer."""
    if unanswered:  # evidence was found but judged not to answer: show that evidence
        return [
            {"title": s["title"], "snippet": s["snippet"][:220], "url": s.get("url")}
            for s in registry.sources[: settings.near_miss_count]
        ]
    try:
        hits = hybrid_search(question, k=settings.near_miss_count, gate=False)
    except Exception:  # noqa: BLE001 — near-misses are a nicety, never a failure
        log.exception("near-miss search failed")
        return []
    return [
        {"title": h.source, "snippet": h.text[:220].strip(), "score": round(h.relevance, 2)}
        for h in hits
    ]


def build_no_answer(
    question: str,
    registry: CitationRegistry,
    findings: list[dict[str, Any]],
    unanswered: bool = False,
) -> dict[str, Any]:
    stats = source_stats()
    kb_empty = not stats
    attempts = registry.attempts
    docs = "empty" if kb_empty else _source_status(attempts, DOC_TOOLS)
    web = _source_status(attempts, WEB_TOOLS)
    research_failed = bool(findings) and all(f["status"] == "degraded" for f in findings)
    if unanswered and docs == "no_results":
        docs = "irrelevant"

    # -- searched: one line per source -------------------------------------------------
    def queries(tools: set[str]) -> list[str]:
        seen: list[str] = []
        for a in attempts:
            if a["tool"] in tools and a["query"] and a["query"] not in seen:
                seen.append(a["query"])
        return seen[:4]

    doc_text = {
        "empty": "No documents have been uploaded yet.",
        "not_searched": "Not searched for this question.",
        "no_results": f"Searched all {len(stats)} documents — nothing relevant.",
        "irrelevant": f"Searched all {len(stats)} documents — the closest passages don't "
        "answer this question.",
        "unavailable": "The document search failed.",
    }[docs]
    web_text = {
        "not_searched": "Not searched for this question."
        + ("" if settings.web_fallback else " (web fallback is turned off)"),
        "no_results": "Searched — no useful results.",
        "irrelevant": "Searched — the results don't answer this question.",
        "unavailable": "Web search was unavailable (rate-limited or offline).",
    }[web]
    searched = [
        {
            "where": "Your documents",
            "status": docs,
            "summary": doc_text,
            "queries": queries(DOC_TOOLS),
        },
        {"where": "The web", "status": web, "summary": web_text, "queries": queries(WEB_TOOLS)},
    ]

    # -- headline + why ---------------------------------------------------------------
    if research_failed:
        reason = "service_unavailable"
        headline = "I couldn't research this right now."
        why = "The AI service was unavailable, so the search agents couldn't run properly."
    elif kb_empty and web != "irrelevant":
        reason = "no_documents"
        headline = "I couldn't find an answer — and no documents are uploaded yet."
        why = "There are no documents to search, and the web didn't have a usable answer."
    elif web == "unavailable":
        reason = "web_unavailable"
        headline = "I couldn't find this in your documents, and web search is unavailable."
        why = "Your documents don't cover it, and I couldn't reach the web to check further."
    else:
        reason = "not_found"
        headline = "I couldn't find a reliable answer to this."
        why = "Neither your documents nor the web had information that answers it."

    covers = [d["title"] or d["source"] for d in stats][:6]
    if covers and reason != "service_unavailable":
        why += " Your documents cover: " + "; ".join(covers) + "."
    why += " I don't answer from memory without a source, so I won't guess."

    # -- suggestions --------------------------------------------------------------------
    suggestions: list[str] = []
    if reason == "service_unavailable" or web == "unavailable":
        suggestions.append("Try again in a minute — the service may be rate-limited.")
    if kb_empty:
        suggestions.append("Upload documents that cover this topic (PDF, TXT, Markdown).")
    else:
        suggestions.append(
            "If this should be in your documents, upload the one that covers it — "
            "or check the closest passages below."
        )
        suggestions.append(
            "Rephrase with the exact terms your documents use (names, error codes, config keys)."
        )
    if web == "unavailable" and not settings.tavily_api_key:
        suggestions.append(
            "Set TAVILY_API_KEY for more reliable web search than the keyless fallback."
        )
    if web == "not_searched" and not settings.web_fallback:
        suggestions.append("Enable web search (WEB_FALLBACK=true) to look beyond your documents.")
    suggestions.append("Split a multi-part question into simpler ones.")

    near = [] if kb_empty else _near_misses(question, registry, unanswered)

    return {
        "reason": reason,
        "headline": headline,
        "explanation": why,
        "searched": searched,
        "near_misses": near,
        "suggestions": suggestions[:4],
        "markdown": _markdown(headline, why, searched, near, suggestions[:4]),
    }


def _markdown(headline, why, searched, near, suggestions) -> str:
    """Plain-markdown version: stored in chat history and returned by the API/evals."""
    lines = [f"**{headline}**", "", why, "", "**What I searched**"]
    for s in searched:
        q = f" (queries: {', '.join(repr(x) for x in s['queries'])})" if s["queries"] else ""
        lines.append(f"- {s['where']}: {s['summary']}{q}")
    if near:
        lines += ["", "**Closest passages (not relevant enough to answer from)**"]
        lines += [f"- *{n['title']}*: {n['snippet'][:160]}…" for n in near]
    lines += ["", "**What you can do**"] + [f"- {s}" for s in suggestions]
    return "\n".join(lines)
