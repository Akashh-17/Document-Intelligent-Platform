"""Agent 4 — Fact Verifier: drafts the cited answer, checks it, self-corrects.

Two graph nodes:

- ``compose``: writes the answer from the shared evidence with the heavy model.
  This is the only LLM output streamed to the user token by token.
- ``verify``:  runs ``verify_citations`` (exact, free) and ``check_groundedness``
  (LLM judge). If it finds invented citations or unsupported claims, it sends
  concrete feedback back to ``compose`` for one revision; otherwise it
  finalises the answer with a confidence level.

This "generate -> critique -> revise" loop is the reflection pattern. Capping it
at ``max_revisions`` bounds cost and latency.
"""

from __future__ import annotations

import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from rag_agent.agents import events
from rag_agent.agents.planner import _history
from rag_agent.agents.prompts import (
    WRITER,
    WRITER_DIRECT,
    WRITER_REWORK,
    WRITER_UNGROUNDED,
    with_date,
)
from rag_agent.agents.state import ResearchState
from rag_agent.citations import CitationRegistry, registry_from_config
from rag_agent.config import settings
from rag_agent.llms import with_fallback
from rag_agent.retrieval.bm25 import tokenize
from rag_agent.tools import WEB_TOOLS, check_groundedness, verify_citations
from rag_agent.tools.common import emit
from rag_agent.vectorstore import list_sources

log = logging.getLogger(__name__)

PER_SOURCE_CHARS = 1500
WEB_TOOL_NAMES = {t.name for t in WEB_TOOLS}


def build_evidence(registry: CitationRegistry, budget: int | None = None) -> str:
    """Numbered evidence for the writer and the judge (identical text for both)."""
    if budget is None:
        budget = settings.evidence_char_budget
        if any(a["tool"] == "read_whole_document" for a in registry.attempts):
            # the page-by-page read must reach the writer whole, or lists come out short
            budget = max(budget, settings.whole_document_char_budget + 2000)
    parts, used = [], 0
    for s in registry.sources:
        body = registry.text(s["index"])[:PER_SOURCE_CHARS]
        block = f"[{s['index']}] {s['title']}" + (f" ({s['url']})" if s.get("url") else "")
        block += f"\n{body}"
        if used + len(block) > budget and parts:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)


def _notes(state: ResearchState) -> str:
    lines = [
        f"- {f['agent']} ({f['status']}, {f['evidence']} sources): {f['notes']}"
        for f in state.get("findings", [])
    ]
    return "\n".join(lines) or "(no research was needed)"


def best_excerpt(text: str, question: str, width: int = 350) -> str:
    """The ``width``-char window of ``text`` that shares the most terms with the question."""
    flat = " ".join(text.split())
    if len(flat) <= width:
        return flat
    terms = set(tokenize(question))
    step = max(1, width // 4)
    starts = range(0, max(1, len(flat) - width + step), step)

    def overlap(start: int) -> int:
        return len(terms & set(tokenize(flat[start : start + width])))

    best = max(starts, key=overlap)  # max() keeps the earliest window on ties
    excerpt = flat[best : best + width]
    return ("…" if best else "") + excerpt + ("…" if best + width < len(flat) else "")


def _is_strong(source: dict) -> bool:
    """Evidence we'd quote without an LLM to judge it: web results, or document
    passages the cross-encoder confirmed. Passages admitted only on embedding
    similarity can be loosely related, and quoting those as "the answer" would
    mislead."""
    if source["kind"] != "document":
        return True
    rerank = (source.get("retrieval") or {}).get("rerank_score")
    return rerank is None or rerank >= settings.min_rerank_score


def extractive_answer(registry: CitationRegistry, question: str, k: int = 3) -> str | None:
    """Last-resort answer when every model is down: quote the best evidence verbatim.

    Verbatim quotes with citations can't be "wrong" the way generated text can,
    and they still give the user something useful instead of an error. Returns
    ``None`` when no evidence is strong enough to quote.
    """
    strong = [s for s in registry.sources if _is_strong(s)][:k]
    if not strong:
        return None
    lines = [
        "_I couldn't reach the language model to write a summary, so here are the most "
        "relevant passages I found._",
        "",
    ]
    for s in strong:
        excerpt = best_excerpt(registry.text(s["index"]), question)
        lines.append(f'- **{s["title"]}**: "{excerpt}" [{s["index"]}]')
    return "\n".join(lines)


# gpt-oss models often cite with CJK lenticular brackets: 【3】 or 【3†L10-L20】.
_LENTICULAR = re.compile(r"【\s*(\d+)[^】]*】")
_UNICODE_FIXES = str.maketrans(
    {
        "\u2011": "-",  # non-breaking hyphen (breaks copy-paste of IDs like AUR-2005)
        "\u2010": "-",  # hyphen
        "\u2013": "-",  # en dash
        "\u202f": " ",  # narrow no-break space
        "\u00a0": " ",  # no-break space
        "\u2009": " ",  # thin space
    }
)


def normalize_answer(text: str) -> str:
    """Canonical citations ``[n]`` and plain ASCII hyphens/spaces in model output."""
    return _LENTICULAR.sub(r"[\1]", text).translate(_UNICODE_FIXES).strip()


# The answer itself admits the evidence lacks the information.
_ADMITS_MISSING = re.compile(
    r"\b(?:does not|doesn't|do not|don't|did not|didn't)\s+(?:contain|include|mention|"
    r"provide|cover|specify|say|define|describe|state|refer)\b"
    r"|\bno (?:information|mention|details?|data)\b"
    r"|\bnot (?:found|available|mentioned|covered|specified|stated|provided)\b"
    r"|\b(?:no|none of the)\b(?:\W+\w+){0,8}?\W+(?:mentions?|refers? to|contains?|states?|says)\b",
    re.IGNORECASE,
)
_FIRST_SENTENCE = re.compile(r"^.*?(?:[.!?](?=\s)|\n|$)", re.DOTALL)


def coverage_backstop(answer: str, judged: str, has_citations: bool) -> str:
    """Deterministic check on the LLM judge's coverage verdict.

    Small judge models sometimes rate "the documents don't mention X" as a full
    answer. The writer must lead with the direct answer, so if the FIRST sentence
    admits the information is missing (or nothing is cited at all), the question
    went unanswered ("none"); an admission later on means "partial".
    """
    if judged == "none" or not _ADMITS_MISSING.search(answer):
        return judged
    first = _FIRST_SENTENCE.match(answer.strip().lstrip("*# ").strip())
    if not has_citations or (first and _ADMITS_MISSING.search(first.group(0))):
        return "none"
    return "partial"


def compose_node(state: ResearchState, config: RunnableConfig) -> dict:
    registry = registry_from_config(config)
    route = (state.get("plan") or {}).get("route")
    revision = state.get("revisions", 0) > 0
    ungrounded = route != "direct" and len(registry) == 0

    if revision:
        emit({"kind": "revision", "reason": state.get("feedback", "")[:300]})
    else:
        events.agent_start("verifier", "Drafting the cited answer")

    if route == "direct" or ungrounded:
        system = WRITER_DIRECT if route == "direct" else WRITER_UNGROUNDED
        library = ", ".join(list_sources()[:12]) or "empty (nothing uploaded yet)"
        messages = [
            SystemMessage(with_date(system)),
            HumanMessage(
                f"The user's document library: {library}\n\n"
                f"Conversation so far:\n{_history(state)}\n\nUser: {state.get('question', '')}"
            ),
        ]
    elif route == "rework":
        last = state.get("last_answer") or {}
        prompt = (
            f"YOUR PREVIOUS ANSWER (to: {last.get('question', '')}):\n{last.get('answer', '')}\n\n"
            f"WHAT THE USER WANTS NOW: {state.get('question', '')}\n\n"
            f"EVIDENCE BEHIND IT (same numbering):\n{build_evidence(registry)}"
        )
        if revision:
            prompt += (
                f"\n\nYOUR PREVIOUS DRAFT:\n{state.get('answer', '')}\n\n"
                f"FACT-CHECK FEEDBACK — fix these problems:\n{state.get('feedback', '')}"
            )
        messages = [SystemMessage(with_date(WRITER_REWORK)), HumanMessage(prompt)]
    else:
        prompt = (
            f"QUESTION: {state.get('question', '')}\n\n"
            f"CONVERSATION SO FAR:\n{_history(state)}\n\n"
            f"RESEARCH NOTES:\n{_notes(state)}\n\n"
            f"EVIDENCE:\n{build_evidence(registry)}"
        )
        if revision:
            prompt += (
                "\n\nYOUR PREVIOUS DRAFT:\n"
                f"{state.get('answer', '')}\n\n"
                f"FACT-CHECK FEEDBACK — fix these problems:\n{state.get('feedback', '')}"
            )
        messages = [SystemMessage(with_date(WRITER)), HumanMessage(prompt)]

    try:
        # primary writer model, then the other tier if it fails (rate limit, 5xx)
        reply = with_fallback("writer", lambda m: m).invoke(messages, config)
        raw = reply.text.strip()
        answer = normalize_answer(raw)
        if answer != raw:  # the UI streamed the raw text; send the clean version
            emit({"kind": "replace_answer", "text": answer})
        return {"answer": answer, "degraded": False}
    except Exception as exc:
        if route in ("direct", "rework") or ungrounded or len(registry) == 0:
            raise  # nothing to fall back on: surface a friendly error
        log.exception("all writer models failed; trying an extractive answer")
        emit({"kind": "degraded", "reason": f"Writer unavailable ({type(exc).__name__})."})
        text = extractive_answer(registry, state.get("question", ""))
        if text is None:  # only weak matches: be honest instead of quoting them
            return {"degraded": True, "unanswered": True}
        emit({"kind": "replace_answer", "text": text})
        return {"answer": text, "degraded": True}


def _confidence(grounded: dict, citations: dict, problems: list[str]) -> str:
    if grounded.get("grounded") is None:
        return "unverified"  # the judge itself failed
    score = grounded.get("score", 0.0)
    if not problems and score >= 0.8:
        return "high"
    if score >= 0.5 and not citations.get("invalid"):
        return "medium"
    return "low"


def verify_node(state: ResearchState, config: RunnableConfig) -> dict:
    registry = registry_from_config(config)
    route = (state.get("plan") or {}).get("route")
    answer = state.get("answer", "")

    if state.get("unanswered"):  # compose already decided there's nothing to say
        return {"unanswered": True}
    if route == "direct":
        return _finalise(answer, {"checked": False, "confidence": "n/a", "evidence_count": 0})
    if len(registry) == 0:  # only reachable with ANSWER_WITHOUT_EVIDENCE=true
        return _finalise(
            answer,
            {
                "checked": False,
                "confidence": "unverified",
                "model_knowledge": True,
                "evidence_count": 0,
            },
        )
    if state.get("degraded"):
        return _finalise(
            answer,
            {"checked": False, "confidence": "extractive", "evidence_count": len(registry)},
        )

    citations = events.run_tool(
        "verifier",
        verify_citations,
        {"answer": answer, "available": registry.indices()},
        config,
    )
    grounded = events.run_tool(
        "verifier",
        check_groundedness,
        {
            "answer": answer,
            "evidence": build_evidence(registry),
            "question": state.get("question", ""),
        },
        config,
    )

    # The evidence exists but doesn't answer the question (e.g. retrieval found
    # loosely related passages). Revising can't fix missing information: give
    # the honest "couldn't find it" response instead.
    if route == "rework":
        # A rewrite answers "make it shorter", not the original question: judging
        # how much of the question it covers would flag every summary as partial.
        grounded["coverage"], grounded["missing_aspects"] = "full", []
    grounded["coverage"] = coverage_backstop(
        answer, grounded.get("coverage", "full"), citations["has_citations"]
    )
    if grounded["coverage"] == "partial" and not grounded.get("missing_aspects"):
        grounded["missing_aspects"] = ["part of the question (see the answer)"]
    if grounded["coverage"] == "none":
        return {"unanswered": True}

    # Gap filling: the documents answered part of the question. If the web hasn't
    # been searched yet, send the Web Researcher after exactly what's missing,
    # then rewrite with the extra evidence (counts as the one revision).
    missing = grounded.get("missing_aspects") or []
    web_tried = any(a["tool"] in WEB_TOOL_NAMES for a in registry.attempts)
    if (
        grounded.get("coverage") == "partial"
        and missing
        and not web_tried
        and settings.web_fallback
        and state.get("revisions", 0) < settings.max_revisions
    ):
        emit(
            {
                "kind": "escalation",
                "from": "verifier",
                "to": "web_researcher",
                "reason": "Your documents don't cover: "
                + "; ".join(missing)
                + ". Checking the web.",
            }
        )
        return {
            "gap_fill": missing,
            "revisions": state.get("revisions", 0) + 1,
            "feedback": "Your draft could not answer: "
            + "; ".join(missing)
            + ". New evidence was gathered for these; complete the answer with it.",
        }

    problems: list[str] = []
    if citations["invalid"]:
        problems.append(
            f"Citations {citations['invalid']} do not exist. Use only: {registry.indices()}."
        )
    if not citations["has_citations"] and grounded.get("grounded") is not True:
        problems.append("The answer cites nothing. Cite evidence numbers inline after claims.")
    if grounded.get("grounded") is False:
        problems += [
            f"Unsupported by the evidence, remove or qualify: {c}"
            for c in grounded.get("unsupported_claims", [])
        ]
        problems += [
            f"Wrong citation, fix the number: {c}" for c in grounded.get("wrong_citations", [])
        ]

    if problems and state.get("revisions", 0) < settings.max_revisions:
        return {
            "feedback": "\n".join(f"- {p}" for p in problems),
            "revisions": state.get("revisions", 0) + 1,
        }

    verification = {
        "checked": True,
        "confidence": _confidence(grounded, citations, problems),
        "grounded": grounded.get("grounded"),
        "score": grounded.get("score"),
        "coverage": grounded.get("coverage", "full"),
        "missing_aspects": grounded.get("missing_aspects", []),
        "unsupported_claims": grounded.get("unsupported_claims", []),
        "invalid_citations": citations["invalid"],
        "revisions": state.get("revisions", 0),
        "remaining_problems": problems,
        "evidence_count": len(registry),
    }
    return _finalise(answer, verification)


def _finalise(answer: str, verification: dict) -> dict:
    emit({"kind": "verification", **verification})
    events.agent_end("verifier", f"Confidence: {verification['confidence']}")
    return {"answer": answer, "verification": verification}


def route_after_verify(state: ResearchState) -> str:
    """Revise until the verifier finalises; hand over if the question went unanswered."""
    if state.get("unanswered"):
        return "no_answer"
    if state.get("gap_fill"):
        return "web_researcher"
    return "guard_output" if state.get("verification") else "compose"
