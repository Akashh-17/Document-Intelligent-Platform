"""Agents 2 & 3 — Document Analyst and Web Researcher.

Both are autonomous ReAct agents (``create_agent``): the model decides which of
its tools to call, reads the results, and decides what to do next, until it
has enough evidence or hits the step limit. Each runs as a sub-graph inside
the main graph, so the UI sees every tool call it makes.

They don't write the answer. Whatever evidence their tools find goes into the
turn's shared ``CitationRegistry``; they return only short notes.

Failure handling (agent middleware):
- ``ToolErrorMiddleware``      a crashing tool becomes an error message the
                               model can react to, instead of killing the agent.
- ``ToolCallLimitMiddleware``  at the step budget the agent is told to stop and
                               summarise — a graceful end instead of a crash.
- ``ModelFallbackMiddleware``  if the model fails (e.g. rate-limited after its
                               retries), the call is retried on the other tier.
- If the agent still fails (provider down), the node falls back to running the
  planned searches directly, with no LLM, so the user still gets evidence.
"""

from __future__ import annotations

import logging
import re
from datetime import date

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelFallbackMiddleware,
    ToolCallLimitMiddleware,
    ToolErrorMiddleware,
)
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphRecursionError

from rag_agent.agents import events
from rag_agent.agents.prompts import DOC_ANALYST, WEB_RESEARCHER, with_date
from rag_agent.agents.state import ResearchState
from rag_agent.citations import registry_from_config
from rag_agent.config import settings
from rag_agent.llms import get_fallback_model, get_model
from rag_agent.privacy import CONFIG_KEY as PRIVATE_KEY
from rag_agent.privacy import internal_names
from rag_agent.retrieval import hybrid_search
from rag_agent.tools import (
    DOCUMENT_TOOLS,
    WEB_TOOLS,
    read_whole_document,
    search_documents,
    web_search,
)
from rag_agent.tools.common import emit
from rag_agent.vectorstore import list_sources
from rag_agent.websearch import is_time_sensitive

log = logging.getLogger(__name__)

_EVIDENCE_KIND = {"doc_analyst": "document", "web_researcher": "web"}
_URL = re.compile(r"https?://[^\s<>()\"']+")


def _count(config: RunnableConfig, kind: str) -> int:
    return sum(1 for s in registry_from_config(config).sources if s["kind"] == kind)


def _queries(agent: str, state: ResearchState) -> list[str]:
    if agent == "web_researcher" and state.get("gap_fill"):
        return state["gap_fill"]
    plan = state.get("plan") or {}
    key = "doc_queries" if agent == "doc_analyst" else "web_queries"
    return plan.get(key) or [state.get("question", "")]


def _private_names(agent: str, state: ResearchState) -> list[str]:
    """Internal names to keep out of web queries — only while documents are in play
    (route documents/both, or filling a gap the documents left). For a pure web
    question the user may mean the public thing, so nothing is held back."""
    if agent != "web_researcher":
        return []
    route = (state.get("plan") or {}).get("route")
    if route in ("documents", "both") or state.get("gap_fill"):
        return internal_names()
    return []


def _task(agent: str, state: ResearchState) -> str:
    names = _private_names(agent, state)
    keep_private = (
        "\nThese names are internal to the user's documents — never put them in a web "
        f"query; search for the general concept instead: {', '.join(names)}."
        if names
        else ""
    )
    if agent == "web_researcher" and state.get("gap_fill"):
        return with_date(
            f"Research task: {state.get('question', '')}\n"
            "The private documents already answered part of this. Find public "
            "information ONLY about: " + "; ".join(state["gap_fill"]) + keep_private
        )
    lines = [f"Research task: {state.get('question', '')}"]
    lines.append("Suggested queries: " + "; ".join(_queries(agent, state)))
    if agent == "web_researcher":
        if (state.get("plan") or {}).get("route") == "documents":
            lines.append("The private documents had nothing relevant; search the public web.")
        urls = _URL.findall(state.get("question", ""))
        if urls:
            lines.append(
                "The user gave these pages; read them with fetch_webpage: " + " ".join(urls)
            )
    return with_date("\n".join(lines) + keep_private)


ANCHOR_RESULT_CHARS = 3500  # first-search evidence shown to the agent (~900 tokens)
WHOLE_DOC_RESULT_CHARS = 4000  # the agent only needs an overview; the writer gets it all

# Questions about ALL of a document: lists and counts, or an overview of "the document".
_ENUMERATE = re.compile(
    r"\b(?:list|enumerate|how many|all (?:the|of the)|every|each of|key points|main points|"
    r"outline|table of contents|chapters|sections)\b",
    re.IGNORECASE,
)
_OVERVIEW = re.compile(
    r"\b(?:summar\w+|overview|explain|describe|walk me through|tell me about"
    r"|what is .{0,30}about)\b",
    re.IGNORECASE,
)
_DOC_NOUN = re.compile(
    r"\b(?:document|doc|pdf|file|report|guide|paper|book|deck|it|this)\b", re.IGNORECASE
)


def is_whole_document_question(question: str) -> bool:
    return bool(_ENUMERATE.search(question)) or (
        bool(_OVERVIEW.search(question)) and bool(_DOC_NOUN.search(question))
    )


def _named_source(question: str, sources: list[str]) -> str | None:
    """A library file the question names ("the portfolio guide" -> Advanced_..._Portfolio.pdf)."""
    words = set(re.findall(r"[a-z0-9]{4,}", question.lower()))
    best, hits = None, 0
    for s in sources:
        stem = set(re.findall(r"[a-z0-9]{4,}", s.lower().rsplit(".", 1)[0])) - {"final", "draft"}
        overlap = len(stem & words)
        if overlap > hits:
            best, hits = s, overlap
    return best


def _whole_document_target(agent: str, state: ResearchState) -> str | None:
    """Which document to read in full, if the question is about all of one."""
    if agent != "doc_analyst":
        return None
    question = state.get("question", "")
    if not is_whole_document_question(question):
        return None
    sources = list_sources()
    if not sources:
        return None
    if len(sources) == 1:
        return sources[0]
    named = _named_source(question, sources)
    if named:
        return named
    hits = hybrid_search(question)  # otherwise: the document the question matches best
    return hits[0].source if hits else None


def _anchor_query(agent: str, state: ResearchState) -> str | None:
    """The Web Researcher's first search: the user's question in their own words.

    For a "both" question the standalone question mixes in document-specific
    parts, so the planner's public-facing query is used instead. Gap-fill
    searches are already narrowly targeted, so they get no anchor.
    """
    if agent != "web_researcher" or state.get("gap_fill"):
        return None
    plan = state.get("plan") or {}
    query = (
        (plan.get("web_queries") or [None])[0]
        if plan.get("route") == "both"
        else state.get("question")
    )
    if query and is_time_sensitive(query) and not re.search(r"\b20\d\d\b", query):
        query = f"{query.rstrip('?. ')} {date.today().year}"  # bias engines to fresh pages
    return query or None


def tool_error_message(exc: Exception, request) -> str:
    """Shown to the model when a tool raises. Names the error type only — never
    internal details (paths, stack traces) that could leak into the answer."""
    name = request.tool_call.get("name", "tool")
    return (
        f"TOOL_ERROR: {name} failed ({type(exc).__name__}). "
        "Try a different query or tool, or finish with what you have."
    )


def _middleware():
    mw = [
        ToolErrorMiddleware(tool_error_message),
        ToolCallLimitMiddleware(run_limit=settings.worker_max_steps, exit_behavior="continue"),
    ]
    fallback = get_fallback_model("worker")
    if fallback is not None:
        mw.append(ModelFallbackMiddleware(fallback))
    return mw


def _direct_search(agent: str, state: ResearchState, config: RunnableConfig) -> None:
    """No-LLM fallback: run the planned queries straight through the search tool."""
    tool = search_documents if agent == "doc_analyst" else web_search
    for query in _queries(agent, state)[:2]:
        try:
            events.run_tool(agent, tool, {"query": query}, config)
        except Exception:  # noqa: BLE001 — best effort
            log.exception("direct %s search failed", agent)


def make_research_node(agent: str):
    """Build a graph node that runs one research sub-agent."""
    tools = DOCUMENT_TOOLS if agent == "doc_analyst" else WEB_TOOLS
    prompt = DOC_ANALYST if agent == "doc_analyst" else WEB_RESEARCHER
    sub_agent = create_agent(
        model=get_model("worker"),
        tools=tools,
        system_prompt=prompt,
        middleware=_middleware(),
        name=agent,
    )
    kind = _EVIDENCE_KIND[agent]

    def node(state: ResearchState, config: RunnableConfig) -> dict:
        task = _task(agent, state)
        events.agent_start(agent, task.splitlines()[0][:160])
        before = _count(config, kind)

        # Safety net above the tool-call limit: model + tools steps per round.
        sub_config = {**config, "recursion_limit": 2 * settings.worker_max_steps + 4}
        names = _private_names(agent, state)
        if names:  # enforced in the web tools too, not just asked for in the prompt
            sub_config["configurable"] = {**config.get("configurable", {}), PRIVATE_KEY: names}
        whole = _whole_document_target(agent, state)
        if whole:
            # "List all the projects" / "explain this document": top-k search misses
            # list items spread over many pages, so read the document page by page.
            read = events.run_tool(
                agent,
                read_whole_document,
                {"document": whole, "question": state.get("question", "")},
                sub_config,
            )
            task += (
                f"\n\nThe whole of {whole} was already read (best passage of every page, "
                f"in order). Results:\n{str(read)[:WHOLE_DOC_RESULT_CHARS]}\n\n"
                "Use these passages; search again only for something they don't cover."
            )
        anchor = _anchor_query(agent, state)
        if anchor:
            # Search the user's own wording first, deterministically. Models fill
            # queries with what they *remember* is current ("Python 3.12 latest
            # release") and then find exactly that; the anchor keeps the evidence
            # tied to what was asked. It also saves the agent one LLM round.
            first = events.run_tool(agent, web_search, {"query": anchor}, sub_config)
            task += (
                f"\n\nA first web search for {anchor!r} already ran. Results:\n"
                f"{str(first)[:ANCHOR_RESULT_CHARS]}\n\n"
                "Search again only if these don't answer the task."
            )
        status = "ok"
        try:
            result = sub_agent.invoke({"messages": [HumanMessage(task)]}, sub_config)
            notes = result["messages"][-1].text.strip()
        except GraphRecursionError:
            status, notes = "step_limit", "Stopped at the step limit."
        except Exception as exc:  # noqa: BLE001 — one failed agent must not sink the turn
            log.exception("%s failed; falling back to direct search", agent)
            _direct_search(agent, state, config)
            status = "degraded"
            notes = (
                f"The research model failed ({type(exc).__name__}); "
                "ran the planned searches directly instead."
            )

        found = _count(config, kind) - before
        if status == "ok" and found == 0:
            status = "no_evidence"
        events.agent_end(
            agent,
            f"{found} source(s) found. {notes[:300]}",
            ok=status != "degraded" or found > 0,
        )
        update = {
            "findings": [
                {"agent": agent, "status": status, "evidence": found, "notes": notes[:800]}
            ]
        }
        if agent == "web_researcher":
            update["gap_fill"] = []  # a gap-fill request is handled exactly once
        return update

    node.__name__ = f"{agent}_node"
    return node


def route_after_docs(state: ResearchState) -> str:
    """Escalate to the web when a documents-only plan found nothing."""
    plan = state.get("plan") or {}
    doc_findings = [f for f in state.get("findings", []) if f["agent"] == "doc_analyst"]
    nothing_found = all(f["evidence"] == 0 for f in doc_findings)
    if plan.get("route") == "documents" and nothing_found and settings.web_fallback:
        emit(
            {
                "kind": "escalation",
                "from": "doc_analyst",
                "to": "web_researcher",
                "reason": "Nothing relevant in your documents; trying the public web.",
            }
        )
        return "web_researcher"
    return "assess"
