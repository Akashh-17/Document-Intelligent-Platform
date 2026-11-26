"""Agent 1 — Query Planner (the supervisor).

Reads the conversation and the document inventory (via ``list_documents``),
then emits a structured ``Plan``: a standalone rewrite of the question, a
route, and focused search queries. Routing is a *structured* decision, not a
free-form ReAct loop: it is one cheap call on the fast model, and its output
can be validated, logged, and evaluated.
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from rag_agent.agents import events
from rag_agent.agents.prompts import PLANNER
from rag_agent.agents.state import Plan, ResearchState
from rag_agent.llms import with_fallback
from rag_agent.tools import list_documents
from rag_agent.tools.common import emit

log = logging.getLogger(__name__)

HISTORY_TURNS = 3  # earlier exchanges shown to the planner for follow-up questions


def latest_user_message(state: ResearchState) -> str:
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            return m.text
    return ""


def _history(state: ResearchState) -> str:
    msgs = [m for m in state.get("messages", []) if isinstance(m, HumanMessage | AIMessage)]
    earlier = msgs[:-1][-2 * HISTORY_TURNS :]
    lines = [
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.text[:400]}" for m in earlier
    ]
    return "\n".join(lines) or "(no earlier conversation)"


def fallback_plan(question: str, has_documents: bool) -> Plan:
    """Used when the planner model fails: search everything we have."""
    return Plan(
        standalone_question=question,
        route="both" if has_documents else "web",
        doc_queries=[question] if has_documents else [],
        web_queries=[question],
        rationale="Planner unavailable; searching all sources.",
    )


def plan_node(state: ResearchState, config: RunnableConfig) -> dict:
    question = latest_user_message(state)
    events.agent_start("planner", "Understanding the question")

    inventory = events.run_tool("planner", list_documents, {}, config)
    has_documents = not inventory.startswith("The document collection is empty")

    planner = with_fallback("planner", lambda m: m.with_structured_output(Plan))
    try:
        plan = planner.invoke(
            [
                SystemMessage(PLANNER),
                HumanMessage(
                    f"DOCUMENT INVENTORY:\n{inventory}\n\n"
                    f"CONVERSATION SO FAR:\n{_history(state)}\n\n"
                    f"LATEST USER MESSAGE:\n{question}"
                ),
            ],
            config,
        )
        if isinstance(plan, dict):
            plan = Plan.model_validate(plan)
    except Exception:  # noqa: BLE001 — never let routing failure kill the turn
        log.exception("planner failed; using fallback plan")
        plan = fallback_plan(question, has_documents)

    # Guard rails on the model's output.
    if not has_documents and plan.route in ("documents", "both"):
        plan.route = "web"  # nothing to search locally
    plan.standalone_question = plan.standalone_question.strip() or question
    plan.doc_queries = [q for q in plan.doc_queries if q.strip()][:3] or [plan.standalone_question]
    plan.web_queries = [q for q in plan.web_queries if q.strip()][:2] or [plan.standalone_question]

    emit(
        {
            "kind": "plan",
            "route": plan.route,
            "question": plan.standalone_question,
            "doc_queries": plan.doc_queries,
            "web_queries": plan.web_queries,
            "rationale": plan.rationale,
        }
    )
    events.agent_end("planner", f"Route: {plan.route}. {plan.rationale}".strip())

    return {
        "question": plan.standalone_question,
        "plan": plan.model_dump(),
        # reset per-turn working state (the checkpointer keeps the last turn's)
        "findings": None,
        "answer": "",
        "verification": None,
        "revisions": 0,
        "feedback": "",
        "evidence_count": 0,
        "unanswered": False,
        "degraded": False,
        "gap_fill": [],
    }


def route_after_plan(state: ResearchState) -> list[str]:
    route = (state.get("plan") or {}).get("route", "both")
    return {
        "documents": ["doc_analyst"],
        "web": ["web_researcher"],
        "both": ["doc_analyst", "web_researcher"],  # run in parallel
        "direct": ["compose"],
    }.get(route, ["doc_analyst", "web_researcher"])
