"""Agent 1 — Query Planner (the supervisor).

Reads the conversation and the document inventory (via ``list_documents``),
then emits a structured ``Plan``: a standalone rewrite of the question, a
route, and focused search queries. Routing is a *structured* decision, not a
free-form ReAct loop: it is one cheap call on the fast model, and its output
can be validated, logged, and evaluated.
"""

from __future__ import annotations

import logging
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from rag_agent.agents import events
from rag_agent.agents.intent import classify
from rag_agent.agents.prompts import PLANNER, with_date
from rag_agent.agents.state import Plan, ResearchState
from rag_agent.citations import registry_from_config
from rag_agent.llms import with_fallback
from rag_agent.tools import list_documents
from rag_agent.tools.common import emit, emit_sources

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


_SMALL_TALK = re.compile(
    r"^\W*(?:hi|hello|hey|yo|sup|wass?up|what'?s up|how are you|thanks|thank you|thx|"
    r"good (?:morning|afternoon|evening|night)|bye|goodbye|ok|okay|cool|great|nice|got it)\b",
    re.IGNORECASE,
)
_ABOUT_ASSISTANT = re.compile(
    r"\b(?:who|what) are you\b|\bwhat can you\b|\bcan you (?:help|do)\b|\bhow do you work\b"
    r"|\byour (?:name|capabilities|features|purpose)\b|\bhelp me\b",
    re.IGNORECASE,
)


def is_conversational(message: str) -> bool:
    """Small talk or a question about the assistant — safe to answer without research.

    Looser than the intent router: it only *confirms* a "direct" decision the
    planner model already made, it never creates one.
    """
    text = message.strip()
    if _SMALL_TALK.match(text) and len(text.split()) <= 8:
        return True
    return bool(_ABOUT_ASSISTANT.search(text)) and len(text.split()) <= 15


def fallback_plan(question: str, has_documents: bool) -> Plan:
    """Used when the planner model fails: search everything we have."""
    return Plan(
        standalone_question=question,
        route="both" if has_documents else "web",
        doc_queries=[question] if has_documents else [],
        web_queries=[question],
        rationale="Planner unavailable; searching all sources.",
    )


def _start_rework(question: str, last: dict, config: RunnableConfig) -> dict:
    """Follow-up that reshapes the previous answer: restore its evidence with the
    same citation numbers, so the rewrite stays cited and can be fact-checked,
    without searching again."""
    registry = registry_from_config(config)
    for e in last.get("evidence") or []:
        registry.register(e["key"], e["source"], e.get("text"), e.get("ref"))
    emit_sources("previous_answer", list(registry.sources))
    plan = Plan(
        standalone_question=question,
        route="rework",
        rationale="Reworking the previous answer from the same sources.",
    )
    return _finish_plan(plan, question)


def plan_node(state: ResearchState, config: RunnableConfig) -> dict:
    question = latest_user_message(state)
    events.agent_start("planner", "Understanding the question")
    last = state.get("last_answer")

    # Deterministic front door: small talk and follow-up rewrites never reach
    # the planner model (it once turned "thank you" into a web search).
    decision = classify(question, has_previous_answer=bool(last))
    emit(
        {
            "kind": "intent",
            "intent": decision.intent,
            "method": decision.method,
            "score": decision.score,
            "nearest": decision.nearest,
        }
    )
    if decision.intent == "chat":
        plan = Plan(
            standalone_question=question,
            route="direct",
            rationale=f"Small talk ({decision.method} match): replying without research.",
        )
        return _finish_plan(plan, question)
    if decision.intent == "rework" and last:
        return _start_rework(question, last, config)

    inventory = events.run_tool("planner", list_documents, {}, config)
    has_documents = not inventory.startswith("The document collection is empty")

    planner = with_fallback("planner", lambda m: m.with_structured_output(Plan))
    previous = f"YOUR PREVIOUS ANSWER WAS ABOUT: {last.get('question', '')}\n\n" if last else ""
    try:
        plan = planner.invoke(
            [
                SystemMessage(with_date(PLANNER)),
                HumanMessage(
                    f"DOCUMENT INVENTORY:\n{inventory}\n\n"
                    f"CONVERSATION SO FAR:\n{_history(state)}\n\n{previous}"
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
    if plan.route == "rework":
        if last:
            return _start_rework(question, last, config)
        # Nothing to rework yet: the chat reply asks what they mean. (Not a
        # factual question, so the research backstop below must not apply.)
        plan.route, plan.rationale = "direct", "Nothing to rework yet."
        return _finish_plan(plan, question)
    if plan.route == "direct" and not is_conversational(question):
        # Small models sometimes answer factual questions "directly" from memory;
        # anything that isn't small talk gets researched (and an honest no-answer
        # if nothing is found).
        plan.route = "both" if has_documents else "web"
        plan.rationale = "Factual question: researching instead of answering from memory."
    if plan.route == "decline":
        plan.doc_queries, plan.web_queries = [], []
    elif not has_documents and plan.route in ("documents", "both"):
        plan.route = "web"  # nothing to search locally
    return _finish_plan(plan, question)


def _finish_plan(plan: Plan, question: str) -> dict:
    """Normalize the plan, report it to the UI, and reset per-turn state."""
    plan.standalone_question = plan.standalone_question.strip() or question
    if plan.route in ("decline", "direct", "rework"):
        plan.doc_queries, plan.web_queries = [], []  # nothing gets searched
    else:
        question = plan.standalone_question
        plan.doc_queries = [q for q in plan.doc_queries if q.strip()][:3] or [question]
        plan.web_queries = [q for q in plan.web_queries if q.strip()][:2] or [question]

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
        "rework": ["compose"],
        "decline": ["decline"],
    }.get(route, ["doc_analyst", "web_researcher"])
