"""Guardrail nodes: the entry and exit gates of the research graph.

    START -> guard_input -> plan -> ... -> guard_output -> END
                  |                             ^
                  +----------- blocked ---------+

- ``guard_input``  masks personal data in the user's message (the masked text
  replaces it in history, so the raw value never reaches Groq, the web search,
  or the checkpoint), then blocks prompt-injection / jailbreak attempts.
- ``decline``      the Query Planner's "decline" route for harmful or
  out-of-scope requests: a fixed, polite refusal (no LLM improvisation).
- ``guard_output`` every answer leaves through here: secrets / IDs redacted,
  system-prompt leakage replaced. It also appends the final answer to the
  conversation history — so nothing unchecked is ever remembered.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from rag_agent.agents import events
from rag_agent.agents.state import ResearchState
from rag_agent.citations import registry_from_config
from rag_agent.config import settings
from rag_agent.guardrails import check_injection, check_output, mask_pii
from rag_agent.tools.common import emit

BLOCKED_INJECTION = (
    "I can't process this request: it looks like an attempt to override my instructions "
    "(prompt injection). If that wasn't your intent, rephrase it as a plain question about "
    "your documents or a topic to look up."
)
DECLINED = (
    "I can't help with that. {reason} I can answer questions about your documents, or look "
    "things up on the web, with citations."
)


def _guardrail(stage: str, action: str, detail: str) -> None:
    emit({"kind": "guardrail", "stage": stage, "action": action, "detail": detail})


def _blocked(category: str, message: str, reasons: list[str]) -> dict:
    emit({"kind": "blocked", "category": category, "message": message, "reasons": reasons})
    verification = {"checked": False, "confidence": "n/a", "blocked": category}
    emit({"kind": "verification", **verification})
    return {
        "blocked": {"category": category, "reasons": reasons},
        "answer": message,
        "verification": verification,
        "plan": {"route": "blocked"},
    }


def guard_input_node(state: ResearchState, config: RunnableConfig) -> dict:
    human = next(
        (m for m in reversed(state.get("messages", [])) if isinstance(m, HumanMessage)), None
    )
    if human is None or not settings.guardrails_enabled:
        return {"blocked": None}

    events.agent_start("guardrails", "Checking the request")
    update: dict = {"blocked": None}
    text = human.text

    if settings.pii_masking:
        masked = mask_pii(text)
        if masked.found:
            text = masked.text
            # same id -> add_messages REPLACES the stored message with the masked one
            update["messages"] = [HumanMessage(content=text, id=human.id)]
            _guardrail(
                "input",
                "masked",
                f"Masked {masked.summary} before sending anything to the AI or the web.",
            )

    verdict = check_injection(text)
    if verdict.blocked:
        _guardrail("input", "blocked", "Possible prompt injection: " + "; ".join(verdict.reasons))
        events.agent_end("guardrails", "Blocked: possible prompt injection.", ok=False)
        return {**update, **_blocked("prompt_injection", BLOCKED_INJECTION, verdict.reasons)}

    classifier = (
        f"Prompt Guard {verdict.classifier_score:.2f}"
        if verdict.classifier_score is not None
        else "heuristics only"
    )
    events.agent_end("guardrails", f"Request passed input checks ({classifier}).")
    return update


def route_after_guard(state: ResearchState) -> str:
    return "guard_output" if state.get("blocked") else "plan"


def decline_node(state: ResearchState, config: RunnableConfig) -> dict:
    reason = ((state.get("plan") or {}).get("rationale") or "").strip()
    reason = (reason.rstrip(".") + ".") if reason else ""
    _guardrail("input", "declined", f"Planner declined the request. {reason}".strip())
    return _blocked("policy", DECLINED.format(reason=reason).replace("  ", " "), [reason])


def guard_output_node(state: ResearchState, config: RunnableConfig) -> dict:
    answer = state.get("answer", "")
    if settings.guardrails_enabled and answer:
        check = check_output(answer)
        if check.changed:
            answer = check.text
            emit({"kind": "replace_answer", "text": answer})
            if check.leaked_prompt:
                _guardrail("output", "blocked", "The answer reproduced internal instructions.")
            else:
                found = ", ".join(
                    f"{n} {k.lower().replace('_', ' ')}" for k, n in check.redacted.items()
                )
                _guardrail(
                    "output", "redacted", f"Redacted sensitive data from the answer: {found}."
                )
    update: dict = {"answer": answer, "messages": [AIMessage(content=answer)]}
    remembered = _remember(state, answer, config)
    if remembered is not None:
        update["last_answer"] = remembered
    return update


_RESEARCH_ROUTES = ("documents", "web", "both", "rework")


def _remember(state: ResearchState, answer: str, config: RunnableConfig) -> dict | None:
    """Keep a researched answer and its evidence for follow-ups ("make it shorter").

    Only answers that went through research and passed the verifier qualify:
    small talk, refusals and "couldn't find it" responses leave the previous
    researched answer in place, so "thanks!" followed by "now as bullets"
    still reworks the real answer.
    """
    route = (state.get("plan") or {}).get("route")
    verification = state.get("verification") or {}
    registry = registry_from_config(config)
    if (
        route not in _RESEARCH_ROUTES
        or state.get("blocked")
        or not answer
        or not len(registry)
        or verification.get("confidence") in (None, "none", "n/a")
    ):
        return None
    last = state.get("last_answer") or {}
    return {
        # a rework answers the same underlying question as the answer it reshaped
        "question": last.get("question") if route == "rework" else state.get("question", ""),
        "answer": answer,
        "evidence": registry.export(),
    }
