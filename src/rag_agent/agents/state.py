"""Graph state shared by the four agents.

``messages`` is the conversation (checkpointed per thread, so follow-up
questions have memory). Everything else is per-turn working state that the
Planner resets at the start of each turn.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

Route = Literal["documents", "web", "both", "direct"]


class Plan(BaseModel):
    """The Query Planner's structured decision for one turn."""

    standalone_question: str = Field(
        description="The user's question rewritten to be self-contained, resolving "
        "pronouns and references to earlier turns."
    )
    route: Route = Field(
        description="documents: answerable from the private documents. web: needs public "
        "or current information. both: needs documents AND external context. direct: "
        "greeting, thanks, or a question about the assistant itself (no research)."
    )
    doc_queries: list[str] = Field(
        default_factory=list,
        description="Up to 3 focused search queries for the documents (keep exact "
        "identifiers like error codes verbatim).",
    )
    web_queries: list[str] = Field(
        default_factory=list, description="Up to 2 focused web search queries."
    )
    rationale: str = Field(default="", description="One short sentence explaining the route.")


def merge_findings(old: list[dict] | None, new: list[dict] | None) -> list[dict]:
    """Reducer: parallel agents append their findings; ``None`` resets for a new turn."""
    if new is None:
        return []
    return [*(old or []), *new]


class ResearchState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    question: str  # standalone question for this turn
    plan: dict[str, Any]
    findings: Annotated[list[dict[str, Any]], merge_findings]
    answer: str
    verification: dict[str, Any] | None
    revisions: int
    feedback: str  # verifier's notes for a revision
    evidence_count: int  # set by the "assess" join after research
    unanswered: bool  # verifier: the evidence doesn't answer the question
    degraded: bool  # writer models were down; answer is extractive quotes
    gap_fill: list[str]  # verifier: aspects the documents missed -> web researcher
