"""The multi-agent research graph.

                       ┌──────────────────┐
                       │  Query Planner   │  rewrite + route (structured output)
                       └────────┬─────────┘
             documents ┌────────┼─────────┐ web            direct
                       ▼   both (parallel) ▼                  │
            ┌──────────────────┐   ┌──────────────────┐       │
            │ Document Analyst │──▶│  Web Researcher  │       │
            │  (ReAct, 7 tools)│ * │ (ReAct, 3 tools) │       │
            └────────┬─────────┘   └────────┬─────────┘       │
                     └───────────┬──────────┘                 │
                                 ▼                            │
                          assess evidence ──── none ───┐      │
                                 │ found               ▼      │
                       ┌─────────▼────────┐   ┌──────────────┐│
                       │  Fact Verifier   │   │ "Couldn't    ││
                       │ compose ─▶ verify│──▶│  find it"    ││
                       │    ▲         │   │ ** │ (no LLM)     ││
                       │    └─revise──┘   │   └──────────────┘│
                       └──────────────────┘ ◀─────────────────┘

  *  escalation: a documents-only plan that finds nothing tries the web.
  ** the verifier hands over when the evidence doesn't answer the question.
  Gap filling: if the answer is only partial and the web wasn't searched, the
  verifier sends the Web Researcher after the missing parts, then revises.

Guardrails wrap the whole graph: ``guard_input`` (PII masking, prompt-injection
blocking) runs first, ``guard_output`` (redaction, prompt-leak check) last; the
planner can also route harmful requests to ``decline``. See agents/guard.py.

All agents share one per-turn ``CitationRegistry`` (passed in the run config),
so evidence found by any agent gets a stable citation number.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from rag_agent.agents.guard import (
    decline_node,
    guard_input_node,
    guard_output_node,
    route_after_guard,
)
from rag_agent.agents.no_answer import assess_node, no_answer_node, route_after_assess
from rag_agent.agents.planner import plan_node, route_after_plan
from rag_agent.agents.researchers import make_research_node, route_after_docs
from rag_agent.agents.state import ResearchState
from rag_agent.agents.verifier import compose_node, route_after_verify, verify_node


def build_graph(checkpointer=None):
    g = StateGraph(ResearchState)
    g.add_node("guard_input", guard_input_node)
    g.add_node("plan", plan_node)
    g.add_node("doc_analyst", make_research_node("doc_analyst"))
    g.add_node("web_researcher", make_research_node("web_researcher"))
    g.add_node("assess", assess_node)
    g.add_node("compose", compose_node)
    g.add_node("verify", verify_node)
    g.add_node("no_answer", no_answer_node)
    g.add_node("decline", decline_node)
    g.add_node("guard_output", guard_output_node)

    g.add_edge(START, "guard_input")
    g.add_conditional_edges("guard_input", route_after_guard, ["plan", "guard_output"])
    g.add_conditional_edges(
        "plan", route_after_plan, ["doc_analyst", "web_researcher", "compose", "decline"]
    )
    g.add_conditional_edges("doc_analyst", route_after_docs, ["web_researcher", "assess"])
    g.add_edge("web_researcher", "assess")
    g.add_conditional_edges("assess", route_after_assess, ["compose", "no_answer"])
    g.add_edge("compose", "verify")
    g.add_conditional_edges(
        "verify", route_after_verify, ["compose", "web_researcher", "no_answer", "guard_output"]
    )
    g.add_edge("no_answer", "guard_output")
    g.add_edge("decline", "guard_output")
    g.add_edge("guard_output", END)
    return g.compile(checkpointer=checkpointer, name="research_team")
