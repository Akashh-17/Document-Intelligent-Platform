"""Progress events the agents publish on LangGraph's custom stream channel.

The SSE layer forwards them to the UI, which draws one timeline section per
agent with that agent's tool calls nested inside.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from rag_agent.tools.common import emit

AGENT_LABELS = {
    "planner": "Query Planner",
    "doc_analyst": "Document Analyst",
    "web_researcher": "Web Researcher",
    "verifier": "Fact Verifier",
}


def agent_start(agent: str, detail: str = "") -> None:
    emit(
        {
            "kind": "agent",
            "status": "start",
            "agent": agent,
            "label": AGENT_LABELS[agent],
            "detail": detail,
        }
    )


def agent_end(agent: str, summary: str = "", ok: bool = True) -> None:
    emit({"kind": "agent", "status": "end", "agent": agent, "summary": summary, "ok": ok})


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _preview(result: Any, limit: int = 500) -> str:
    """Human-readable one-liner for the timeline (verifier tools return dicts)."""
    if isinstance(result, dict) and "invalid" in result:  # verify_citations
        parts = [f"cites {result['cited'] or 'nothing'}"]
        if result["invalid"]:
            parts.append(f"invented: {result['invalid']}")
        if result["uncited_sentences"]:
            parts.append(_plural(len(result["uncited_sentences"]), "uncited sentence"))
        return ", ".join(parts)
    if isinstance(result, dict) and "grounded" in result:  # check_groundedness
        if result.get("grounded") is None:
            return f"judge unavailable: {result.get('error', '')}"[:limit]
        verdict = "supported" if result["grounded"] else "NOT fully supported"
        text = f"{verdict} (score {result.get('score', 0):.2f})"
        if result.get("unsupported_claims"):
            text += f"; unsupported: {'; '.join(result['unsupported_claims'])}"
        return text[:limit]
    text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    return text[:limit]


def run_tool(agent: str, tool: BaseTool, args: dict, config: RunnableConfig) -> Any:
    """Call a tool directly from a node, publishing the same start/end events
    the UI gets for tool calls made inside the ReAct sub-agents."""
    call_id = f"{agent}-{tool.name}-{uuid.uuid4().hex[:8]}"
    emit({"kind": "tool", "status": "start", "agent": agent, "id": call_id, "tool": tool.name})
    try:
        result = tool.invoke(args, config)
    except Exception as exc:
        emit(
            {
                "kind": "tool",
                "status": "end",
                "agent": agent,
                "id": call_id,
                "tool": tool.name,
                "ok": False,
                "result_preview": f"{type(exc).__name__}: {exc}",
            }
        )
        raise
    emit(
        {
            "kind": "tool",
            "status": "end",
            "agent": agent,
            "id": call_id,
            "tool": tool.name,
            "ok": True,
            "result_preview": _preview(result),
        }
    )
    return result
