"""Translate the multi-agent graph's stream into a flat SSE event stream.

We stream with ``subgraphs=True`` so events from inside the research
sub-agents arrive too, each tagged with a namespace like
``("doc_analyst:<task-id>",)`` that tells us which agent produced it.

| LangGraph stream                      | SSE event(s)                                  |
|---------------------------------------|-----------------------------------------------|
| messages from the ``compose`` node    | ``token`` (only the final answer is streamed) |
| updates inside a sub-agent            | ``tool_start`` / ``tool_end`` (with ``agent``)|
| custom ``agent``                      | ``agent_start`` / ``agent_end``               |
| custom ``tool`` (planner / verifier)  | ``tool_start`` / ``tool_end``                 |
| custom ``sources``                    | ``sources``                                   |
| custom ``plan`` / ``escalation`` /    | ``plan`` / ``escalation`` /                   |
| ``revision`` / ``verification``       | ``revision`` / ``verification``               |
| custom ``no_answer``                  | ``no_answer`` (structured "couldn't find it") |
| custom ``degraded`` / ``replace_answer`` | ``degraded`` / ``replace_answer``          |

Errors carry ``retryable`` so the UI can offer a Retry button only when
trying again could help (rate limits, timeouts), not for a bad API key.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from rag_agent.agent import run_config
from rag_agent.config import settings

log = logging.getLogger(__name__)

ANSWER_NODE = "compose"
_CUSTOM_PASSTHROUGH = {
    "sources",
    "plan",
    "escalation",
    "revision",
    "verification",
    "no_answer",
    "degraded",
    "replace_answer",
    "guardrail",
    "blocked",
    "intent",
    "web_search",
}


def _sse(event: str, data: dict) -> dict:
    """Shape expected by sse_starlette.EventSourceResponse."""
    return {"event": event, "data": json.dumps(data, ensure_ascii=False, default=str)}


def _result_preview(content: Any, limit: int = 500) -> str:
    text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    return text[:limit]


def _agent_of(namespace: tuple[str, ...]) -> str:
    """``("doc_analyst:1234",)`` -> ``"doc_analyst"``; top level -> ``""``."""
    return namespace[0].split(":", 1)[0] if namespace else ""


def friendly_error(exc: BaseException) -> tuple[str, bool]:
    """Map provider/infra exceptions to ``(message a user can act on, retryable)``."""
    name = type(exc).__name__
    text = str(exc)
    if name == "MissingAPIKeyError":
        return text, False
    if name in {"AuthenticationError", "PermissionDeniedError"}:
        return "The Groq API key was rejected. Check GROQ_API_KEY in your .env file.", False
    if name == "RateLimitError" or "429" in text:
        return (
            "The language model is rate-limited right now (Groq free tier). "
            "Wait a few seconds and try again.",
            True,
        )
    if name in {"APITimeoutError", "APIConnectionError", "TimeoutError", "InternalServerError"}:
        return "Couldn't reach the language model (network/timeout). Please try again.", True
    if name == "GraphRecursionError":
        return (
            "The agents took too many steps without finishing. Try a more specific question.",
            False,
        )
    return f"Something went wrong while answering ({name}). Please try again.", True


def _custom_event(thread_id: str, chunk: dict) -> tuple[str, dict] | None:
    kind = chunk.get("kind")
    body = {k: v for k, v in chunk.items() if k != "kind"}
    if kind in ("agent", "tool"):
        status = body.pop("status", "start")
        return f"{kind}_{status}", {"thread_id": thread_id, **body}
    if kind in _CUSTOM_PASSTHROUGH:
        return kind, {"thread_id": thread_id, **body}
    return None


def _subagent_tool_events(thread_id: str, agent: str, chunk: dict, seen: set[str]):
    """tool_start / tool_end events from a research sub-agent's update."""
    for node_update in (chunk or {}).values():
        for m in (node_update or {}).get("messages", []) or []:
            if isinstance(m, AIMessage) and m.tool_calls:
                for tc in m.tool_calls:
                    tc_id = tc.get("id") or f"{agent}-{tc.get('name', '')}"
                    if tc_id in seen:
                        continue
                    seen.add(tc_id)
                    yield _sse(
                        "tool_start",
                        {
                            "thread_id": thread_id,
                            "agent": agent,
                            "id": tc_id,
                            "tool": tc.get("name", ""),
                            "args": tc.get("args", {}),
                        },
                    )
            if isinstance(m, ToolMessage):
                yield _sse(
                    "tool_end",
                    {
                        "thread_id": thread_id,
                        "agent": agent,
                        "id": m.tool_call_id,
                        "tool": m.name or "",
                        "ok": getattr(m, "status", "success") != "error",
                        "result_preview": _result_preview(m.content),
                    },
                )


async def agent_event_stream(
    graph,
    message: str,
    thread_id: str,
    is_disconnected=None,
) -> AsyncIterator[dict]:
    """Yield SSE event dicts for a single chat turn."""
    run_id = str(uuid.uuid4())
    # run_id pins the LangSmith root run so the UI can attach feedback to it.
    config = run_config(thread_id=thread_id, run_id=run_id)

    yield _sse("start", {"thread_id": thread_id, "run_id": run_id})

    try:
        async with asyncio.timeout(settings.turn_timeout_s):
            async for event in _stream_events(graph, message, thread_id, config, is_disconnected):
                yield event
    except TimeoutError:
        log.warning("turn timed out (thread=%s run=%s)", thread_id, run_id)
        yield _sse(
            "error",
            {
                "thread_id": thread_id,
                "message": f"This took longer than {int(settings.turn_timeout_s)}s, so I stopped. "
                "Try again, or ask a narrower question.",
                "fatal": True,
                "retryable": True,
            },
        )
    except Exception as exc:  # surface as a fatal SSE error, don't 500 mid-stream
        log.exception("agent run failed (thread=%s run=%s)", thread_id, run_id)
        message, retryable = friendly_error(exc)
        yield _sse(
            "error",
            {"thread_id": thread_id, "message": message, "fatal": True, "retryable": retryable},
        )

    yield _sse("end", {"thread_id": thread_id, "run_id": run_id})


async def _stream_events(graph, message, thread_id, config, is_disconnected):
    """The turn's graph stream, translated into SSE events (see module docstring)."""
    seen_tool_starts: set[str] = set()
    async for namespace, mode, chunk in graph.astream(
        {"messages": [{"role": "user", "content": message}]},
        config=config,
        stream_mode=["messages", "updates", "custom"],
        subgraphs=True,
    ):
        if is_disconnected is not None and await is_disconnected():
            break

        if mode == "messages":
            msg, meta = chunk
            is_answer = not namespace and meta.get("langgraph_node") == ANSWER_NODE
            if is_answer and isinstance(msg, AIMessageChunk) and msg.text:
                yield _sse("token", {"thread_id": thread_id, "delta": msg.text})

        elif mode == "updates" and namespace:
            for event in _subagent_tool_events(
                thread_id, _agent_of(namespace), chunk, seen_tool_starts
            ):
                yield event

        elif mode == "custom" and isinstance(chunk, dict):
            event = _custom_event(thread_id, chunk)
            if event:
                yield _sse(*event)
