"""Scripted chat model for running the agent graph offline (no API key).

Replies are scripted per agent: the model looks at the system prompt to see
which agent is calling (the Document Analyst and Web Researcher run in
parallel and share the "worker" model, so a single queue would race).
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import ConfigDict, PrivateAttr

Reply = AIMessage | dict | Exception | Callable[[list[BaseMessage]], Any]


class ScriptedChatModel(BaseChatModel):
    """``scripts`` maps a substring of the system prompt -> list of replies.

    A reply may be an AIMessage (text and/or tool_calls), a dict (returned by
    ``with_structured_output``), an Exception (raised), or a callable that
    receives the messages and returns one of those.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    scripts: dict[str, list[Any]]
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    calls: list[tuple[str, list[BaseMessage]]] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _next(self, messages: list[BaseMessage]) -> Any:
        system = next((m.text for m in messages if isinstance(m, SystemMessage)), "")
        with self._lock:
            for key, queue in self.scripts.items():
                if key in system:
                    if not queue:
                        raise AssertionError(f"script for {key!r} exhausted")
                    self.calls.append((key, messages))
                    reply = queue.pop(0)
                    break
            else:
                raise AssertionError(f"no script matches system prompt: {system[:80]!r}")
        if callable(reply) and not isinstance(reply, AIMessage | dict):
            reply = reply(messages)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        def run(messages):
            msgs = messages.to_messages() if hasattr(messages, "to_messages") else messages
            reply = self._next(msgs)
            return schema.model_validate(reply) if isinstance(reply, dict) else reply

        return RunnableLambda(run)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        reply = self._next(messages)
        return ChatResult(generations=[ChatGeneration(message=reply)])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        reply: AIMessage = self._next(messages)
        chunk = AIMessageChunk(
            content=reply.content,
            tool_call_chunks=[
                {
                    "name": tc["name"],
                    "args": json.dumps(tc["args"]),
                    "id": tc["id"],
                    "index": i,
                    "type": "tool_call_chunk",
                }
                for i, tc in enumerate(reply.tool_calls)
            ],
        )
        if run_manager and reply.content:
            run_manager.on_llm_new_token(reply.text, chunk=ChatGenerationChunk(message=chunk))
        yield ChatGenerationChunk(message=chunk)


def call(tool: str, call_id: str, **args) -> dict:
    return {"name": tool, "args": args, "id": call_id, "type": "tool_call"}


def tool_msg(text: str = "", *calls: dict) -> AIMessage:
    return AIMessage(content=text, tool_calls=list(calls))


def plan(route: str, question: str, doc_queries=(), web_queries=()) -> dict:
    return {
        "standalone_question": question,
        "route": route,
        "doc_queries": list(doc_queries),
        "web_queries": list(web_queries),
        "rationale": f"test route {route}",
    }


def verdict(grounded: bool = True, score: float = 1.0, unsupported=()) -> dict:
    return {
        "grounded": grounded,
        "score": score,
        "unsupported_claims": list(unsupported),
        "wrong_citations": [],
    }


# Keys matching each agent's system prompt (see rag_agent/agents/prompts.py).
PLANNER = "Query Planner"
DOCS = "Document Analyst"
WEB = "Web Researcher"
WRITER = "writing the final answer"
DIRECT = "needs no research"
JUDGE = "strict fact-checker"
SUMMARIZER = "Summarize the document"
