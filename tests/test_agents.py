"""The four-agent graph end to end, driven by scripted fake models.

Each test scripts what every agent's model replies and asserts on the SSE
events the UI would receive.
"""

from __future__ import annotations

import json

import pytest
from fakes import (
    DIRECT,
    DOCS,
    JUDGE,
    PLANNER,
    WEB,
    WRITER,
    ScriptedChatModel,
    call,
    plan,
    stub_web,
    tool_msg,
    verdict,
)
from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from rag_agent.agent import build_agent
from rag_agent.ingest import ensure_seeded
from rag_agent.llms import override_models
from rag_agent.streaming import agent_event_stream

ROLES = ("planner", "worker", "writer", "verifier", "summarizer")


@pytest.fixture(scope="module", autouse=True)
def _seed():
    ensure_seeded()


@pytest.fixture(autouse=True)
def _fake_web(monkeypatch):
    stub_web(
        monkeypatch,
        lambda q: [
            {
                "title": "Mutual TLS - overview",
                "url": "https://example.com/mtls",
                "content": "Mutual TLS (mTLS) means both client and server present certificates.",
                "score": None,
            }
        ],
    )


async def run_turn(model, message: str, thread_id: str = "t-1", saver=None):
    """Run one chat turn through the real graph + SSE layer; return parsed events."""

    async def go(checkpointer):
        graph = build_agent(checkpointer)
        return [
            (ev["event"], json.loads(ev["data"]))
            async for ev in agent_event_stream(graph, message=message, thread_id=thread_id)
        ]

    with override_models(dict.fromkeys(ROLES, model)):
        if saver is not None:
            return await go(saver)
        async with AsyncSqliteSaver.from_conn_string(":memory:") as s:
            return await go(s)


def of(events, kind):
    return [d for k, d in events if k == kind]


def answer(events):
    return "".join(d["delta"] for d in of(events, "token"))


# --- happy paths -----------------------------------------------------------------


async def test_both_route_runs_research_agents_in_parallel_and_verifies():
    model = ScriptedChatModel(
        scripts={
            PLANNER: [plan("both", "What is AUR-3001 and what is mTLS?", ["AUR-3001"], ["mTLS"])],
            DOCS: [
                tool_msg("", call("search_documents", "d1", query="AUR-3001")),
                AIMessage("AUR-3001 is CLIENT_CERT_MISSING."),
            ],
            WEB: [
                tool_msg("", call("web_search", "w1", query="mTLS")),
                AIMessage("mTLS = mutual TLS."),
            ],
            WRITER: [AIMessage("AUR-3001 means no client certificate was sent.")],
            JUDGE: [verdict(grounded=True, score=1.0)],
        }
    )
    events = await run_turn(model, "What is AUR-3001 and what is mTLS?")

    started = [d["agent"] for d in of(events, "agent_start")]
    assert started[:2] == ["guardrails", "planner"] and started[-1] == "verifier"
    assert {"doc_analyst", "web_researcher"} <= set(started)
    assert of(events, "plan")[0]["route"] == "both"

    # tool calls are attributed to the agent that made them
    tools = {(d["agent"], d["tool"]) for d in of(events, "tool_start")}
    assert {
        ("planner", "list_documents"),
        ("doc_analyst", "search_documents"),
        ("web_researcher", "web_search"),
        ("verifier", "verify_citations"),
        ("verifier", "check_groundedness"),
    } <= tools

    # one numbering across both agents: no duplicate citation numbers
    indexes = [s["index"] for d in of(events, "sources") for s in d["sources"]]
    assert len(indexes) == len(set(indexes)) and min(indexes) == 1
    kinds = {s["kind"] for d in of(events, "sources") for s in d["sources"]}
    assert kinds == {"document", "web"}

    # only the writer's text is streamed — no planner JSON, no research notes
    assert answer(events) == "AUR-3001 means no client certificate was sent."
    v = of(events, "verification")[0]
    assert v["confidence"] == "high" and v["checked"] is True
    assert not of(events, "error")


async def test_direct_route_skips_research_and_fact_check():
    model = ScriptedChatModel(
        scripts={
            PLANNER: [plan("direct", "hello")],
            DIRECT: [AIMessage("Hi! Ask me about your documents.")],
        }
    )
    events = await run_turn(model, "hello")
    assert [d["agent"] for d in of(events, "agent_start")] == ["guardrails", "planner", "verifier"]
    assert answer(events) == "Hi! Ask me about your documents."
    assert of(events, "verification")[0]["confidence"] == "n/a"
    assert not any(k == JUDGE for k, _ in model.calls)


# --- self-correction & escalation ------------------------------------------------


async def test_verifier_sends_unsupported_draft_back_for_revision():
    model = ScriptedChatModel(
        scripts={
            PLANNER: [plan("documents", "What is the max event size?", ["max event size"])],
            DOCS: [
                tool_msg("", call("search_documents", "d1", query="max event size")),
                AIMessage("Found the limit."),
            ],
            WRITER: [
                AIMessage("The limit is 256 KB [1], and it was raised in 2019 [9]."),
                AIMessage("The limit is 256 KB [1]."),
            ],
            JUDGE: [
                verdict(grounded=False, score=0.5, unsupported=["raised in 2019"]),
                verdict(grounded=True, score=1.0),
            ],
        }
    )
    events = await run_turn(model, "What is the max event size?")
    revision = of(events, "revision")
    assert len(revision) == 1 and "2019" in revision[0]["reason"]
    assert "[9]" in revision[0]["reason"] or "do not exist" in revision[0]["reason"]

    # the writer saw the feedback the second time
    writer_calls = [msgs for key, msgs in model.calls if key == WRITER]
    assert len(writer_calls) == 2 and "FACT-CHECK FEEDBACK" in writer_calls[1][-1].text

    # the UI clears on "revision"; the text after it is the final answer
    after = events[[k for k, _ in events].index("revision") :]
    assert answer(after) == "The limit is 256 KB [1]."
    v = of(events, "verification")[0]
    assert v["confidence"] == "high" and v["revisions"] == 1


async def test_documents_route_escalates_to_web_when_nothing_found():
    model = ScriptedChatModel(
        scripts={
            PLANNER: [plan("documents", "Best chocolate cake recipe?", ["chocolate cake"])],
            DOCS: [
                tool_msg("", call("search_documents", "d1", query="chocolate cake recipe")),
                AIMessage("Nothing relevant in the documents."),
            ],
            WEB: [
                tool_msg("", call("web_search", "w1", query="chocolate cake recipe")),
                AIMessage("Found a web result."),
            ],
            WRITER: [AIMessage("From the web [1].")],
            JUDGE: [verdict()],
        }
    )
    events = await run_turn(model, "Best chocolate cake recipe?")
    assert of(events, "escalation")[0]["to"] == "web_researcher"
    doc_end = next(d for d in of(events, "agent_end") if d["agent"] == "doc_analyst")
    assert doc_end["summary"].startswith("0 source(s) found")
    assert "web_researcher" in [d["agent"] for d in of(events, "agent_start")]


# --- resilience ------------------------------------------------------------------


async def test_planner_failure_falls_back_to_searching_everything():
    model = ScriptedChatModel(
        scripts={
            PLANNER: [RuntimeError("planner model down")],
            DOCS: [AIMessage("Skipped tools.")],
            WEB: [AIMessage("Skipped tools.")],
        }
    )
    events = await run_turn(model, "What does AUR-2005 mean?")
    assert of(events, "plan")[0]["route"] == "both"
    # the scripted agents searched nothing, so the user gets the structured
    # "couldn't find it" response — and the writer is never asked to improvise
    assert of(events, "no_answer") and not any(k == WRITER for k, _ in model.calls)
    assert of(events, "verification")[0]["confidence"] == "none"
    assert not of(events, "error")


async def test_research_agent_crash_falls_back_to_direct_search():
    model = ScriptedChatModel(
        scripts={
            PLANNER: [plan("both", "q", ["AUR-2005"], ["x"])],
            DOCS: [
                tool_msg("", call("search_documents", "d1", query="AUR-2005")),
                AIMessage("ok"),
            ],
            WEB: [RuntimeError("worker exploded")],
            WRITER: [AIMessage("AUR-2005 is CROSS_TENANT_JOIN [1].")],
            JUDGE: [verdict()],
        }
    )
    events = await run_turn(model, "q")
    web_end = next(d for d in of(events, "agent_end") if d["agent"] == "web_researcher")
    # the model crashed, but the planned query still ran straight through web_search
    assert "ran the planned searches directly" in web_end["summary"]
    assert ("web_researcher", "web_search") in {
        (d["agent"], d["tool"]) for d in of(events, "tool_start")
    }
    assert answer(events) == "AUR-2005 is CROSS_TENANT_JOIN [1]."
    assert not of(events, "error")


async def test_writer_failure_becomes_friendly_error():
    model = ScriptedChatModel(
        scripts={PLANNER: [plan("direct", "hi")], DIRECT: [TimeoutError("upstream timed out")]}
    )
    events = await run_turn(model, "hi")
    err = of(events, "error")
    assert err and err[0]["fatal"] and "try again" in err[0]["message"].lower()
    assert events[-1][0] == "end"


# --- conversation memory ---------------------------------------------------------


async def test_follow_up_question_sees_previous_turn():
    first = ScriptedChatModel(
        scripts={
            PLANNER: [plan("direct", "hi")],
            DIRECT: [AIMessage("Hello, I can help with Aurora.")],
        }
    )
    second = ScriptedChatModel(
        scripts={PLANNER: [plan("direct", "what did you say?")], DIRECT: [AIMessage("ok")]}
    )
    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        await run_turn(first, "hi", thread_id="mem", saver=saver)
        await run_turn(second, "what did you just say?", thread_id="mem", saver=saver)
    planner_prompt = second.calls[0][1][-1].text
    assert "Hello, I can help with Aurora." in planner_prompt
    assert "User: hi" in planner_prompt
