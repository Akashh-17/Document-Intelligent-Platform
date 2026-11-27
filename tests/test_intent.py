"""The intent router: chat like a person, rework follow-ups without re-searching,
and research everything else.

Bugs this guards against (found by using the app):
- "thank you" was routed to the web and answered with a wikiHow citation;
- "hey wassup" ran a web search.
"""

from __future__ import annotations

import pytest
from fakes import (
    DIRECT,
    DOCS,
    JUDGE,
    PLANNER,
    WRITER,
    ScriptedChatModel,
    call,
    plan,
    tool_msg,
    verdict,
)
from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from test_agents import answer, of, run_turn

from rag_agent.agents.intent import classify, is_small_talk
from rag_agent.ingest import ensure_seeded

REWORK = "reworking your previous answer"  # WRITER_REWORK system prompt


@pytest.fixture(scope="module", autouse=True)
def _seed():
    ensure_seeded()


# --- layer 1: lexicon -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("thank you", True),
        ("Thanks a lot!", True),
        ("ok great, thank you so much 🙏", True),
        ("hey wassup", True),
        ("heyy what's up bro", True),
        ("how are you doing today?", True),
        ("lol ok", True),
        ("thanks, what does AUR-2005 mean?", False),  # a real question after the thanks
        ("hey, what is mTLS?", False),
        ("👍", False),  # no words: let the other layers decide
    ],
)
def test_lexicon(text, expected):
    assert is_small_talk(text) is expected


# --- layer 2: semantic router (held-out phrasings, not in the example lists) --------------


@pytest.mark.parametrize(
    "text",
    [
        "how's life",
        "thank you so much, that was great",
        "lol that's funny",
        "who made you?",
        "what can you help with?",
        "what's your name buddy",
        "hello! how is it going",
        "can you help me?",
    ],
)
def test_semantic_chat(text):
    assert classify(text).intent == "chat"


@pytest.mark.parametrize(
    "text",
    [
        "what is kubernetes",
        "hey what's the capital of france",
        "what's up with bitcoin price today",
        "summarize my document",
        "what does error AUR-3001 mean",
        "hi can you explain RAG",
        "what time is it in tokyo",
        "give me 5 interview questions on python",
        "what's new in python 3.14",
        "thanks! also what is mTLS?",
        "who are you voting for",  # closest call in calibration (0.72 < 0.75)
    ],
)
def test_real_questions_go_to_the_planner(text):
    assert classify(text, has_previous_answer=True).intent == "task"


@pytest.mark.parametrize(
    "text",
    [
        "shorter please",
        "make it simpler",
        "translate it into french",
        "bullet points please",
        "can you make that a table",
        "eli5",
        "summarize your answer",
    ],
)
def test_rework_needs_a_previous_answer(text):
    assert classify(text, has_previous_answer=True).intent == "rework"
    assert classify(text, has_previous_answer=False).intent != "rework"


# --- end to end ----------------------------------------------------------------------------


@pytest.mark.parametrize("message", ["thank you", "hey wassup", "who made you?"])
async def test_chat_never_triggers_research(message):
    model = ScriptedChatModel(
        scripts={
            # the planner model would have searched the web; it must not be asked
            PLANNER: [plan("web", "how to respond", [], ["how to respond to thank you"])],
            DIRECT: [AIMessage("Hey! What can I look up for you?")],
        }
    )
    events = await run_turn(model, message)
    assert of(events, "plan")[0]["route"] == "direct"
    assert of(events, "intent")[0]["intent"] == "chat"
    assert answer(events) == "Hey! What can I look up for you?"
    assert not any(k == PLANNER for k, _ in model.calls)
    assert not of(events, "tool_start") and not of(events, "sources")


def _researched_turn() -> ScriptedChatModel:
    return ScriptedChatModel(
        scripts={
            PLANNER: [plan("documents", "What is AUR-3001?", ["AUR-3001"])],
            DOCS: [
                tool_msg("", call("search_documents", "d1", query="AUR-3001")),
                AIMessage("Found it."),
            ],
            WRITER: [AIMessage("AUR-3001 means the client sent no certificate [1].")],
            JUDGE: [verdict()],
        }
    )


async def test_follow_up_rework_reuses_evidence_without_searching():
    rework = ScriptedChatModel(
        scripts={
            REWORK: [AIMessage("No client cert [1].")],
            JUDGE: [verdict()],
        }
    )
    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        first = await run_turn(_researched_turn(), "What is AUR-3001?", "rw", saver)
        events = await run_turn(rework, "make it shorter", "rw", saver)

    assert of(events, "plan")[0]["route"] == "rework"
    assert not of(events, "tool_start") or all(
        d["agent"] == "verifier" for d in of(events, "tool_start")
    )  # no searching, only the fact-check
    assert answer(events) == "No client cert [1]."
    # the same evidence, with the same citation numbers, is shown again
    before = {s["index"]: s["title"] for d in of(first, "sources") for s in d["sources"]}
    after = {s["index"]: s["title"] for d in of(events, "sources") for s in d["sources"]}
    assert after == before
    # the rewrite is fact-checked like any answer
    assert of(events, "verification")[0]["confidence"] == "high"
    prompt = rework.calls[0][1][-1].text
    assert "AUR-3001 means the client sent no certificate [1]." in prompt


async def test_rework_skips_small_talk_in_between():
    chat = ScriptedChatModel(scripts={DIRECT: [AIMessage("Anytime!")]})
    rework = ScriptedChatModel(scripts={REWORK: [AIMessage("- No cert [1]")], JUDGE: [verdict()]})
    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        await run_turn(_researched_turn(), "What is AUR-3001?", "rw2", saver)
        await run_turn(chat, "thanks!", "rw2", saver)
        events = await run_turn(rework, "bullet points please", "rw2", saver)
    assert of(events, "plan")[0]["route"] == "rework"
    assert "AUR-3001 means the client sent no certificate" in rework.calls[0][1][-1].text


async def test_planner_rework_without_a_previous_answer_becomes_chat():
    model = ScriptedChatModel(
        scripts={
            PLANNER: [plan("rework", "rewrite the answer as a poem")],
            DIRECT: [AIMessage("Happy to — which answer should I turn into a poem?")],
        }
    )
    events = await run_turn(model, "rewrite the answer as a poem")
    assert of(events, "plan")[0]["route"] == "direct"
    assert not of(events, "sources")
