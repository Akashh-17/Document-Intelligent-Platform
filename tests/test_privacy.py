"""Internal names from the user's library never reach a public web search
while the documents are in play (the "Amazon Aurora" bug)."""

from __future__ import annotations

import pytest
from fakes import (
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
from test_agents import of, run_turn

from rag_agent.config import settings
from rag_agent.ingest import ensure_seeded
from rag_agent.privacy import internal_names, scrub_query


@pytest.fixture(scope="module", autouse=True)
def _seed():
    ensure_seeded()


@pytest.fixture
def searched(monkeypatch):
    """Capture what actually gets sent to the search engine."""
    sent: list[str] = []

    def fake_ddg(query):
        sent.append(query)
        return [
            {
                "title": "Mutual TLS",
                "url": "https://example.com/mtls",
                "content": "mTLS: both sides present certificates.",
                "score": None,
            }
        ]

    stub_web(monkeypatch, fake_ddg)
    return sent


def test_library_name_is_detected_but_generic_words_are_not():
    names = internal_names()
    assert "aurora" in names
    assert not {"reference", "runbook", "changelog", "error", "configuration"} & set(names)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Aurora mutual TLS authentication", "mutual TLS authentication"),
        ("Aurora's mTLS setup", "mTLS setup"),
        ("what is mutual TLS", "what is mutual TLS"),
        ("aurora", ""),
        ("Amazon Aurora mutual TLS", "mutual TLS"),  # public homonym dropped too
        ("how does Aurora's mTLS work", "how does mTLS work"),  # ordinary words kept
        ("Explain Aurora TLS setup", "Explain TLS setup"),  # acronyms kept
    ],
)
def test_scrub_query(query, expected):
    assert scrub_query(query, ["aurora"])[0] == expected


def test_explicit_private_terms_setting(monkeypatch):
    monkeypatch.setattr(settings, "private_terms", "Helios, Zephyr")
    assert {"helios", "zephyr", "aurora"} <= set(internal_names())


def _both_route_scripts(web_query: str) -> dict:
    q = "What is mTLS, and how does Aurora use it?"
    return {
        PLANNER: [plan("both", q, ["Aurora mTLS"], ["what is mutual TLS"])],
        DOCS: [tool_msg("", call("search_documents", "d1", query="Aurora mTLS")), AIMessage("ok")],
        WEB: [tool_msg("", call("web_search", "w1", query=web_query)), AIMessage("ok")],
        WRITER: [AIMessage("mTLS means both sides present certificates [1].")],
        JUDGE: [verdict()],
    }


async def test_internal_name_is_kept_out_of_web_search_when_docs_are_in_play(searched):
    model = ScriptedChatModel(scripts=_both_route_scripts("Amazon Aurora mutual TLS"))
    events = await run_turn(model, "What is mTLS, and how does Aurora use it?")
    # anchor search (planner's public query) + the agent's own, scrubbed query:
    # neither "Aurora" nor the "Amazon" homonym ever left the server
    assert searched == ["what is mutual TLS", "mutual TLS"]
    assert not any("aurora" in s.lower() or "amazon" in s.lower() for s in searched)
    notes = [d for d in of(events, "guardrail") if d["action"] == "private"]
    assert notes and "Aurora" in notes[0]["detail"]
    web_task = next(msgs for key, msgs in model.calls if key == WEB)[-1].text
    assert "never put them in a web query" in web_task and "aurora" in web_task


async def test_pure_web_question_is_not_scrubbed(searched):
    q = "What is the Aurora borealis?"
    model = ScriptedChatModel(
        scripts={
            PLANNER: [plan("web", q, [], ["Aurora borealis"])],
            WEB: [tool_msg("", call("web_search", "w1", query="Aurora borealis")), AIMessage("ok")],
            WRITER: [AIMessage("Polar lights [1].")],
            JUDGE: [verdict()],
        }
    )
    await run_turn(model, q)
    # the user meant the public thing: nothing is scrubbed (anchor + agent query)
    assert searched == [q, "Aurora borealis"]


async def test_privacy_can_be_turned_off(searched, monkeypatch):
    monkeypatch.setattr(settings, "web_query_privacy", False)
    model = ScriptedChatModel(scripts=_both_route_scripts("Aurora mutual TLS"))
    await run_turn(model, "What is mTLS, and how does Aurora use it?")
    assert searched == ["what is mutual TLS", "Aurora mutual TLS"]
