"""Phase 3 — failure handling. What the user gets when things go wrong.

Covers: no evidence anywhere, web unavailable, empty knowledge base, evidence
that doesn't answer the question, partial answers, model fallback, every model
down (extractive answer), research-model crash, crashing tools, turn timeout,
and the opt-in "answer from general knowledge" policy.
"""

from __future__ import annotations

import asyncio

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
    tool_msg,
    verdict,
)
from langchain_core.messages import AIMessage
from test_agents import ROLES, answer, of, run_turn

import rag_agent.agents.no_answer as no_answer_mod
import rag_agent.tools.documents as doc_tools
import rag_agent.tools.web as web_tools
from rag_agent.agents.no_answer import build_no_answer
from rag_agent.citations import CitationRegistry
from rag_agent.config import settings
from rag_agent.ingest import ensure_seeded
from rag_agent.llms import override_models
from rag_agent.streaming import friendly_error


@pytest.fixture(scope="module", autouse=True)
def _seed():
    ensure_seeded()


def no_web_results(monkeypatch):
    monkeypatch.setattr(web_tools, "_search_duckduckgo", lambda q: [])


def web_down(monkeypatch):
    def boom(q):
        raise ConnectionError("ddg rate limited")

    monkeypatch.setattr(web_tools, "_search_duckduckgo", boom)


def researchers_search(doc_query: str, web_query: str) -> dict:
    return {
        DOCS: [
            tool_msg("", call("search_documents", "d1", query=doc_query)),
            AIMessage("Nothing relevant."),
        ],
        WEB: [
            tool_msg("", call("web_search", "w1", query=web_query)),
            AIMessage("Nothing useful."),
        ],
    }


# --- nothing found anywhere ---------------------------------------------------------


async def test_nothing_found_anywhere_gives_structured_honest_response(monkeypatch):
    no_web_results(monkeypatch)
    q = "What is the capital of Atlantis?"
    model = ScriptedChatModel(
        scripts={PLANNER: [plan("both", q, [q], [q])], **researchers_search(q, q)}
    )
    events = await run_turn(model, q)

    na = of(events, "no_answer")
    assert len(na) == 1
    na = na[0]
    assert na["reason"] == "not_found"
    where = {s["where"]: s for s in na["searched"]}
    assert where["Your documents"]["status"] == "no_results"
    assert where["Your documents"]["queries"] == [q]
    assert where["The web"]["status"] == "no_results"
    assert na["near_misses"], "closest passages from the documents are offered"
    assert any("upload" in s.lower() for s in na["suggestions"])
    assert "won't guess" in na["explanation"]
    assert "Your documents cover:" in na["explanation"]

    # no model was asked to write an answer: nothing could be hallucinated
    assert not any(k in (WRITER, JUDGE) for k, _ in model.calls)
    assert not of(events, "token")
    assert of(events, "verification")[0]["confidence"] == "none"
    assert not of(events, "error")


async def test_web_down_is_reported_as_unavailable_not_as_no_results(monkeypatch):
    web_down(monkeypatch)
    q = "Who won the 2030 World Cup?"
    model = ScriptedChatModel(
        scripts={PLANNER: [plan("both", q, [q], [q])], **researchers_search(q, q)}
    )
    events = await run_turn(model, q)
    na = of(events, "no_answer")[0]
    assert na["reason"] == "web_unavailable"
    web = next(s for s in na["searched"] if s["where"] == "The web")
    assert web["status"] == "unavailable"
    assert any("try again" in s.lower() for s in na["suggestions"])


def test_empty_knowledge_base_says_so(monkeypatch):
    monkeypatch.setattr(no_answer_mod, "source_stats", lambda: [])
    registry = CitationRegistry()
    registry.log_attempt("web_search", "x", "no_results")
    na = build_no_answer("x", registry, findings=[])
    assert na["reason"] == "no_documents"
    assert na["searched"][0]["status"] == "empty"
    assert na["near_misses"] == []
    assert any("Upload documents" in s for s in na["suggestions"])
    assert na["markdown"].startswith("**I couldn't find an answer")


# --- evidence found, but it doesn't (fully) answer -----------------------------------


async def test_evidence_that_doesnt_answer_becomes_no_answer(monkeypatch):
    no_web_results(monkeypatch)
    q = "How do I configure Kubernetes pod autoscaling?"
    model = ScriptedChatModel(
        scripts={
            PLANNER: [plan("documents", q, ["configure settings change"])],
            DOCS: [
                tool_msg("", call("search_documents", "d1", query="configuration change")),
                AIMessage("Found Aurora config docs only."),
            ],
            WRITER: [AIMessage("The documents describe Aurora config, not Kubernetes.")],
            JUDGE: [{**verdict(grounded=True, score=1.0), "coverage": "none"}],
        }
    )
    events = await run_turn(model, q)
    kinds = [k for k, _ in events]
    # the draft streamed, then the verifier replaced it with the honest response
    assert kinds.index("token") < kinds.index("no_answer")
    na = of(events, "no_answer")[0]
    assert na["near_misses"] and na["near_misses"][0]["title"].startswith("aurora_config")
    v = of(events, "verification")[-1]
    assert v["confidence"] == "none" and v["checked"] is True
    assert sum(1 for k, _ in model.calls if k == WRITER) == 1  # no pointless revision


async def test_partial_answer_triggers_web_gap_filling():
    """Docs answer half the question -> verifier sends the Web Researcher after
    the missing half -> the writer revises with the new evidence."""
    q = "What is mTLS and how does Aurora use it?"
    model = ScriptedChatModel(
        scripts={
            PLANNER: [plan("documents", q, ["Aurora mTLS authentication"])],
            DOCS: [
                tool_msg("", call("search_documents", "d1", query="Aurora mTLS")),
                AIMessage("Aurora requires mTLS + OIDC."),
            ],
            WEB: [
                tool_msg("", call("web_search", "w1", query="what is mTLS")),
                AIMessage("mTLS = mutual TLS."),
            ],
            WRITER: [
                AIMessage("Aurora requires mTLS plus OIDC [1]. The documents don't define mTLS."),
                AIMessage(
                    "mTLS means both sides present certificates [5]; Aurora requires it [1]."
                ),
            ],
            JUDGE: [
                {**verdict(), "coverage": "partial", "missing_aspects": ["definition of mTLS"]},
                verdict(),
            ],
        }
    )
    events = await run_turn(model, q)
    esc = of(events, "escalation")
    assert esc and esc[0]["from"] == "verifier" and "definition of mTLS" in esc[0]["reason"]
    web_task = next(msgs for key, msgs in model.calls if key == WEB)[-1].text
    assert "ONLY about: definition of mTLS" in web_task
    writer_calls = [msgs for key, msgs in model.calls if key == WRITER]
    assert (
        len(writer_calls) == 2
        and "could not answer: definition of mTLS" in writer_calls[1][-1].text
    )
    v = of(events, "verification")[-1]
    assert v["coverage"] == "full" and v["revisions"] == 1


async def test_partial_answer_reports_what_is_missing(monkeypatch):
    monkeypatch.setattr(settings, "web_fallback", False)  # no gap filling possible
    q = "What is the max event size and the office address?"
    model = ScriptedChatModel(
        scripts={
            PLANNER: [plan("documents", q, ["max event size"])],
            DOCS: [
                tool_msg("", call("search_documents", "d1", query="max event size")),
                AIMessage("Found the size."),
            ],
            WRITER: [AIMessage("The max event size is 256 KB [1]. The address isn't covered.")],
            JUDGE: [{**verdict(), "coverage": "partial", "missing_aspects": ["office address"]}],
        }
    )
    events = await run_turn(model, q)
    v = of(events, "verification")[0]
    assert v["coverage"] == "partial" and v["missing_aspects"] == ["office address"]
    assert v["confidence"] == "high"  # what it did say is fully supported
    assert not of(events, "no_answer")


# --- model failures ------------------------------------------------------------------


def _doc_turn_scripts(q: str) -> dict:
    return {
        PLANNER: [plan("documents", q, ["AUR-2005"])],
        DOCS: [
            tool_msg("", call("search_documents", "d1", query="AUR-2005")),
            AIMessage("Found it."),
        ],
        JUDGE: [verdict()],
    }


async def test_writer_falls_back_to_other_model_on_rate_limit():
    class RateLimitError(Exception):
        pass

    q = "What is AUR-2005?"
    primary = ScriptedChatModel(
        scripts={**_doc_turn_scripts(q), WRITER: [RateLimitError("429 Too Many Requests")]}
    )
    backup = ScriptedChatModel(scripts={WRITER: [AIMessage("CROSS_TENANT_JOIN [1].")]})
    with override_models({"writer_fallback": backup}):
        events = await run_turn(primary, q)
    assert answer(events) == "CROSS_TENANT_JOIN [1]."
    assert of(events, "verification")[0]["confidence"] == "high"
    assert not of(events, "error")


async def test_all_writer_models_down_returns_extractive_answer():
    q = "What is AUR-2005?"
    model = ScriptedChatModel(
        scripts={**_doc_turn_scripts(q), WRITER: [ConnectionError("groq down")]}
    )
    events = await run_turn(model, q)
    replaced = of(events, "replace_answer")
    assert replaced and "most relevant passages" in replaced[0]["text"]
    assert "[1]" in replaced[0]["text"] and "AUR-2005" in replaced[0]["text"]
    assert of(events, "degraded")
    v = of(events, "verification")[0]
    assert v["confidence"] == "extractive" and v["checked"] is False
    assert not any(k == JUDGE for k, _ in model.calls)  # verbatim quotes need no judge
    assert not of(events, "error")


async def test_tool_crash_is_returned_to_the_agent_not_raised(monkeypatch):
    no_web_results(monkeypatch)

    def broken_search(*a, **k):
        raise RuntimeError("chroma exploded: /secret/internal/path")

    monkeypatch.setattr(doc_tools, "hybrid_search", broken_search)
    q = "What is AUR-2005?"
    seen_by_model: list[str] = []

    def notes(messages):
        seen_by_model.append(messages[-1].text)
        return AIMessage("The search tool failed.")

    model = ScriptedChatModel(
        scripts={
            PLANNER: [plan("documents", q, ["AUR-2005"])],
            DOCS: [tool_msg("", call("search_documents", "d1", query="AUR-2005")), notes],
            WEB: [AIMessage("Nothing.")],
        }
    )
    events = await run_turn(model, q)
    assert seen_by_model and seen_by_model[0].startswith("TOOL_ERROR: search_documents failed")
    assert "/secret/internal/path" not in seen_by_model[0]  # internals never leak
    tool_end = next(d for d in of(events, "tool_end") if d["tool"] == "search_documents")
    assert tool_end["ok"] is False
    assert of(events, "no_answer") and not of(events, "error")


async def test_turn_timeout_is_a_retryable_error(monkeypatch):
    monkeypatch.setattr(settings, "turn_timeout_s", 0.5)

    def slow(messages):
        import time

        time.sleep(2)
        return plan("direct", "What can you do?")

    model = ScriptedChatModel(scripts={PLANNER: [slow]})
    events = await run_turn(model, "What can you do?")  # not small talk: the planner runs
    err = of(events, "error")
    assert err and err[0]["retryable"] is True and "longer than" in err[0]["message"]
    assert events[-1][0] == "end"
    await asyncio.sleep(2)  # let the abandoned worker thread finish before teardown


@pytest.mark.parametrize(
    ("exc", "retryable"),
    [
        (type("RateLimitError", (Exception,), {})("429"), True),
        (type("APITimeoutError", (Exception,), {})("slow"), True),
        (type("AuthenticationError", (Exception,), {})("bad key"), False),
        (type("GraphRecursionError", (Exception,), {})("loop"), False),
    ],
)
def test_errors_say_whether_retrying_helps(exc, retryable):
    message, can_retry = friendly_error(exc)
    assert can_retry is retryable and message


# --- policy: answering without evidence ------------------------------------------------


async def test_opt_in_general_knowledge_answer_is_clearly_labelled(monkeypatch):
    no_web_results(monkeypatch)
    monkeypatch.setattr(settings, "answer_without_evidence", True)
    q = "What is the capital of France?"
    model = ScriptedChatModel(
        scripts={
            PLANNER: [plan("both", q, [q], [q])],
            **researchers_search(q, q),
            "general knowledge": [AIMessage("⚠️ Not found ... Paris.")],
        }
    )
    events = await run_turn(model, q)
    assert not of(events, "no_answer")
    assert answer(events).startswith("⚠️ Not found")
    v = of(events, "verification")[0]
    assert v["confidence"] == "unverified" and v["model_knowledge"] is True


async def test_extractive_mode_never_quotes_weak_matches(monkeypatch):
    """Writers down + only loosely-related evidence -> honest no-answer, not quotes."""
    no_web_results(monkeypatch)
    monkeypatch.setattr(settings, "min_rerank_score", 1.01)  # nothing counts as confirmed
    q = "What is the capital of Atlantis?"
    model = ScriptedChatModel(
        scripts={
            PLANNER: [plan("documents", q, ["query gateway timeout"])],
            DOCS: [
                tool_msg("", call("search_documents", "d1", query="query gateway timeout")),
                AIMessage("Found something loosely related."),
            ],
            WRITER: [ConnectionError("groq down")],
        }
    )
    events = await run_turn(model, q)
    assert of(events, "degraded") and not of(events, "replace_answer")
    assert of(events, "no_answer") and not of(events, "error")


def test_roles_cover_every_agent():
    assert set(ROLES) == {"planner", "worker", "writer", "verifier", "summarizer"}


# --- deterministic backstops on real-model quirks (found in live Groq runs) ------------


@pytest.mark.parametrize(
    ("text", "cited", "expected"),
    [
        (
            "No publicly available source in the evidence mentions the team's pizza [1].",
            True,
            "none",
        ),
        ("The provided evidence does not contain any information about pizza.", False, "none"),
        ("Aurora requires mTLS [1]. The evidence doesn't define mTLS.", True, "partial"),
        ("AUR-2005 is CROSS_TENANT_JOIN [1]. No retry is needed.", True, "full"),
        ("Retention is 14 days [2]; it is not configurable beyond 90 days.", True, "full"),
    ],
)
def test_coverage_backstop_catches_answers_that_admit_missing_info(text, cited, expected):
    from rag_agent.agents.verifier import coverage_backstop

    assert coverage_backstop(text, "full", cited) == expected


def test_normalize_answer_fixes_gpt_oss_citations_and_unicode():
    from rag_agent.agents.verifier import normalize_answer

    raw = "AUR\u20112005 means X\u202fY \u30103\u3011\u30104\u2020L10-L20\u3011"
    assert normalize_answer(raw) == "AUR-2005 means X Y [3][4]"


def test_placeholder_keys_from_env_example_count_as_unset():
    from rag_agent.config import Settings

    s = Settings(
        groq_api_key="gsk_real",
        tavily_api_key="tvly-...",
        langsmith_api_key="lsv2_pt_...",
        langsmith_tracing=True,
    )
    assert s.tavily_api_key is None and s.web_backend == "duckduckgo"
    assert s.langsmith_tracing is False  # no tracing without a real key


# --- planner routing backstop (found by the RAGAS behaviour eval) ---------------------------


async def test_factual_question_routed_direct_is_researched_instead(monkeypatch):
    """The small planner once routed "Which pizza toppings does the team prefer?" to
    `direct`, answering from memory without searching. Factual questions must be
    researched, and with no evidence the user gets the honest no-answer."""
    no_web_results(monkeypatch)
    q = "Which pizza toppings does the Aurora on-call team prefer?"
    model = ScriptedChatModel(
        scripts={PLANNER: [plan("direct", q)], **researchers_search("pizza", "pizza")}
    )
    events = await run_turn(model, q)
    assert of(events, "plan")[0]["route"] == "both"
    assert of(events, "no_answer") and not any(k == "needs no research" for k, _ in model.calls)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("hi!", True),
        ("thanks a lot", True),
        ("What can you do?", True),
        ("Which pizza toppings does the Aurora on-call team prefer?", False),
        ("hi, what does AUR-2005 mean and how do I fix it in production?", False),
    ],
)
def test_is_conversational(text, expected):
    from rag_agent.agents.planner import is_conversational

    assert is_conversational(text) is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("thank you", True),
        ("Thanks a lot!", True),
        ("ok great, thank you so much 🙏", True),
        ("hi", True),
        ("thanks, what does AUR-2005 mean?", False),  # a real question after the thanks
        ("What can you do?", False),  # about the assistant: the planner sees the library
        ("ok", True),
        ("👍", False),  # no words: let the planner decide
    ],
)
def test_is_small_talk(text, expected):
    from rag_agent.agents.planner import is_small_talk

    assert is_small_talk(text) is expected


async def test_small_talk_never_triggers_research(monkeypatch):
    """The planner model once routed "thank you" to the web and cited wikiHow's
    "How to Respond to Thank You". Pure pleasantries now skip the planner model."""
    model = ScriptedChatModel(
        scripts={
            PLANNER: [plan("web", "how to respond to thank you", [], ["respond to thank you"])],
            "needs no research": [AIMessage("You're welcome!")],
        }
    )
    events = await run_turn(model, "thank you")
    assert of(events, "plan")[0]["route"] == "direct"
    assert answer(events) == "You're welcome!"
    assert not any(k == PLANNER for k, _ in model.calls)  # no LLM call to route it
    assert not of(events, "tool_start") and not of(events, "sources")
