"""The web search pipeline (rag_agent/websearch.py), offline: engines and pages are faked."""

from __future__ import annotations

import time

import pytest

from rag_agent import websearch
from rag_agent.config import settings
from rag_agent.tools.web import web_search


@pytest.fixture(autouse=True)
def _fresh_state():
    websearch._search_cache.clear()
    websearch._page_cache.clear()
    websearch._breaker._failures.clear()


def row(title, url, snippet):
    return {"title": title, "url": url, "snippet": snippet}


# --- small pieces ---------------------------------------------------------------------------


def test_canonical_url_merges_the_same_page():
    a = websearch.canonical_url("https://www.example.com/docs/?utm_source=x&id=3#intro")
    b = websearch.canonical_url("http://example.com/docs?id=3")
    assert a == b == "https://example.com/docs?id=3"


@pytest.mark.parametrize(
    ("query", "fresh"),
    [
        ("latest python release", True),
        ("bitcoin price today", True),
        ("world cup 2026 final score", True),
        ("what is mutual TLS", False),
    ],
)
def test_time_sensitive_queries_also_search_news(query, fresh):
    assert websearch.is_time_sensitive(query) is fresh


def test_boilerplate_filter_keeps_prose_and_drops_menus():
    prose = (
        "Mutual TLS is a method for mutual authentication in which both the client and the "
        "server present certificates to each other before any data is exchanged."
    )
    menu = "Home Products Pricing Docs Blog Careers Login Signup Tutorials Courses Jobs Events"
    assert websearch.is_prose(prose)
    assert not websearch.is_prose(menu)


def test_fusion_dedupes_caps_domains_and_drops_junk():
    per_engine = {
        "a": [
            row("A1", "https://site.com/1", "one"),
            row("A2", "https://site.com/2", "two"),
            row("A3", "https://site.com/3", "three"),  # third from one domain: capped
            row("Pin", "https://www.pinterest.com/pin/1", "junk"),
        ],
        "b": [
            row("B1", "https://other.org/x", "x"),
            row("A1 again", "https://www.site.com/1/?utm_source=b", "longer snippet for one"),
        ],
    }
    hits = websearch._fuse(per_engine)
    urls = [websearch.canonical_url(h.url) for h in hits]
    assert urls[0] == "https://site.com/1"  # both engines agree -> ranked first by RRF
    assert hits[0].engines == ["a", "b"] and hits[0].snippet == "longer snippet for one"
    assert "https://site.com/3" not in urls
    assert not any("pinterest" in u for u in urls)


# --- robustness: hedging, circuit breaker, cache ---------------------------------------------


def test_hedged_search_does_not_wait_for_a_slow_engine():
    def slow():
        time.sleep(3)
        return [row("slow", "https://slow.com", "late")]

    def fast():
        return [row("fast", "https://fast.com", "on time")]

    def broken():
        raise ConnectionError("blocked")

    start = time.monotonic()
    run = websearch._hedged(
        {"slow": slow, "fast": fast, "broken": broken}, first_grace_s=0.3, deadline_s=5
    )
    assert time.monotonic() - start < 1.5
    assert list(run.results) == ["fast"]
    assert set(run.failed) == {"slow", "broken"}


def test_wikipedia_answering_first_does_not_cut_off_the_web_engines():
    def wiki():
        return [row("Wikipedia: X", "https://en.wikipedia.org/wiki/X", "encyclopedia")]

    def web():
        time.sleep(0.6)  # slower than the grace period, counted from Wikipedia
        return [row("web", "https://web.com", "fresh page")]

    run = websearch._hedged({"wikipedia": wiki, "yahoo": web}, first_grace_s=0.3, deadline_s=3)
    assert set(run.results) == {"wikipedia", "yahoo"}


def test_circuit_breaker_skips_a_failing_engine_then_retries():
    breaker = websearch._CircuitBreaker(threshold=2, cooldown_s=0.2)
    breaker.record("ddg", ok=False)
    assert breaker.available("ddg")
    breaker.record("ddg", ok=False)
    assert not breaker.available("ddg")  # open: skipped
    time.sleep(0.25)
    assert breaker.available("ddg")  # half-open: one retry allowed
    breaker.record("ddg", ok=True)
    assert breaker.available("ddg")


def test_empty_results_and_outage_are_reported_differently(monkeypatch):
    monkeypatch.setattr(
        websearch, "_search_keyless", lambda q, f: websearch.EngineRun(empty=["yahoo"])
    )
    assert websearch.search("zxqv nonsense").hits == []

    websearch._search_cache.clear()
    monkeypatch.setattr(
        websearch, "_search_keyless", lambda q, f: websearch.EngineRun(failed=["yahoo"])
    )
    with pytest.raises(websearch.WebSearchUnavailable):
        websearch.search("anything")


def test_repeat_search_is_served_from_cache(monkeypatch):
    calls = []

    def keyless(query, fresh):
        calls.append(query)
        return websearch.EngineRun(
            results={"stub": [row("mTLS", "https://e.com/mtls", "Mutual TLS authentication")]}
        )

    monkeypatch.setattr(websearch, "_search_keyless", keyless)
    first = websearch.search("mutual TLS")
    second = websearch.search("  Mutual   TLS ")
    assert calls == ["mutual TLS"] and second.cached and not first.cached


# --- relevance: re-ranking and deep reading ---------------------------------------------------


def test_reranker_orders_by_relevance_and_gate_drops_off_topic(monkeypatch):
    rows = [
        row("Pizza dough recipe", "https://food.com/pizza", "How to make pizza dough at home."),
        row(
            "Mutual TLS explained",
            "https://sec.com/mtls",
            "Mutual TLS (mTLS) is when both client and server authenticate with certificates.",
        ),
    ]
    monkeypatch.setattr(
        websearch, "_search_keyless", lambda q, f: websearch.EngineRun(results={"stub": rows})
    )
    report = websearch.search("what is mutual TLS authentication")
    assert report.hits[0].url == "https://sec.com/mtls"
    assert all("pizza" not in h.url for h in report.hits)  # below the relevance gate
    assert report.dropped == 1


def test_deep_read_keeps_the_relevant_passages(monkeypatch):
    page = "\n\n".join(
        [
            "Home Products Pricing Docs Blog Careers Login Signup",  # menu: filtered
            "Our company was founded in 2010 and has offices in many cities around the "
            "world, with a focus on customer happiness and great coffee.",
            "Mutual TLS (mTLS) is a type of authentication in which the two parties in a "
            "connection authenticate each other using the TLS protocol: the server checks "
            "the client's certificate as well as the client checking the server's.",
        ]
    )
    monkeypatch.setattr(settings, "web_deep_read", 1)
    monkeypatch.setattr(websearch, "fetch_url_text", lambda url: (page, "mTLS"))
    monkeypatch.setattr(
        websearch,
        "_search_keyless",
        lambda q, f: websearch.EngineRun(
            results={"stub": [row("mTLS guide", "https://sec.com/mtls", "A guide to mTLS.")]}
        ),
    )
    hit = websearch.search("how does mutual TLS authentication work").hits[0]
    assert hit.read and len(hit.passages) == 1
    assert "authenticate each other" in hit.passages[0]
    assert "authenticate each other" in hit.text and "A guide to mTLS." in hit.text


def test_web_search_tool_reports_outcomes(monkeypatch):
    monkeypatch.setattr(
        websearch, "_search_keyless", lambda q, f: websearch.EngineRun(failed=["yahoo"])
    )
    assert web_search.invoke({"query": "x"}).startswith("WEB_SEARCH_UNAVAILABLE")

    websearch._search_cache.clear()
    monkeypatch.setattr(
        websearch, "_search_keyless", lambda q, f: websearch.EngineRun(empty=["yahoo"])
    )
    assert web_search.invoke({"query": "x"}).startswith("NO_WEB_RESULTS")


# --- the anchor search (end to end) ------------------------------------------------------------


async def test_first_web_search_uses_the_users_words_not_model_memory(monkeypatch):
    """Live bug: asked for "the latest stable python release", the models searched
    "Python 3.12 latest stable release date" (their training-time belief) and the
    answer, though "verified", was a year out of date. The first search now uses
    the user's own wording, plus the current year for time-sensitive questions."""
    from datetime import date

    from fakes import JUDGE, PLANNER, WEB, WRITER, ScriptedChatModel, plan, stub_web, verdict
    from langchain_core.messages import AIMessage
    from test_agents import run_turn

    searched: list[str] = []

    def engines(query):
        searched.append(query)
        return [
            {"title": "Python 3.15", "url": "https://python.org/3.15", "content": "3.15 is out"}
        ]

    stub_web(monkeypatch, engines)
    q = "what's the latest stable python release?"
    model = ScriptedChatModel(
        scripts={
            PLANNER: [plan("web", q, [], ["Python 3.12 latest stable release date"])],
            WEB: [AIMessage("Python 3.15 is the latest [1].")],  # anchor was enough
            WRITER: [AIMessage("Python 3.15 is the latest stable release [1].")],
            JUDGE: [verdict()],
        }
    )
    await run_turn(model, q)
    assert searched == [f"what's the latest stable python release {date.today().year}"]
    task = next(msgs for key, msgs in model.calls if key == WEB)[-1].text
    assert "already ran" in task and "3.15 is out" in task
