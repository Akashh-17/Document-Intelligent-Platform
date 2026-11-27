"""The 12 agent tools, called directly (no LLM except scripted fakes)."""

from __future__ import annotations

import socket

import pytest
from fakes import SUMMARIZER, ScriptedChatModel
from langchain_core.messages import AIMessage

from rag_agent.citations import CONFIG_KEY, CitationRegistry
from rag_agent.ingest import ensure_seeded
from rag_agent.llms import override_models
from rag_agent.tools import (
    ALL_TOOLS,
    calculator,
    compare_documents,
    fetch_webpage,
    get_adjacent_chunks,
    list_documents,
    search_documents,
    search_within_document,
    summarize_document,
    verify_citations,
)
from rag_agent.tools.common import resolve_source
from rag_agent.tools.documents import safe_eval
from rag_agent.tools.web import BlockedURLError, check_public_url, html_to_text


@pytest.fixture(scope="module", autouse=True)
def _seed():
    ensure_seeded()


@pytest.fixture
def config():
    return {"configurable": {CONFIG_KEY: CitationRegistry()}}


def test_there_are_twelve_uniquely_named_tools():
    names = [t.name for t in ALL_TOOLS]
    assert len(names) == 13 and len(set(names)) == 13


# --- calculator ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("730 / 14", "52.142857142857146"),
        ("256 * 1024", "262144"),
        ("2^10", "1024"),
        ("round(730 / 365)", "2"),
        ("max(14, 90) - min(14, 90)", "76"),
    ],
)
def test_calculator_computes(expr, expected):
    assert calculator.invoke({"expression": expr}).endswith(f"= {expected}")


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('echo hi')",
        "open('x')",
        "(1).__class__",
        "9 ** 99999",
        "1/0",
        "a + 1",
    ],
)
def test_calculator_refuses_anything_but_arithmetic(expr):
    assert calculator.invoke({"expression": expr}).startswith("CALCULATOR_ERROR")


def test_safe_eval_never_uses_python_eval():
    with pytest.raises(ValueError):
        safe_eval("[x for x in ().__class__.__bases__]")


# --- verify_citations ------------------------------------------------------------


def test_verify_citations_flags_invented_numbers_and_uncited_claims():
    answer = (
        "The max event size is 256 KB [1]. Retention is 14 days [2, 7].\n"
        "Queries time out after thirty seconds by default."
    )
    r = verify_citations.invoke({"answer": answer, "available": [1, 2, 3]})
    assert r["cited"] == [1, 2, 7]
    assert r["invalid"] == [7] and r["ok"] is False
    assert r["uncited_sentences"] == ["Queries time out after thirty seconds by default."]


def test_verify_citations_ok():
    r = verify_citations.invoke({"answer": "It is 256 KB [1].", "available": [1]})
    assert r["ok"] and r["has_citations"] and not r["uncited_sentences"]


# --- web safety ------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:8000/api/health",
        "http://127.0.0.1/",
        "http://10.0.0.5/admin",
        "http://169.254.169.254/latest/meta-data/",
        "file:///etc/passwd",
        "ftp://example.com/x",
        "http://[::1]/",
    ],
)
def test_ssrf_guard_blocks_internal_targets(url):
    with pytest.raises(BlockedURLError):
        check_public_url(url)


def test_ssrf_guard_allows_public_host(monkeypatch):
    monkeypatch.setattr(
        socket, "getaddrinfo", lambda *a, **k: [(None, None, None, "", ("93.184.215.14", 443))]
    )
    check_public_url("https://example.com/page")  # no exception


def test_fetch_webpage_reports_blocked_url(config):
    out = fetch_webpage.invoke({"url": "http://localhost:8000/api/health"}, config)
    assert out.startswith("FETCH_BLOCKED")
    assert len(config["configurable"][CONFIG_KEY]) == 0  # nothing registered


def test_html_to_text_keeps_content_drops_chrome():
    html = (
        "<html><head><title>T</title><script>evil()</script></head><body>"
        "<nav>menu</nav><main><h1>Hello</h1><p>Real content.</p></main><footer>f</footer>"
        "</body></html>"
    )
    text, title = html_to_text(html)
    assert title == "T" and "Real content." in text
    assert "evil" not in text and "menu" not in text


# --- document tools --------------------------------------------------------------


def test_list_documents_shows_inventory():
    out = list_documents.invoke({})
    assert "aurora_error_codes.md" in out and "Error Code Reference" in out


def test_same_chunk_keeps_its_citation_number_across_tools(config):
    first = search_documents.invoke({"query": "AUR-2005"}, config)
    again = search_within_document.invoke(
        {"document": "aurora_error_codes.md", "query": "AUR-2005"}, config
    )
    assert first.split("\n", 1)[0] == again.split("\n", 1)[0]  # same "[n] title" label


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("aurora_error_codes.md", "aurora_error_codes.md"),
        ("aurora_error_codes", "aurora_error_codes.md"),
        ("the oncall runbook", None),
        ("aurora_oncal_runbook.md", "aurora_oncall_runbook.md"),
    ],
)
def test_resolve_source_is_lenient(name, expected):
    got = resolve_source(name)
    if expected:
        assert got == expected
    else:
        assert got in (None, "aurora_oncall_runbook.md")


def test_search_within_document_stays_inside_it(config):
    out = search_within_document.invoke(
        {"document": "aurora_config_reference.md", "query": "timeout"}, config
    )
    registry = config["configurable"][CONFIG_KEY]
    assert registry.sources and all(
        s["title"].startswith("aurora_config_reference.md") for s in registry.sources
    )
    assert "query.timeout_ms" in out


def test_unknown_document_lists_alternatives(config):
    out = search_within_document.invoke({"document": "budget.xlsx", "query": "x"}, config)
    assert out.startswith("UNKNOWN_DOCUMENT") and "aurora_faq.md" in out


def test_get_adjacent_chunks_reads_neighbours(config):
    search_documents.invoke({"query": "AUR-2005 CROSS_TENANT_JOIN"}, config)
    out = get_adjacent_chunks.invoke({"citation": 1}, config)
    registry = config["configurable"][CONFIG_KEY]
    assert len(registry) > 1 and "aurora_error_codes.md" in out
    assert get_adjacent_chunks.invoke({"citation": 99}, config).startswith("INVALID_CITATION")


def test_compare_documents_returns_both_sides(config):
    out = compare_documents.invoke(
        {
            "document_a": "aurora_faq.md",
            "document_b": "aurora_config_reference.md",
            "aspect": "raw event retention",
        },
        config,
    )
    assert "=== aurora_faq.md ===" in out and "=== aurora_config_reference.md ===" in out


def test_summarize_document_uses_summarizer_and_registers_citation(config):
    fake = ScriptedChatModel(scripts={SUMMARIZER: [AIMessage("Aurora on-call duties and alerts.")]})
    with override_models({"summarizer": fake}):
        out = summarize_document.invoke({"document": "aurora_oncall_runbook.md"}, config)
    assert out.startswith("[1] Summary of aurora_oncall_runbook.md")
    assert config["configurable"][CONFIG_KEY].text(1) == "Aurora on-call duties and alerts."
    # the summarizer saw the document text
    assert "AuroraIngestLagHigh" in fake.calls[0][1][-1].text
