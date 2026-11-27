"""Web Researcher tools — public information outside the private documents.

| Tool             | What it does                                                 |
|------------------|--------------------------------------------------------------|
| web_search       | Hedged metasearch + re-rank + read top pages (websearch.py)  |
| fetch_webpage    | Read the full text of one URL (SSRF-guarded)                 |
| wikipedia_lookup | Encyclopedic summary of a concept, entity or term            |
"""

from __future__ import annotations

from urllib.parse import quote

import httpx
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from rag_agent import websearch
from rag_agent.citations import registry_from_config
from rag_agent.privacy import CONFIG_KEY as PRIVATE_KEY
from rag_agent.privacy import scrub_query
from rag_agent.tools.common import (
    emit,
    emit_sources,
    format_evidence,
    logs_attempt,
    register_evidence,
)
from rag_agent.websearch import (  # re-exported: the SSRF guard lives with the fetcher
    USER_AGENT,
    BlockedURLError,
    check_public_url,
    fetch_url_text,
    html_to_text,
)

__all__ = ["BlockedURLError", "check_public_url", "html_to_text", "WEB_TOOLS"]

_TIMEOUT = httpx.Timeout(10.0, connect=5.0)
_MAX_PAGE_CHARS = 4000


# --- web_search -----------------------------------------------------------------


def _keep_private(query: str, config: RunnableConfig) -> tuple[str, str]:
    """Strip internal names from an outgoing web query (see rag_agent/privacy.py).

    Returns the cleaned query and a note telling the agent what was removed
    ("" when nothing changed). The names arrive through the run config, set by
    the Web Researcher only while the user's documents are in play.
    """
    names = ((config or {}).get("configurable") or {}).get(PRIVATE_KEY) or []
    if not names:
        return query, ""
    cleaned, removed = scrub_query(query, names)
    if not removed:
        return query, ""
    emit(
        {
            "kind": "guardrail",
            "stage": "retrieval",
            "action": "private",
            "detail": f"Kept {', '.join(n.title() for n in removed)} out of the web search",
            "original": query,  # lets the UI show the query that was actually sent
            "sent": cleaned,
        }
    )
    note = (
        f"(Internal name {', '.join(removed)} was removed from this public search; "
        f"searched for: {cleaned or 'nothing'}.)\n"
    )
    return cleaned, note


def _register_web(config: RunnableConfig, title: str, url: str | None, text: str, **extra):
    """Register a web result through the retrieval guardrail (sanitized text)."""
    return register_evidence(
        registry_from_config(config),
        key=f"web:{websearch.canonical_url(url) if url else title}",
        source={"kind": "web", "title": title, "url": url, **extra},
        text=text,
    )


def _clean(config: RunnableConfig, entry: dict) -> str:
    """The guardrail-sanitized text stored for ``entry``."""
    return registry_from_config(config).text(entry["index"])


@tool
@logs_attempt("query")
def web_search(query: str, config: RunnableConfig) -> str:
    """Search the public web. Runs several search engines at once, re-ranks the
    results by relevance and reads the best pages in full, so one call is
    usually enough. Returns passages labelled [n].
    """
    query, note = _keep_private(query, config)
    if not query:
        return note + "NO_WEB_RESULTS: nothing left to search once internal names were removed."
    try:
        report = websearch.search(query)
    except websearch.WebSearchUnavailable as exc:
        return f"WEB_SEARCH_UNAVAILABLE: {exc}"
    except Exception as exc:  # noqa: BLE001 — never let search crash the agent
        return f"WEB_SEARCH_UNAVAILABLE: {type(exc).__name__}: {exc}"
    if not report.hits:
        why = " (results were found but none were relevant)" if report.dropped else ""
        return f"NO_WEB_RESULTS: the web search returned nothing useful for this query{why}."

    known = len(registry_from_config(config))  # pages an earlier search already found
    entries, bodies = [], []
    for h in report.hits:
        entry = _register_web(
            config,
            h.title,
            h.url,
            h.text[:1800],
            score=h.score,
            read=h.read,
            date=h.date,
            engines=h.engines,
        )
        entries.append(entry)
        bodies.append(_clean(config, entry))
    emit_sources("web_search", [e for e in entries if e["index"] > known])
    emit(
        {
            "kind": "web_search",
            "query": query,
            "engines": report.engines_ok,
            "results": len(report.hits),
            "pages_read": report.pages_read,
            "fresh": report.fresh,
            "cached": report.cached,
        }
    )
    return note + format_evidence(entries, bodies)


# --- fetch_webpage --------------------------------------------------------------


@tool
@logs_attempt("url")
def fetch_webpage(url: str, config: RunnableConfig) -> str:
    """Read the main text of one public web page (e.g. a URL from web_search)."""
    try:
        text, title = fetch_url_text(url)
    except BlockedURLError as exc:
        return f"FETCH_BLOCKED: {exc}"
    except httpx.HTTPError as exc:
        return f"FETCH_FAILED: {type(exc).__name__}: {exc}"
    if not text.strip():
        return "FETCH_FAILED: the page has no readable text."
    body = text[:_MAX_PAGE_CHARS]
    entry = _register_web(config, title or url, url, body, read=True)
    emit_sources("fetch_webpage", [entry])
    more = " (truncated)" if len(text) > _MAX_PAGE_CHARS else ""
    return format_evidence([entry], [_clean(config, entry) + more])


# --- wikipedia_lookup -------------------------------------------------------------

_WIKI_API = "https://en.wikipedia.org/w/api.php"
_WIKI_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/"


@tool
@logs_attempt("topic")
def wikipedia_lookup(topic: str, config: RunnableConfig) -> str:
    """Get the Wikipedia summary for a well-known concept, technology, or entity.

    Best for definitions and background ("what is ClickHouse?", "what is mTLS?").
    """
    topic, note = _keep_private(topic, config)
    if not topic:
        return (
            note + "NO_WIKIPEDIA_ARTICLE: nothing left to look up once internal names were removed."
        )
    try:
        with httpx.Client(timeout=_TIMEOUT, headers={"User-Agent": USER_AGENT}) as client:
            found = client.get(
                _WIKI_API,
                params={
                    "action": "opensearch",
                    "search": topic,
                    "limit": 1,
                    "namespace": 0,
                    "format": "json",
                },
            )
            found.raise_for_status()
            titles = found.json()[1]
            if not titles:
                return f"NO_WIKIPEDIA_ARTICLE: nothing found for '{topic}'."
            page = client.get(_WIKI_SUMMARY + quote(titles[0].replace(" ", "_")))
            page.raise_for_status()
            data = page.json()
    except (httpx.HTTPError, ValueError, IndexError) as exc:
        return f"WIKIPEDIA_UNAVAILABLE: {type(exc).__name__}: {exc}"

    extract = (data.get("extract") or "").strip()
    if not extract or data.get("type") == "disambiguation":
        return f"NO_WIKIPEDIA_ARTICLE: '{topic}' is ambiguous or has no summary."
    url = (data.get("content_urls") or {}).get("desktop", {}).get("page")
    entry = _register_web(
        config, f"Wikipedia: {data.get('title', titles[0])}", url, extract, read=True
    )
    emit_sources("wikipedia_lookup", [entry])
    return note + format_evidence([entry], [_clean(config, entry)])


WEB_TOOLS = [web_search, fetch_webpage, wikipedia_lookup]
