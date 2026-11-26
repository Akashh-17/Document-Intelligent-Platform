"""Web Researcher tools — public information outside the private documents.

| Tool             | What it does                                                 |
|------------------|--------------------------------------------------------------|
| web_search       | Search the web (Tavily if keyed, else keyless DuckDuckGo)    |
| fetch_webpage    | Read the full text of one URL (SSRF-guarded)                 |
| wikipedia_lookup | Encyclopedic summary of a concept, entity or term            |
"""

from __future__ import annotations

import ipaddress
import re
import socket
from urllib.parse import quote, urljoin, urlparse

import httpx
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from rag_agent.citations import registry_from_config
from rag_agent.config import settings
from rag_agent.privacy import CONFIG_KEY as PRIVATE_KEY
from rag_agent.privacy import scrub_query
from rag_agent.tools.common import (
    emit,
    emit_sources,
    format_evidence,
    logs_attempt,
    register_evidence,
)

USER_AGENT = "DocumentIntelligencePlatform/0.2 (educational RAG project)"
_TIMEOUT = httpx.Timeout(10.0, connect=5.0)
_MAX_BYTES = 2 * 1024 * 1024
_MAX_PAGE_CHARS = 4000


# --- web_search -----------------------------------------------------------------


def _search_tavily(query: str) -> list[dict]:
    from langchain_tavily import TavilySearch

    result = TavilySearch(max_results=5).invoke({"query": query})
    results = result.get("results", []) if isinstance(result, dict) else []
    return [
        {
            "title": r.get("title") or r.get("url") or "result",
            "url": r.get("url"),
            "content": r.get("content") or "",
            "score": r.get("score"),
        }
        for r in results
    ]


def _search_duckduckgo(query: str) -> list[dict]:
    from ddgs import DDGS

    results = DDGS().text(query, max_results=5) or []
    return [
        {
            "title": r.get("title") or r.get("href") or "result",
            "url": r.get("href"),
            "content": r.get("body") or "",
            "score": None,
        }
        for r in results
    ]


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


def _register_web(config: RunnableConfig, title: str, url: str | None, text: str, score=None):
    """Register a web result through the retrieval guardrail (sanitized text)."""
    return register_evidence(
        registry_from_config(config),
        key=f"web:{url or title}",
        source={"kind": "web", "title": title, "url": url, "score": score},
        text=text,
    )


def _clean(config: RunnableConfig, entry: dict) -> str:
    """The guardrail-sanitized text stored for ``entry``."""
    return registry_from_config(config).text(entry["index"])


@tool
@logs_attempt("query")
def web_search(query: str, config: RunnableConfig) -> str:
    """Search the public web for current events or facts not in the documents.

    Returns short result snippets labelled [n]; use fetch_webpage to read one fully.
    """
    query, note = _keep_private(query, config)
    if not query:
        return note + "NO_WEB_RESULTS: nothing left to search once internal names were removed."
    try:
        results = _search_tavily(query) if settings.tavily_api_key else _search_duckduckgo(query)
    except Exception as exc:  # noqa: BLE001 — rate limits, network errors: degrade gracefully
        return f"WEB_SEARCH_UNAVAILABLE: {type(exc).__name__}: {exc}"
    if not results:
        return "NO_WEB_RESULTS: the web search returned nothing for this query."

    entries, bodies = [], []
    for r in results:
        body = r["content"][:1200]
        entry = _register_web(config, r["title"], r["url"], body, r["score"])
        entries.append(entry)
        bodies.append(_clean(config, entry))
    emit_sources("web_search", entries)
    return note + format_evidence(entries, bodies)


# --- fetch_webpage --------------------------------------------------------------


class BlockedURLError(ValueError):
    """The URL is not allowed (bad scheme, or resolves to a private address)."""


def check_public_url(url: str) -> None:
    """Reject URLs that could reach internal infrastructure (SSRF protection).

    The URL comes from an LLM, which may have been steered by text on a web page
    or in a document (prompt injection). Without this check, the agent could be
    tricked into fetching http://localhost:8000/... or cloud metadata at
    169.254.169.254 and leaking the response into its answer.

    Known limit: the HTTP client resolves the host again, so a DNS-rebinding
    attacker could swap addresses between check and fetch. Fully closing that
    needs pinning the connection to the checked IP (or an egress proxy).
    """
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise BlockedURLError(f"only http(s) URLs are allowed, got '{parsed.scheme}'")
    host = parsed.hostname
    if not host:
        raise BlockedURLError("URL has no host")
    try:
        infos = socket.getaddrinfo(host, parsed.port or 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise BlockedURLError(f"cannot resolve host '{host}'") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global or ip.is_multicast:
            raise BlockedURLError(f"'{host}' resolves to a non-public address ({ip})")


def html_to_text(html: str) -> tuple[str, str]:
    """Main readable text and title of an HTML page."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    title = (soup.title.string or "").strip() if soup.title and soup.title.string else ""
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()
    root = soup.find("main") or soup.find("article") or soup.body or soup
    text = re.sub(r"\n\s*\n+", "\n\n", root.get_text("\n"))
    text = re.sub(r"[ \t]+", " ", text).strip()
    return text, title


def fetch_url_text(url: str) -> tuple[str, str]:
    """Fetch a public URL (re-checking every redirect hop) and return (text, title)."""
    with httpx.Client(timeout=_TIMEOUT, headers={"User-Agent": USER_AGENT}) as client:
        for _ in range(4):  # at most 3 redirects
            check_public_url(url)
            with client.stream("GET", url, follow_redirects=False) as resp:
                if resp.is_redirect:
                    url = urljoin(url, resp.headers.get("location", ""))
                    continue
                resp.raise_for_status()
                ctype = resp.headers.get("content-type", "")
                if not ctype.startswith(("text/html", "text/plain", "application/xhtml")):
                    raise BlockedURLError(f"unsupported content type '{ctype}'")
                body = b""
                for chunk in resp.iter_bytes():
                    body += chunk
                    if len(body) > _MAX_BYTES:
                        break
                raw = body.decode(resp.encoding or "utf-8", errors="replace")
                return html_to_text(raw) if "html" in ctype else (raw, "")
        raise BlockedURLError("too many redirects")


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
    entry = _register_web(config, title or url, url, body)
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
    entry = _register_web(config, f"Wikipedia: {data.get('title', titles[0])}", url, extract)
    emit_sources("wikipedia_lookup", [entry])
    return note + format_evidence([entry], [_clean(config, entry)])


WEB_TOOLS = [web_search, fetch_webpage, wikipedia_lookup]
