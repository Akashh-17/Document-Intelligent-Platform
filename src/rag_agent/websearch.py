"""Web search engine behind the ``web_search`` tool — a small Perplexity-style pipeline.

A raw search API returns ten blue links with 200-character snippets. Answering
from that is how assistants end up citing wikiHow for "thank you". This module
turns a query into a handful of *relevant, read* passages:

    query ─┬─▶ engines in parallel (hedged) ──┐
           └─▶ news vertical (time-sensitive) ─┤
                                               ▼
              fuse rankings (RRF) · canonical-URL dedupe · drop junk domains
                                               ▼
              cross-encoder re-rank of the snippets against the query
                                               ▼
              read the top pages in full (SSRF-guarded), keep their best passages
                                               ▼
                               top results ─▶ evidence [n]

Robustness, because free search endpoints are flaky (measured: any single
engine fails 30-60% of the time from one IP):

- **Hedged requests.** Every engine is queried at once; we take what arrives
  before a short deadline after the first success ("The Tail at Scale",
  Dean & Barroso 2013). One slow or blocked engine never fails the search.
- **Circuit breaker.** An engine that keeps failing is skipped for a while
  instead of burning a thread on every query.
- **Fallback chain.** Tavily (if keyed) → keyless metasearch. Deep reading is
  best effort: a page that times out keeps its snippet.
- **TTL cache.** Repeat queries (follow-ups, revisions, evals) are instant and
  don't spend rate limit.
"""

from __future__ import annotations

import ipaddress
import logging
import re
import socket
import threading
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import httpx

from rag_agent.config import settings
from rag_agent.retrieval.fusion import reciprocal_rank_fusion
from rag_agent.retrieval.rerank import get_reranker

log = logging.getLogger(__name__)

USER_AGENT = "DocumentIntelligencePlatform/0.2 (educational RAG project)"
_TIMEOUT = httpx.Timeout(10.0, connect=5.0)
_MAX_BYTES = 2 * 1024 * 1024
_POOL = ThreadPoolExecutor(max_workers=16, thread_name_prefix="websearch")


class WebSearchUnavailable(RuntimeError):
    """Every search provider failed (network down, all engines rate-limited)."""


@dataclass
class WebHit:
    title: str
    url: str
    snippet: str
    engines: list[str] = field(default_factory=list)
    date: str | None = None  # ISO date for news results
    kind: str = "web"  # web | news
    score: float | None = None  # cross-encoder relevance 0..1 (None = not re-ranked)
    passages: list[str] = field(default_factory=list)  # best passages after reading the page

    @property
    def read(self) -> bool:
        return bool(self.passages)

    @property
    def text(self) -> str:
        """Evidence text: the page's best passages if it was read, else the snippet."""
        if not self.passages:
            return self.snippet
        return (
            "\n…\n".join([self.snippet, *self.passages])
            if self.snippet
            else "\n…\n".join(self.passages)
        )


@dataclass
class SearchReport:
    query: str
    hits: list[WebHit]
    engines_ok: list[str] = field(default_factory=list)
    engines_failed: list[str] = field(default_factory=list)
    fresh: bool = False  # time-sensitive query: news vertical included
    cached: bool = False
    dropped: int = 0  # results removed by the relevance gate

    @property
    def pages_read(self) -> int:
        return sum(h.read for h in self.hits)


# --- small utilities -------------------------------------------------------------------

_FRESH = re.compile(
    r"\b(?:latest|newest|recent(?:ly)?|today|tonight|yesterday|tomorrow|this (?:week|month|year)|"
    r"current(?:ly)?|right now|news|breaking|live|score|price|stock|weather|forecast|"
    r"released?|launch(?:ed)?|announce[ds]?|update[sd]?|20[2-3]\d)\b",
    re.IGNORECASE,
)


def is_time_sensitive(query: str) -> bool:
    return bool(_FRESH.search(query))


_TRACKING = re.compile(r"^(?:utm_.*|ref|ref_src|fbclid|gclid|mc_cid|mc_eid|igshid|si)$", re.I)


def canonical_url(url: str) -> str:
    """One key per page: lower-case host without www, no fragment, no tracking params."""
    p = urlparse(url.strip())
    host = (p.hostname or "").lower().removeprefix("www.").removeprefix("m.")
    query = urlencode([(k, v) for k, v in parse_qsl(p.query) if not _TRACKING.match(k)])
    path = p.path.rstrip("/") or "/"
    return urlunparse(("https", host, path, "", query, ""))


def domain_of(url: str) -> str:
    return (urlparse(url).hostname or "").lower().removeprefix("www.")


# Social / image sites: snippets without substance, or login walls.
_JUNK = ("pinterest.", "facebook.com", "instagram.com", "tiktok.com")
# Fine as results, but not worth fetching (JS apps, login walls, video).
_NO_READ = ("youtube.com", "youtu.be", "reddit.com", "x.com", "twitter.com", "linkedin.com")


def _is(domain: str, patterns: tuple[str, ...]) -> bool:
    return any(domain == p or domain.endswith("." + p) or p in domain for p in patterns)


class _TTLCache:
    def __init__(self, ttl_s: float, max_items: int = 256) -> None:
        self.ttl_s, self.max_items = ttl_s, max_items
        self._data: dict[str, tuple[float, object]] = {}
        self._lock = threading.Lock()

    def get(self, key: str):
        with self._lock:
            item = self._data.get(key)
            if item and time.monotonic() - item[0] < self.ttl_s:
                return item[1]
            self._data.pop(key, None)
            return None

    def put(self, key: str, value) -> None:
        with self._lock:
            if len(self._data) >= self.max_items:
                self._data.pop(next(iter(self._data)))  # oldest first (insertion order)
            self._data[key] = (time.monotonic(), value)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


_search_cache = _TTLCache(settings.web_cache_ttl_s)
_page_cache = _TTLCache(settings.web_cache_ttl_s * 2, max_items=64)


class _CircuitBreaker:
    """Skip an engine for ``cooldown_s`` after ``threshold`` consecutive failures."""

    def __init__(self, threshold: int = 3, cooldown_s: float = 120.0) -> None:
        self.threshold, self.cooldown_s = threshold, cooldown_s
        self._failures: dict[str, tuple[int, float]] = {}
        self._lock = threading.Lock()

    def available(self, engine: str) -> bool:
        with self._lock:
            count, since = self._failures.get(engine, (0, 0.0))
            if count < self.threshold:
                return True
            if time.monotonic() - since > self.cooldown_s:  # half-open: try again
                self._failures[engine] = (self.threshold - 1, time.monotonic())
                return True
            return False

    def record(self, engine: str, ok: bool) -> None:
        with self._lock:
            if ok:
                self._failures.pop(engine, None)
            else:
                count, _ = self._failures.get(engine, (0, 0.0))
                self._failures[engine] = (count + 1, time.monotonic())


_breaker = _CircuitBreaker()


# --- providers -----------------------------------------------------------------------------


def _search_tavily(query: str) -> list[dict]:
    from langchain_tavily import TavilySearch

    result = TavilySearch(max_results=8).invoke({"query": query})
    results = result.get("results", []) if isinstance(result, dict) else []
    return [
        {"title": r.get("title") or r.get("url"), "url": r.get("url"), "snippet": r.get("content")}
        for r in results
    ]


_WIKI_API = "https://en.wikipedia.org/w/api.php"


def _search_wikipedia(query: str) -> list[dict]:
    """Wikipedia's own search API: keyless, fast and rarely rate-limited, so the
    pipeline still has evidence for definitional questions when every
    commercial engine is blocking us."""
    with httpx.Client(timeout=_TIMEOUT, headers={"User-Agent": USER_AGENT}) as client:
        resp = client.get(
            _WIKI_API,
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": 3,
                "srprop": "snippet",
                "format": "json",
            },
        )
        resp.raise_for_status()
        rows = resp.json().get("query", {}).get("search", [])
    return [
        {
            "title": f"Wikipedia: {r['title']}",
            "url": "https://en.wikipedia.org/wiki/" + r["title"].replace(" ", "_"),
            "snippet": re.sub(r"<[^>]+>", "", r.get("snippet", "")),
        }
        for r in rows
    ]


def _engine_search(engine: str, query: str) -> list[dict]:
    """One engine via the ddgs metasearch library (no API key)."""
    if engine == "wikipedia":
        return _search_wikipedia(query)
    from ddgs import DDGS

    rows = DDGS(timeout=6).text(query, max_results=8, backend=engine) or []
    return [
        {"title": r.get("title") or r.get("href"), "url": r.get("href"), "snippet": r.get("body")}
        for r in rows
    ]


def _search_news(query: str) -> list[dict]:
    from ddgs import DDGS

    rows = DDGS(timeout=6).news(query, max_results=6, timelimit="m") or []
    return [
        {
            "title": r.get("title") or r.get("url"),
            "url": r.get("url"),
            "snippet": r.get("body"),
            "date": (r.get("date") or "")[:10] or None,
            "kind": "news",
        }
        for r in rows
    ]


@dataclass
class EngineRun:
    results: dict[str, list[dict]] = field(default_factory=dict)  # engine -> rows
    failed: list[str] = field(default_factory=list)  # errors and timeouts
    empty: list[str] = field(default_factory=list)  # answered "no results"


SUPPLEMENTS = frozenset({"wikipedia", "news"})  # add to web results; never end the wait


def _hedged(
    jobs: dict[str, Callable[[], list[dict]]], first_grace_s: float, deadline_s: float
) -> EngineRun:
    """Run provider calls in parallel; return once ``first_grace_s`` has passed since
    the first general web engine answered (or two did, or the deadline hit).
    Stragglers keep running in the pool and are ignored.

    Supplements (Wikipedia, news) don't start the grace clock: Wikipedia always
    answers first, and stopping on it would cut off the real web engines.
    """
    futures: dict[Future, str] = {_POOL.submit(fn): name for name, fn in jobs.items()}
    run = EngineRun()
    results, failed = run.results, run.failed
    start = time.monotonic()
    first_ok: float | None = None
    pending = set(futures)
    while pending:
        now = time.monotonic()
        limit = start + deadline_s
        if first_ok is not None:
            limit = min(limit, first_ok + first_grace_s)
        if now >= limit:
            break
        done, pending = wait(pending, timeout=limit - now, return_when=FIRST_COMPLETED)
        for fut in done:
            name = futures[fut]
            try:
                rows = [r for r in (fut.result() or []) if r.get("url")]
            except Exception as exc:  # noqa: BLE001 — one engine failing is normal
                log.info("web engine %s failed: %s", name, type(exc).__name__)
                # ddgs reports an empty result page as an exception
                (run.empty if "no results" in str(exc).lower() else failed).append(name)
                continue
            if rows:
                results[name] = rows
                if name not in SUPPLEMENTS:
                    first_ok = first_ok or time.monotonic()
            else:
                run.empty.append(name)
        if len([n for n in results if n not in SUPPLEMENTS]) >= 2:
            break
        if pending and all(futures[f] in SUPPLEMENTS for f in pending) and first_ok is None:
            # every general engine has already failed: don't wait on supplements alone
            first_ok = time.monotonic()
    failed += [futures[f] for f in pending]  # timed out
    return run


def _search_keyless(query: str, fresh: bool) -> EngineRun:
    engines = [e.strip() for e in settings.web_engines.split(",") if e.strip()]
    live = [e for e in engines if _breaker.available(e)] or engines
    jobs = {e: (lambda e=e: _engine_search(e, query)) for e in live}
    if fresh:
        jobs["news"] = lambda: _search_news(query)
    run = _hedged(jobs, first_grace_s=1.5, deadline_s=settings.web_search_deadline_s)
    for e in live:
        _breaker.record(e, e in run.results)
    return run


# --- ranking ------------------------------------------------------------------------------


def _fuse(per_engine: dict[str, list[dict]]) -> list[WebHit]:
    """Merge engine rankings with RRF; a page several engines agree on rises."""
    hits: dict[str, WebHit] = {}
    rankings = []
    for engine, rows in per_engine.items():
        ranking = []
        for r in rows:
            url = str(r["url"])
            if _is(domain_of(url), _JUNK):
                continue
            key = canonical_url(url)
            if key in ranking:
                continue
            ranking.append(key)
            hit = hits.get(key)
            if hit is None:
                hits[key] = WebHit(
                    title=str(r.get("title") or url).strip(),
                    url=url,
                    snippet=re.sub(r"\s+", " ", str(r.get("snippet") or "")).strip(),
                    date=r.get("date"),
                    kind=r.get("kind", "web"),
                )
                hit = hits[key]
            elif len(str(r.get("snippet") or "")) > len(hit.snippet):
                hit.snippet = re.sub(r"\s+", " ", str(r["snippet"])).strip()
            hit.engines.append(engine)
        rankings.append(ranking)
    fused = [hits[k] for k, _ in reciprocal_rank_fusion(rankings)]
    # at most two results per site, so one domain can't crowd out the rest
    per_domain: dict[str, int] = {}
    out = []
    for h in fused:
        d = domain_of(h.url)
        per_domain[d] = per_domain.get(d, 0) + 1
        if per_domain[d] <= 2:
            out.append(h)
    return out


def _rerank(query: str, hits: list[WebHit]) -> list[WebHit]:
    reranker = get_reranker()
    if reranker is None or not hits:
        return hits
    try:
        scores = reranker.score(query, [f"{h.title}. {h.snippet}" for h in hits])
    except Exception:  # noqa: BLE001 — keep the fused order
        log.exception("web re-rank failed")
        return hits
    for h, s in zip(hits, scores, strict=True):
        h.score = round(s, 4)
    return sorted(hits, key=lambda h: h.score or 0.0, reverse=True)


# --- reading pages ----------------------------------------------------------------------


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
    cached = _page_cache.get(url)
    if cached is not None:
        return cached
    headers = {"User-Agent": USER_AGENT, "Accept-Language": "en"}
    with httpx.Client(timeout=_TIMEOUT, headers=headers) as client:
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
                result = html_to_text(raw) if "html" in ctype else (raw, "")
                _page_cache.put(url, result)
                return result
        raise BlockedURLError("too many redirects")


_WORD = re.compile(r"[a-z0-9]{3,}")
_STOPWORDS = frozenset(
    "the a an is are was were be of and or to in on at for with as by that this it its from "
    "which can will has have not but also these their they you your".split()
)


def is_prose(passage: str) -> bool:
    """Boilerplate filter (the jusText idea): real sentences are ~30-50% stopwords;
    navigation menus, tag clouds and link lists have almost none."""
    words = re.findall(r"[a-zA-Z']+", passage.lower())
    if len(words) < 12:
        return False
    return sum(w in _STOPWORDS for w in words) / len(words) >= 0.15


def passages_of(text: str, size: int = 700) -> list[str]:
    """Split page text into ~``size``-char windows on paragraph boundaries."""
    out, buf = [], ""
    for para in re.split(r"\n\s*\n", text):
        para = re.sub(r"\s+", " ", para).strip()
        if len(para) < 40 or not is_prose(para):  # menus, captions, link lists
            continue
        if buf and len(buf) + len(para) > size:
            out.append(buf)
            buf = ""
        buf = f"{buf} {para}".strip() if buf else para[: size * 2]
    if buf:
        out.append(buf)
    return out


def best_passages(
    query: str, text: str, keep: int = 2, shortlist: int = 6
) -> tuple[list[str], float | None]:
    """The page's most relevant passages and the best one's relevance score.

    A cheap keyword-overlap shortlist first, then the cross-encoder on those
    few: scoring every window of a long page would take seconds on CPU.
    """
    windows = passages_of(text)
    if not windows:
        return [], None
    terms = set(_WORD.findall(query.lower()))
    overlap = [len(terms & set(_WORD.findall(w.lower()))) for w in windows]
    order = sorted(range(len(windows)), key=lambda i: -overlap[i])
    shortlisted = [windows[i] for i in order[:shortlist]]
    reranker = get_reranker()
    if reranker is None:
        return shortlisted[:keep], None
    scores = reranker.score(query, shortlisted)
    ranked = sorted(zip(scores, shortlisted, strict=True), key=lambda x: -x[0])
    # passages must be clearly on-topic, not merely "not irrelevant"
    floor = max(settings.web_min_score, 0.2)
    return [w for s, w in ranked[:keep] if s >= floor], ranked[0][0]


def _deep_read(query: str, hits: list[WebHit]) -> None:
    """Read the top results in parallel; a page that fails just keeps its snippet."""
    targets = [
        h
        for h in hits
        if not _is(domain_of(h.url), _NO_READ) and not h.url.lower().endswith(".pdf")
    ][: settings.web_deep_read]
    if not targets:
        return

    def read(hit: WebHit) -> None:
        text, _ = fetch_url_text(hit.url)
        hit.passages, best = best_passages(query, text)
        if best is not None and hit.passages:
            # judge the page by its best passage, not only its 200-char snippet
            hit.score = round(max(hit.score or 0.0, best), 4)

    futures = {_POOL.submit(read, h): h for h in targets}
    done, _ = wait(futures, timeout=settings.web_read_deadline_s)
    for fut in done:
        if fut.exception() is not None:
            log.info("could not read %s: %s", futures[fut].url, type(fut.exception()).__name__)


# --- entry point ----------------------------------------------------------------------------


def search(query: str) -> SearchReport:
    """Search the web and return relevant, read results (best first).

    Raises ``WebSearchUnavailable`` only if every provider failed; "nothing
    relevant" is an empty ``hits`` list, so callers can tell the two apart.
    """
    key = " ".join(query.lower().split())
    cached = _search_cache.get(key)
    if cached is not None:
        return SearchReport(**{**cached.__dict__, "cached": True})

    fresh = is_time_sensitive(query)
    per_engine: dict[str, list[dict]] = {}
    failed: list[str] = []
    empty: list[str] = []
    if settings.tavily_api_key:
        try:
            rows = [r for r in _search_tavily(query) if r.get("url")]
            if rows:
                per_engine["tavily"] = rows
            else:
                empty.append("tavily")
        except Exception as exc:  # noqa: BLE001 — fall back to keyless search
            log.warning("Tavily failed (%s); using keyless search", type(exc).__name__)
            failed.append("tavily")
    if "tavily" not in per_engine:
        run = _search_keyless(query, fresh)
        per_engine.update(run.results)
        failed += run.failed
        empty += run.empty
    if not per_engine:
        if empty:  # engines answered, with nothing: an honest "no results"
            return SearchReport(query, [], engines_failed=sorted(set(failed)), fresh=fresh)
        raise WebSearchUnavailable(f"all search engines failed ({', '.join(failed) or 'none'})")

    hits = _rerank(query, _fuse(per_engine))
    # Read the top pages BEFORE gating: a relevant page with a thin snippet
    # ("A guide to mTLS.") would otherwise be thrown away unread.
    if settings.web_deep_read > 0:
        _deep_read(query, hits)
        hits.sort(key=lambda h: h.score or 0.0, reverse=True)
    relevant = [h for h in hits if h.score is None or h.score >= settings.web_min_score]
    report = SearchReport(
        query=query,
        hits=relevant[: settings.web_max_results],
        engines_ok=sorted(per_engine),
        engines_failed=sorted(set(failed)),
        fresh=fresh,
        dropped=len(hits) - len(relevant),
    )
    _search_cache.put(key, report)
    return report
