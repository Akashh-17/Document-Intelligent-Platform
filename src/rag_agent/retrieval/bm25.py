"""BM25 keyword index over every chunk in the vector store.

The index lives in memory and is rebuilt lazily whenever the store's
``generation`` counter changes (bumped on every ingest / delete), so it never
serves stale results after an upload. Rebuilding is O(total tokens) — fine for
thousands of chunks; at millions you would move to a search engine with a
native inverted index (OpenSearch, Elasticsearch, or Chroma/Qdrant sparse vectors).
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from rag_agent import vectorstore

# Small English stopword list: enough to stop "the/what/is" dominating BM25
# on short questions, without dropping meaningful domain words.
_STOPWORDS = frozenset(
    """a an and are as at be by can do does for from how i in is it its me my of on or
    our so that the their then there these this to was we what when where which who
    why will with you your""".split()
)

# Compound technical tokens: "aur-2001", "query.timeout_ms", "aurora.events.v1", "v3.2"
_COMPOUND = re.compile(r"[a-z0-9]+(?:[._\-/][a-z0-9]+)+")
_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens, plus whole compound identifiers.

    "AUR-2001 query.timeout_ms" ->
        ["aur", "2001", "query", "timeout", "ms", "aur-2001", "query.timeout_ms"]

    Keeping the compound form lets an exact identifier match strongly, while the
    parts still match partial mentions ("error 2001", "timeout").
    """
    lowered = text.lower()
    words = [w for w in _WORD.findall(lowered) if w not in _STOPWORDS]
    compounds = _COMPOUND.findall(lowered)
    return words + compounds


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    metadata: dict


class BM25Index:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._generation = -1
        self._chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None

    def _ensure_fresh(self) -> None:
        gen = vectorstore.generation()
        if gen == self._generation:
            return
        with self._lock:
            if gen == self._generation:  # another thread rebuilt it meanwhile
                return
            chunks = [Chunk(*row) for row in vectorstore.all_chunks()]
            corpus = [tokenize(c.text) for c in chunks]
            # BM25Okapi divides by corpus size, so guard the empty case.
            self._bm25 = BM25Okapi(corpus) if corpus else None
            self._chunks = chunks
            self._generation = gen

    def search(self, query: str, k: int, source: str | None = None) -> list[tuple[Chunk, float]]:
        """Top-k chunks with a positive BM25 score (zero = no shared terms).

        ``source`` restricts results to one document. IDF statistics still come
        from the whole corpus, which keeps scores comparable across calls.
        """
        self._ensure_fresh()
        tokens = tokenize(query)
        if self._bm25 is None or not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        hits = [
            (self._chunks[i], float(scores[i]))
            for i in ranked
            if scores[i] > 0
            and (source is None or self._chunks[i].metadata.get("source") == source)
        ]
        return hits[:k]


_index = BM25Index()


def get_bm25_index() -> BM25Index:
    return _index
