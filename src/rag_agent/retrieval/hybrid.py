"""Hybrid retriever: dense + BM25 -> RRF -> cross-encoder -> relevance gate.

``hybrid_search`` supports several modes so the retrieval benchmark can compare
them on the same queries:

- ``dense``          vector search only (the original behaviour)
- ``bm25``           keyword search only
- ``hybrid``         dense + BM25 fused with RRF
- ``hybrid_rerank``  hybrid, then cross-encoder re-ranking (production default)

Every returned chunk carries the signals that produced it (dense rank/score,
BM25 rank/score, RRF score, rerank score) so the UI and traces can show *why*
a chunk was retrieved — useful for debugging and for explaining the system.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Literal

from rag_agent.config import settings
from rag_agent.retrieval.bm25 import get_bm25_index
from rag_agent.retrieval.fusion import reciprocal_rank_fusion
from rag_agent.retrieval.rerank import get_reranker
from rag_agent.vectorstore import get_vectorstore

log = logging.getLogger(__name__)

Mode = Literal["dense", "bm25", "hybrid", "hybrid_rerank"]


@dataclass
class RetrievedChunk:
    id: str
    text: str
    metadata: dict
    relevance: float  # the score the relevance gate used (0..1, higher = better)
    dense_rank: int | None = None
    dense_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None

    @property
    def source(self) -> str:
        return self.metadata.get("source", "document")

    def signals(self) -> dict:
        """Compact, JSON-friendly retrieval explanation for the UI."""
        d = asdict(self)
        keep = ("dense_rank", "bm25_rank", "rerank_score", "dense_score", "rrf_score")
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items() if k in keep}


def _dense(query: str, k: int, source: str | None = None) -> list[tuple[str, str, dict, float]]:
    flt = {"source": source} if source else None
    hits = get_vectorstore().similarity_search_with_relevance_scores(query, k=k, filter=flt)
    return [(d.id, d.page_content, d.metadata, float(s)) for d, s in hits if d.id]


def hybrid_search(
    query: str,
    k: int | None = None,
    mode: Mode | None = None,
    source: str | None = None,
    gate: bool = True,
) -> list[RetrievedChunk]:
    """Return up to ``k`` chunks that pass the relevance gate, best first.

    ``source`` restricts the search to one document (exact file name).
    ``gate=False`` skips the relevance gate — used to show "closest matches"
    when nothing was relevant enough to answer from.
    """
    k = k or settings.retriever_k
    mode = mode or settings.retrieval_mode
    reranker = get_reranker() if mode == "hybrid_rerank" else None
    if mode == "hybrid_rerank" and reranker is None:
        mode = "hybrid"  # reranker disabled or failed to load: degrade, don't fail

    candidates: dict[str, RetrievedChunk] = {}

    # --- 1. candidate generation -------------------------------------------
    dense_ids: list[str] = []
    if mode != "bm25":
        for rank, (cid, text, md, score) in enumerate(_dense(query, settings.dense_k, source), 1):
            candidates[cid] = RetrievedChunk(
                id=cid, text=text, metadata=md, relevance=score, dense_rank=rank, dense_score=score
            )
            dense_ids.append(cid)

    bm25_ids: list[str] = []
    if mode != "dense":
        for rank, (chunk, score) in enumerate(
            get_bm25_index().search(query, settings.bm25_k, source=source), 1
        ):
            c = candidates.get(chunk.id) or RetrievedChunk(
                id=chunk.id, text=chunk.text, metadata=chunk.metadata, relevance=0.0
            )
            c.bm25_rank, c.bm25_score = rank, score
            candidates[chunk.id] = c
            bm25_ids.append(chunk.id)

    # --- 2. ordering ---------------------------------------------------------
    if mode == "dense":
        ordered = [candidates[i] for i in dense_ids]
    elif mode == "bm25":
        ordered = [candidates[i] for i in bm25_ids]
    else:
        fused = reciprocal_rank_fusion([dense_ids, bm25_ids], k=settings.rrf_k)
        for cid, score in fused:
            candidates[cid].rrf_score = score
        ordered = [candidates[cid] for cid, _ in fused]

    # --- 3. re-ranking (only the head of the fused list: cross-encoders are slow)
    if reranker is not None:
        head = ordered[: settings.rerank_top_n]
        for chunk, score in zip(head, reranker.score(query, [c.text for c in head]), strict=True):
            chunk.rerank_score = score
            chunk.relevance = score
        ordered = sorted(head, key=lambda c: c.rerank_score or 0.0, reverse=True)

    # --- 4. relevance gate: better to return nothing than confidently irrelevant text
    if not gate:
        return ordered[:k]
    return [c for c in ordered if _passes_gate(c, mode)][:k]


def _passes_gate(c: RetrievedChunk, mode: str) -> bool:
    """Keep a chunk if *any* independent signal vouches for it.

    Measured on the retrieval benchmark: small cross-encoders are excellent at
    ordering but score some correct paraphrase matches near 0, so using the
    rerank score as the only gate throws good evidence away. Off-topic queries,
    by contrast, fail every signal at once (rerank ~0, cosine < 0.45, no shared
    keywords), so an OR of the three still rejects them.
    """
    if mode == "bm25":
        return True  # BM25 only returns chunks sharing at least one query term
    rerank_ok = c.rerank_score is not None and c.rerank_score >= settings.min_rerank_score
    dense_ok = c.dense_score is not None and c.dense_score >= settings.min_relevance_score
    # A top keyword hit (exact error code / config key) passes even when its
    # embedding similarity is mediocre.
    keyword_ok = c.bm25_rank is not None and c.bm25_rank <= settings.bm25_strong_rank
    if not rerank_ok and (dense_ok or keyword_ok):
        # report the signal that actually vouched for the chunk
        c.relevance = max(c.relevance, c.dense_score or 0.0, settings.min_relevance_score)
    return rerank_ok or dense_ok or keyword_ok
