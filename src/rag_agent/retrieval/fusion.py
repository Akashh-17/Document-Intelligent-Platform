"""Reciprocal Rank Fusion (Cormack, Clarke & Büttcher, SIGIR 2009).

    RRF(d) = Σ_r  1 / (k + rank_r(d))      over every ranking r that contains d

Only *ranks* are used, never raw scores. That is the point: BM25 scores
(unbounded, corpus-dependent) and cosine similarities (0..1) are not on
comparable scales, so adding or averaging them is meaningless. Ranks are.

``k`` (default 60, the value from the paper) damps the advantage of being
ranked #1 by a single retriever, so a document that both retrievers rank
moderately well beats one that only a single retriever ranks first.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Hashable]], k: int = 60
) -> list[tuple[Hashable, float]]:
    """Fuse ranked ID lists (best first) into one list of ``(id, score)``, best first.

    Ties keep the order in which IDs were first seen, so results are deterministic.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    scores: dict[Hashable, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    # sorted() is stable, and dicts preserve insertion order -> deterministic ties
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)
