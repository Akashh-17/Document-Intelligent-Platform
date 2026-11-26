"""Cross-encoder re-ranking (local, via fastembed / ONNX).

A bi-encoder (our embedding model) encodes query and passage *separately*, so
it can pre-compute passage vectors and search millions quickly, but it never
sees the two texts together. A cross-encoder reads ``[query, passage]`` as one
input and scores how well the passage answers the query: much more accurate,
but too slow to run over a whole corpus. So we use it only on the ~10 fused
candidates — the classic "retrieve wide, re-rank narrow" pattern.

The ms-marco MiniLM cross-encoder outputs a raw logit; we squash it with a
sigmoid to get a 0..1 relevance score that can be thresholded.
"""

from __future__ import annotations

import logging
import math
from functools import lru_cache

from rag_agent.config import settings

log = logging.getLogger(__name__)


class Reranker:
    def __init__(self, model_name: str) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model_name = model_name
        self._model = TextCrossEncoder(model_name=model_name)

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Relevance in [0, 1] for each passage, in input order."""
        if not passages:
            return []
        logits = self._model.rerank(query, passages)
        return [1.0 / (1.0 + math.exp(-float(x))) for x in logits]


@lru_cache
def get_reranker() -> Reranker | None:
    """The shared reranker, or ``None`` if disabled or it failed to load.

    Failing to load (e.g. offline on first run, so the weights can't download)
    degrades retrieval to RRF-only instead of breaking every query.
    """
    if not settings.rerank_enabled:
        return None
    try:
        return Reranker(settings.reranker_model)
    except Exception:
        log.exception(
            "Could not load reranker %r — continuing without re-ranking", settings.reranker_model
        )
        return None
