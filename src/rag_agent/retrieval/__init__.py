"""Hybrid retrieval: BM25 keyword search + dense vector search, fused with
Reciprocal Rank Fusion, then re-ranked by a cross-encoder.

    query ─┬─> dense (Chroma, BGE embeddings) ─┐
           │                                   ├─> RRF fusion ─> cross-encoder ─> threshold ─> top-k
           └─> sparse (BM25 over all chunks) ──┘

Why both? Dense embeddings understand paraphrases ("how long is data kept?")
but are weak on exact identifiers (error codes, config keys, version numbers).
BM25 is the opposite. Fusing the two rankings is robust to either failing.
"""

from rag_agent.retrieval.hybrid import RetrievedChunk, hybrid_search

__all__ = ["RetrievedChunk", "hybrid_search"]
