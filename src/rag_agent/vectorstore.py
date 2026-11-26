"""Chroma vector store, persisted to disk for zero-setup RAG.

Besides the LangChain ``Chroma`` wrapper, this module keeps a ``generation``
counter that is bumped on every write. The in-memory BM25 index compares it
to know when to rebuild, so keyword search never goes stale after an upload.
"""

from __future__ import annotations

import itertools
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document

from rag_agent.config import settings
from rag_agent.embeddings import get_embeddings

_generation = itertools.count(1)
_current_generation = 0


def _bump() -> None:
    global _current_generation
    _current_generation = next(_generation)


def generation() -> int:
    """Monotonic counter that changes whenever stored chunks change."""
    return _current_generation


@lru_cache
def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_dir,
        # Cosine distance: BGE vectors are normalised, and cosine gives
        # relevance scores in a stable [0, 1] range (1 - distance).
        collection_metadata={"hnsw:space": "cosine"},
    )


def collection_count() -> int:
    """Number of stored chunks (0 when empty / fresh)."""
    try:
        return get_vectorstore()._collection.count()
    except Exception:  # noqa: BLE001 — health checks must never raise
        return 0


def add_chunks(chunks: list[Document], ids: list[str]) -> None:
    get_vectorstore().add_documents(chunks, ids=ids)
    _bump()


def delete_by_source(source: str) -> int:
    """Remove every chunk that came from ``source``. Returns how many were removed."""
    col = get_vectorstore()._collection
    ids = col.get(where={"source": source}, include=[]).get("ids", [])
    if ids:
        col.delete(ids=ids)
        _bump()
    return len(ids)


def all_chunks() -> list[tuple[str, str, dict]]:
    """Every stored chunk as ``(id, text, metadata)`` — the BM25 corpus."""
    got = get_vectorstore()._collection.get(include=["documents", "metadatas"])
    return list(
        zip(
            got.get("ids") or [],
            got.get("documents") or [],
            got.get("metadatas") or [],
            strict=True,
        )
    )


def list_sources() -> list[str]:
    """Distinct source names currently indexed."""
    col = get_vectorstore()._collection
    metas = col.get(include=["metadatas"]).get("metadatas") or []
    return sorted({m.get("source", "unknown") for m in metas if m})


def source_stats() -> list[dict]:
    """Per-document inventory: name, chunk count, pages, first heading."""
    got = get_vectorstore()._collection.get(include=["documents", "metadatas"])
    docs: dict[str, dict] = {}
    for text, md in zip(got.get("documents") or [], got.get("metadatas") or [], strict=True):
        src = (md or {}).get("source", "unknown")
        d = docs.setdefault(src, {"source": src, "chunks": 0, "pages": set(), "title": ""})
        d["chunks"] += 1
        if md.get("page"):
            d["pages"].add(md["page"])
        if md.get("chunk_index") == 0:
            first = next((ln.strip("# ").strip() for ln in text.splitlines() if ln.strip()), "")
            d["title"] = first[:120]
    return [
        {**d, "pages": len(d["pages"]) or None}
        for d in sorted(docs.values(), key=lambda x: x["source"])
    ]


def chunks_for_source(source: str, indices: list[int] | None = None) -> list[tuple[str, str, dict]]:
    """Chunks of one document as ``(id, text, metadata)``, in document order."""
    where: dict = {"source": source}
    if indices is not None:
        where = {"$and": [{"source": source}, {"chunk_index": {"$in": indices}}]}
    got = get_vectorstore()._collection.get(where=where, include=["documents", "metadatas"])
    rows = list(
        zip(
            got.get("ids") or [],
            got.get("documents") or [],
            got.get("metadatas") or [],
            strict=True,
        )
    )
    return sorted(rows, key=lambda r: r[2].get("chunk_index", 0))
