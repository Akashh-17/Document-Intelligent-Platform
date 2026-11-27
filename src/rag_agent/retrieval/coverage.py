"""Whole-document reading for questions about ALL of a document.

Search returns the top-k passages, which is right for "what does error X
mean?" and wrong for "list all the projects in the PDF" or "explain this
document": the answer is spread over the whole file, and top-4 drops part of
it (the system once listed 4 of a guide's 5 projects).

Sending every chunk would blow the model's token budget (a 13-page PDF is
~25K characters). So we pick for *coverage*: split the document into sections
(its pages, or equal slices for text files), keep the passage of each section
that best matches the question (cross-encoder), and return them in document
order. Every part of the document is represented, and the reader sees it in
the order it was written, which is what listing and summarizing need.
"""

from __future__ import annotations

import logging

from rag_agent.config import settings
from rag_agent.retrieval.bm25 import tokenize
from rag_agent.retrieval.hybrid import RetrievedChunk
from rag_agent.retrieval.rerank import get_reranker
from rag_agent.vectorstore import chunks_for_source

log = logging.getLogger(__name__)

SECTIONS_FOR_TEXT = 12  # text/Markdown files have no pages: split into this many slices
SHORTLIST = 3  # per section, cheap keyword shortlist before the cross-encoder


def coverage_passages(
    source: str, question: str, budget: int | None = None
) -> list[RetrievedChunk]:
    """The best passage of every page/section of ``source``, in document order."""
    rows = chunks_for_source(source)  # (id, text, metadata), in document order
    if not rows:
        return []
    budget = budget or settings.whole_document_char_budget

    # 1. group into sections: PDF pages, or equal slices of a text file
    sections: dict[int, list[tuple[str, str, dict]]] = {}
    has_pages = any(md.get("page") for _, _, md in rows)
    for i, row in enumerate(rows):
        key = (row[2].get("page") or 0) if has_pages else i * SECTIONS_FOR_TEXT // len(rows)
        sections.setdefault(int(key), []).append(row)

    # 2. shortlist per section by keyword overlap, then score with the cross-encoder
    terms = set(tokenize(question))
    shortlist: list[tuple[int, tuple[str, str, dict]]] = []
    for key, group in sections.items():
        ranked = sorted(group, key=lambda r: -len(terms & set(tokenize(r[1]))))
        shortlist += [(key, r) for r in ranked[:SHORTLIST]]
    scores = [0.0] * len(shortlist)
    reranker = get_reranker()
    if reranker is not None:
        try:
            scores = reranker.score(question, [r[1] for _, r in shortlist])
        except Exception:  # noqa: BLE001 — keyword order is a fine fallback
            log.exception("coverage re-rank failed")

    best: dict[int, tuple[float, tuple[str, str, dict]]] = {}
    for (key, row), score in zip(shortlist, scores, strict=True):
        if key not in best or score > best[key][0]:
            best[key] = (score, row)

    # 3. fit the budget: drop the least relevant sections, keep document order
    chosen = sorted(best.items(), key=lambda kv: -kv[1][0])
    kept, used = [], 0
    for key, (score, row) in chosen:
        if used + len(row[1]) > budget and kept:
            continue
        kept.append((key, score, row))
        used += len(row[1])
    kept.sort(key=lambda k: (k[0], k[2][2].get("chunk_index", 0)))
    return [
        RetrievedChunk(id=cid, text=text, metadata=md, relevance=round(score, 4))
        for _, score, (cid, text, md) in kept
    ]
