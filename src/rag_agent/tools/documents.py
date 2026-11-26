"""Document Analyst tools — everything that reads the private knowledge base.

| Tool                    | What it does                                             |
|-------------------------|----------------------------------------------------------|
| list_documents          | Inventory of indexed documents (also used by the Planner)|
| search_documents        | Hybrid search across all documents                       |
| search_within_document  | Hybrid search restricted to one document                 |
| get_adjacent_chunks     | Expand a cited passage with its neighbouring chunks      |
| summarize_document      | LLM summary of a whole document                          |
| compare_documents       | Side-by-side evidence from two documents on one aspect   |
| calculator              | Exact arithmetic over numbers found in documents         |
"""

from __future__ import annotations

import ast
import math
import operator

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from rag_agent.citations import registry_from_config
from rag_agent.llms import with_fallback
from rag_agent.retrieval import RetrievedChunk, hybrid_search
from rag_agent.tools.common import (
    SNIPPET_CHARS,
    emit_sources,
    format_evidence,
    logs_attempt,
    register_chunk,
    resolve_source,
    unknown_source_message,
)
from rag_agent.vectorstore import chunks_for_source, source_stats

NO_DOCS = "NO_RELEVANT_DOCUMENTS: nothing in the document collection is relevant to this query."


def _evidence_result(tool_name: str, hits: list[RetrievedChunk], config: RunnableConfig) -> str:
    registry = registry_from_config(config)
    entries = [register_chunk(registry, h) for h in hits]
    emit_sources(tool_name, entries)
    return format_evidence(entries, [h.text for h in hits])


@tool
def list_documents() -> str:
    """List every document in the private knowledge base with its title and size.

    Use this to learn what the collection covers before deciding where to search.
    """
    stats = source_stats()
    if not stats:
        return "The document collection is empty. No documents have been uploaded."
    lines = [
        f'- {d["source"]}: "{d["title"]}" ({d["chunks"]} chunks'
        + (f", {d['pages']} pages" if d.get("pages") else "")
        + ")"
        for d in stats
    ]
    return f"{len(stats)} documents indexed:\n" + "\n".join(lines)


@tool
@logs_attempt("query")
def search_documents(query: str, config: RunnableConfig) -> str:
    """Search ALL private documents for passages relevant to the query.

    Works for natural-language questions and exact identifiers (error codes,
    config keys, version numbers). Returns passages labelled [n] for citation.
    """
    hits = hybrid_search(query)
    return _evidence_result("search_documents", hits, config) if hits else NO_DOCS


@tool
@logs_attempt("query")
def search_within_document(document: str, query: str, config: RunnableConfig) -> str:
    """Search inside ONE named document (use a file name from list_documents)."""
    source = resolve_source(document)
    if source is None:
        return unknown_source_message(document)
    hits = hybrid_search(query, source=source)
    if not hits:
        return f"NO_RELEVANT_DOCUMENTS: nothing in {source} is relevant to this query."
    return _evidence_result("search_within_document", hits, config)


@tool
def get_adjacent_chunks(citation: int, config: RunnableConfig, window: int = 1) -> str:
    """Read the text just before and after a cited document passage.

    Use when passage [citation] looks cut off or needs surrounding context.
    """
    registry = registry_from_config(config)
    ref = registry.ref(citation)
    if not ref or ref.get("chunk_index") is None:
        return f"INVALID_CITATION: [{citation}] is not a document passage found in this turn."
    window = max(1, min(window, 2))
    center = int(ref["chunk_index"])
    wanted = [i for i in range(center - window, center + window + 1) if i != center and i >= 0]
    rows = chunks_for_source(ref["source"], wanted)
    if not rows:
        return f"[{citation}] has no neighbouring passages (it is the whole document)."
    hits = [RetrievedChunk(id=cid, text=text, metadata=md, relevance=0.0) for cid, text, md in rows]
    return _evidence_result("get_adjacent_chunks", hits, config)


_SUMMARY_INPUT_CHARS = 6000


@tool
def summarize_document(document: str, config: RunnableConfig) -> str:
    """Summarize one whole document (file name from list_documents).

    Use for broad questions like "what is this document about?".
    """
    source = resolve_source(document)
    if source is None:
        return unknown_source_message(document)
    rows = chunks_for_source(source)
    full = "\n\n".join(text for _, text, _ in rows)
    truncated = len(full) > _SUMMARY_INPUT_CHARS
    model = with_fallback("summarizer", lambda m: m)
    reply = model.invoke(
        [
            SystemMessage(
                "Summarize the document in at most 150 words: purpose, key facts and "
                "numbers, and notable limitations. Use only the document text."
            ),
            HumanMessage(full[:_SUMMARY_INPUT_CHARS]),
        ],
        config,
    )
    summary = reply.text.strip()
    registry = registry_from_config(config)
    entry = registry.register(
        key=f"summary:{source}",
        source={
            "kind": "document",
            "title": f"{source} (summary)",
            "snippet": summary[:SNIPPET_CHARS],
            "score": None,
        },
        text=summary,
    )
    emit_sources("summarize_document", [entry])
    note = " (summary of the first part; document was truncated)" if truncated else ""
    return f"[{entry['index']}] Summary of {source}{note}:\n{summary}"


@tool
def compare_documents(document_a: str, document_b: str, aspect: str, config: RunnableConfig) -> str:
    """Pull evidence about one aspect from two documents, side by side.

    Use for "how does X differ between A and B" or to check two documents agree.
    """
    sections = []
    for name in (document_a, document_b):
        source = resolve_source(name)
        if source is None:
            sections.append(unknown_source_message(name))
            continue
        hits = hybrid_search(aspect, k=2, source=source)
        body = _evidence_result("compare_documents", hits, config) if hits else "(nothing relevant)"
        sections.append(f"=== {source} ===\n{body}")
    return "\n\n".join(sections)


# --- calculator ------------------------------------------------------------------

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "log10": math.log10,
    "ceil": math.ceil,
    "floor": math.floor,
}


def safe_eval(expression: str) -> float | int:
    """Evaluate arithmetic WITHOUT ``eval``: walk the AST and allow only math.

    Never ``eval`` LLM-generated text — it is arbitrary code execution. This
    whitelist accepts numbers, + - * / // % **, parentheses and a few functions.
    """

    def ev(node: ast.AST):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("exponent too large")
            return _BIN_OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](ev(node.operand))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCS
            and not node.keywords
        ):
            return _FUNCS[node.func.id](*(ev(a) for a in node.args))
        raise ValueError(f"unsupported expression element: {type(node).__name__}")

    if len(expression) > 200:
        raise ValueError("expression too long")
    return ev(ast.parse(expression.replace("^", "**"), mode="eval"))


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression exactly, e.g. "730 / 14" or "256 * 1024".

    Use for any computation over numbers found in the evidence — never do
    arithmetic in your head.
    """
    try:
        result = safe_eval(expression)
    except (ValueError, SyntaxError, ZeroDivisionError, TypeError, OverflowError) as exc:
        return f"CALCULATOR_ERROR: {exc}"
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return f"{expression} = {result}"


DOCUMENT_TOOLS = [
    search_documents,
    search_within_document,
    get_adjacent_chunks,
    summarize_document,
    compare_documents,
    calculator,
    list_documents,
]
