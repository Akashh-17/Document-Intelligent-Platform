"""The platform's 13 tools, grouped by the agent that owns them.

Query Planner     list_documents
Document Analyst  search_documents, read_whole_document, search_within_document,
                  get_adjacent_chunks, summarize_document, compare_documents, calculator
                  (+ list_documents, shared)
Web Researcher    web_search, fetch_webpage, wikipedia_lookup
Fact Verifier     verify_citations, check_groundedness
"""

from rag_agent.tools.documents import (
    DOCUMENT_TOOLS,
    calculator,
    compare_documents,
    get_adjacent_chunks,
    list_documents,
    read_whole_document,
    search_documents,
    search_within_document,
    summarize_document,
)
from rag_agent.tools.verification import (
    VERIFICATION_TOOLS,
    check_groundedness,
    verify_citations,
)
from rag_agent.tools.web import WEB_TOOLS, fetch_webpage, web_search, wikipedia_lookup

ALL_TOOLS = [*DOCUMENT_TOOLS, *WEB_TOOLS, *VERIFICATION_TOOLS]

__all__ = [
    "ALL_TOOLS",
    "DOCUMENT_TOOLS",
    "VERIFICATION_TOOLS",
    "WEB_TOOLS",
    "calculator",
    "check_groundedness",
    "compare_documents",
    "fetch_webpage",
    "get_adjacent_chunks",
    "list_documents",
    "read_whole_document",
    "search_documents",
    "search_within_document",
    "summarize_document",
    "verify_citations",
    "web_search",
    "wikipedia_lookup",
]
