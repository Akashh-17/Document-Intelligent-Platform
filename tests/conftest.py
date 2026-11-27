"""Test setup: isolated storage and no real API keys.

Settings are read when ``rag_agent.config`` is first imported, so the
environment must be prepared here, before any test module imports rag_agent.
"""

from __future__ import annotations

import os
import tempfile

_TMP = tempfile.mkdtemp(prefix="rag-agent-tests-")
os.environ["CHROMA_DIR"] = os.path.join(_TMP, "chroma")
os.environ["SQLITE_PATH"] = os.path.join(_TMP, "memory.sqlite")
os.environ["CHROMA_COLLECTION"] = "test_documents"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["GROQ_API_KEY"] = ""  # tests never call the real LLM
os.environ["TAVILY_API_KEY"] = ""
os.environ["WEB_DEEP_READ"] = "0"  # tests never fetch real pages (see test_websearch.py)
