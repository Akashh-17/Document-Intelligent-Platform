"""Turn-wide evidence store with stable citation numbers.

One ``CitationRegistry`` is created per chat turn and handed to every agent
and tool through ``config["configurable"]["citations"]``. It solves two problems:

1. **Consistent numbering.** Every piece of evidence any tool finds is
   registered and receives a number that is unique for the whole turn.
   Registering the same chunk / URL again returns its existing number, so the
   answer's [n] markers always match the Sources panel — even when several
   agents run in parallel.
2. **Shared evidence.** The Document Analyst and Web Researcher run as separate
   sub-agents; whatever they find lands here, so the synthesis step sees all of
   it with full text, not just the workers' short summaries.

Thread-safe, because LangGraph runs parallel branches / tool calls in worker threads.
"""

from __future__ import annotations

import threading
from typing import Any

from langchain_core.runnables import RunnableConfig

CONFIG_KEY = "citations"


class CitationRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_key: dict[str, int] = {}
        self.sources: list[dict[str, Any]] = []  # public, UI-safe entries
        self._texts: dict[int, str] = {}  # full evidence text, kept server-side
        self._refs: dict[int, dict[str, Any]] = {}  # e.g. chunk id / metadata
        self.attempts: list[dict[str, Any]] = []  # every search tried this turn

    def register(
        self,
        key: str,
        source: dict[str, Any],
        text: str | None = None,
        ref: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return ``source`` with its turn-wide ``index`` (1-based) filled in."""
        with self._lock:
            if key in self._by_key:
                return self.sources[self._by_key[key] - 1]
            index = len(self.sources) + 1
            entry = {**source, "index": index, "id": f"src-{index}"}
            self._by_key[key] = index
            self.sources.append(entry)
            if text is not None:
                self._texts[index] = text
            if ref is not None:
                self._refs[index] = ref
            return entry

    def text(self, index: int) -> str:
        with self._lock:
            entry = self.sources[index - 1] if 0 < index <= len(self.sources) else None
            return self._texts.get(index) or (entry or {}).get("snippet", "")

    def ref(self, index: int) -> dict[str, Any] | None:
        return self._refs.get(index)

    def indices(self) -> list[int]:
        with self._lock:
            return [s["index"] for s in self.sources]

    def export(self) -> list[dict[str, Any]]:
        """Everything needed to rebuild this registry with the same numbers later
        (a follow-up like "make it shorter" reuses the previous turn's evidence)."""
        with self._lock:
            keys = {i: k for k, i in self._by_key.items()}
            return [
                {
                    "key": keys[s["index"]],
                    "source": {k: v for k, v in s.items() if k not in ("index", "id")},
                    "text": self._texts.get(s["index"]),
                    "ref": self._refs.get(s["index"]),
                }
                for s in self.sources
            ]

    def __len__(self) -> int:
        return len(self.sources)

    # --- search attempts: lets the "no answer" response explain *why* -------------

    def log_attempt(self, tool: str, query: str, outcome: str, detail: str = "") -> None:
        """Record one search. ``outcome``: ok | no_results | unavailable | blocked | error."""
        with self._lock:
            self.attempts.append(
                {"tool": tool, "query": query, "outcome": outcome, "detail": detail[:200]}
            )


def registry_from_config(config: RunnableConfig | None) -> CitationRegistry:
    """Fetch the turn's registry; fall back to a private one (numbers from 1)."""
    configurable = (config or {}).get("configurable") or {}
    registry = configurable.get(CONFIG_KEY)
    return registry if isinstance(registry, CitationRegistry) else CitationRegistry()
