"""Per-client sliding-window rate limit for the chat endpoint.

Each chat turn costs several LLM calls against a shared Groq quota, so one
client hammering the endpoint could exhaust it for everyone. In-memory and
per-process — fine for a single-instance deployment; behind a load balancer
you'd move this to Redis or the API gateway.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque


class RateLimiter:
    def __init__(self, limit: int, window_s: float = 60.0) -> None:
        self.limit = limit
        self.window_s = window_s
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> float | None:
        """Record a hit. Returns None if allowed, else seconds until a slot frees up."""
        if self.limit <= 0:
            return None
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] >= self.window_s:
                hits.popleft()
            if len(hits) >= self.limit:
                return round(self.window_s - (now - hits[0]), 1)
            hits.append(now)
            return None
