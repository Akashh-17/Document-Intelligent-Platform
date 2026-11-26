"""Retrieval benchmark: dense vs BM25 vs hybrid vs hybrid+rerank.

No LLM calls, no API key — it measures the retriever alone, so it's fast,
free and deterministic. Run:

    uv run python -m evals.retrieval_benchmark            # table to stdout
    uv run python -m evals.retrieval_benchmark --json out.json

Each answerable query is labelled with the source file and a phrase that only
the correct chunk contains. Unanswerable queries test the relevance gate: the
right behaviour is to return *nothing*.

Metrics
- Hit@1   — the right chunk is ranked first
- Hit@k   — the right chunk is anywhere in the k results the LLM sees
- MRR     — mean reciprocal rank of the right chunk (0 if missing)
- Reject  — share of unanswerable queries for which nothing was returned
- ms/q    — mean latency per query
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass

from rag_agent.config import settings
from rag_agent.ingest import ensure_seeded
from rag_agent.retrieval import hybrid_search

MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")


@dataclass(frozen=True)
class Case:
    query: str
    source: str | None  # None = unanswerable from the corpus
    must_contain: str = ""
    kind: str = "semantic"  # semantic | keyword | unanswerable


CASES: list[Case] = [
    # --- keyword: exact identifiers, where embeddings tend to struggle -------
    Case("AUR-2005", "aurora_error_codes.md", "CROSS_TENANT_JOIN", "keyword"),
    Case("What does AUR-1007 mean?", "aurora_error_codes.md", "INGEST_BACKPRESSURE", "keyword"),
    Case("AUR-4001 error", "aurora_error_codes.md", "STORAGE_QUOTA_EXCEEDED", "keyword"),
    Case("query.cache_ttl_s", "aurora_config_reference.md", "query.cache_ttl_s", "keyword"),
    Case(
        "default for ingest.grpc_port", "aurora_config_reference.md", "ingest.grpc_port", "keyword"
    ),
    Case("AuroraQueryP99High", "aurora_oncall_runbook.md", "AuroraQueryP99High", "keyword"),
    Case("aurora-ctl backfill pause", "aurora_oncall_runbook.md", "backfill pause", "keyword"),
    Case("AUR-3004", "aurora_error_codes.md", "PAT_NOT_SUPPORTED", "keyword"),
    # --- semantic: paraphrases with little word overlap ----------------------
    Case(
        "How big can a single event be before it gets rejected?",
        "aurora_error_codes.md",
        "EVENT_TOO_LARGE",
    ),
    Case(
        "My dashboard request keeps getting cancelled after half a minute",
        "aurora_error_codes.md",
        "QUERY_TIMEOUT",
    ),
    Case(
        "Who gets paged if the main on-call engineer ignores the alert?",
        "aurora_oncall_runbook.md",
        "secondary on-call",
    ),
    Case(
        "How quickly must someone respond to a critical outage?",
        "aurora_oncall_runbook.md",
        "within 5 minutes",
    ),
    Case(
        "Can I log in with a personal token instead of certificates?",
        "aurora_error_codes.md",
        "personal access",
    ),
    Case(
        "How do I safely roll out a settings change?",
        "aurora_config_reference.md",
        "canary shard",
    ),
    Case(
        "When does the on-call week switch over?",
        "aurora_oncall_runbook.md",
        "every Monday",
    ),
    Case(
        "How long is aggregated daily data kept?",
        "aurora_config_reference.md",
        "retention.rollup_1d_days",
    ),
    # --- unanswerable: the gate should return nothing ------------------------
    Case("What is the capital of France?", None, kind="unanswerable"),
    Case("Give me a recipe for chocolate cake", None, kind="unanswerable"),
    Case("Who won the 2022 FIFA World Cup?", None, kind="unanswerable"),
    Case("How do I configure Kubernetes pod autoscaling?", None, kind="unanswerable"),
]


def _rank_of(case: Case, hits) -> int | None:
    for rank, h in enumerate(hits, start=1):
        if h.source == case.source and case.must_contain.lower() in h.text.lower():
            return rank
    return None


def evaluate(mode: str, k: int) -> dict:
    hit1, hitk, rr, latencies, rejected, per_kind = [], [], [], [], [], {}
    for case in CASES:
        t0 = time.perf_counter()
        hits = hybrid_search(case.query, k=k, mode=mode)
        latencies.append((time.perf_counter() - t0) * 1000)
        if case.source is None:
            rejected.append(len(hits) == 0)
            continue
        rank = _rank_of(case, hits)
        hit1.append(rank == 1)
        hitk.append(rank is not None)
        rr.append(1 / rank if rank else 0.0)
        per_kind.setdefault(case.kind, []).append(rank is not None)

    def pct(xs):
        return round(100 * sum(xs) / len(xs), 1) if xs else None

    return {
        "mode": mode,
        "hit@1": pct(hit1),
        f"hit@{k}": pct(hitk),
        "mrr": round(statistics.mean(rr), 3),
        "keyword_hit": pct(per_kind.get("keyword", [])),
        "semantic_hit": pct(per_kind.get("semantic", [])),
        "reject": pct(rejected),
        "ms/q": round(statistics.mean(latencies[1:] or latencies), 1),  # skip warm-up
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=settings.retriever_k)
    parser.add_argument("--json", help="also write results to this file")
    args = parser.parse_args()

    ensure_seeded()
    for mode in MODES:  # warm up models so latency isn't dominated by loading
        hybrid_search("warm up", k=1, mode=mode)

    rows = [evaluate(m, args.k) for m in MODES]
    cols = list(rows[0])
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    print(" | ".join(c.ljust(widths[c]) for c in cols))
    print("-+-".join("-" * widths[c] for c in cols))
    for r in rows:
        print(" | ".join(str(r[c]).ljust(widths[c]) for c in cols))
    n_ans = sum(c.source is not None for c in CASES)
    print(f"\n{n_ans} answerable + {len(CASES) - n_ans} unanswerable queries, k={args.k}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"k": args.k, "results": rows}, f, indent=2)


if __name__ == "__main__":
    main()
