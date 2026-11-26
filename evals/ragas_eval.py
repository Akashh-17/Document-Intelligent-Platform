"""End-to-end RAG evaluation with RAGAS + behaviour checks.

Run:   uv run python -m evals.ragas_eval            # agent runs (cached) + RAGAS + report
       uv run python -m evals.ragas_eval --fresh    # re-run every question
       uv run python -m evals.ragas_eval --no-ragas # behaviour metrics only (cheap)

Two kinds of measurement:

1. **RAGAS** (answerable questions) — LLM-judged quality of the RAG pipeline:
   - faithfulness        claims in the answer supported by the retrieved contexts
   - answer_relevancy    the answer addresses the question (embedding similarity of
                         questions regenerated from the answer)
   - context_precision   relevant contexts are ranked above irrelevant ones
   - context_recall      the contexts contain what the reference answer needs
2. **Behaviour** (all questions, no LLM) — does the system do the right thing:
   - answerable:   answer rate, citation rate, verifier confidence, latency
   - unanswerable: abstention rate (honest "couldn't find it" instead of a guess)
   - adversarial:  block rate (injections blocked, harmful requests declined)

The RAGAS judge is a *different model family* (Qwen) from the system under test
(gpt-oss), which avoids a judge grading its own outputs favourably, and it runs on
its own Groq rate-limit bucket.

Free-tier friendly: agent runs (``agent_runs.json``) and RAGAS scores
(``ragas_scores.json``, keyed by a hash of each sample) are cached, so an
interrupted run (Groq's daily token cap) resumes where it stopped, and changing
one question only re-runs and re-scores that question.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from evals.golden_set import ADVERSARIAL, ALL, ANSWERABLE, UNANSWERABLE, GoldenItem
from rag_agent.config import settings

RESULTS = Path(__file__).parent / "results"
RUNS_FILE = RESULTS / "agent_runs.json"
REPORT_JSON = RESULTS / "ragas_report.json"
SCORES_FILE = RESULTS / "ragas_scores.json"
REPORT_MD = RESULTS / "RAGAS_REPORT.md"

JUDGE_MODEL = "qwen/qwen3.8-27b"
MAX_CONTEXTS = 5  # contexts per sample given to RAGAS (cost control)
CONTEXT_CHARS = 1500
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
_CITE = re.compile(r"\[(\d+)\]")


class DailyLimitReached(RuntimeError):
    pass


# --- step 1: run the agents ------------------------------------------------------------


def _load_runs() -> dict:
    return json.loads(RUNS_FILE.read_text(encoding="utf-8")) if RUNS_FILE.exists() else {}


def _save_runs(runs: dict) -> None:
    RESULTS.mkdir(exist_ok=True)
    RUNS_FILE.write_text(json.dumps(runs, indent=2, ensure_ascii=False), encoding="utf-8")


def _is_daily_limit(exc: Exception) -> bool:
    text = str(exc).lower()
    return "per day" in text or "tokens per day" in text or "(tpd)" in text or "(rpd)" in text


def run_agents(items: list[GoldenItem], fresh: bool, pause_s: float) -> dict:
    from rag_agent.agent import build_agent, run_research
    from rag_agent.ingest import ensure_seeded

    ensure_seeded()
    graph = build_agent()
    runs = {} if fresh else _load_runs()
    todo = [it for it in items if it.id not in runs or runs[it.id].get("error")]
    print(f"Agent runs: {len(items) - len(todo)} cached, {len(todo)} to run")

    for n, item in enumerate(todo, 1):
        print(f"  [{n}/{len(todo)}] {item.id}: {item.question[:70]}")
        t0 = time.perf_counter()
        try:
            r = run_research(item.question, graph=graph)
            runs[item.id] = {
                "question": item.question,
                "kind": item.kind,
                "answer": r["answer"],
                "contexts": r["contexts"],
                "outcome": r["outcome"],
                "route": (r["plan"] or {}).get("route"),
                "verification": r["verification"],
                "latency_s": round(time.perf_counter() - t0, 1),
            }
        except Exception as exc:  # noqa: BLE001 — record and keep going
            runs[item.id] = {"question": item.question, "kind": item.kind, "error": repr(exc)[:300]}
            _save_runs(runs)
            if _is_daily_limit(exc):
                raise DailyLimitReached(str(exc)[:300]) from exc
        _save_runs(runs)
        if pause_s and n < len(todo):
            time.sleep(pause_s)  # stay under the per-minute token limit
    return runs


# --- step 2: RAGAS ----------------------------------------------------------------------


def _ragas_contexts(run: dict) -> list[str]:
    """Contexts the answer actually cited first, then the rest — capped for cost."""
    contexts = run.get("contexts") or []
    cited = [int(n) for n in _CITE.findall(run.get("answer", "")) if 0 < int(n) <= len(contexts)]
    order = list(dict.fromkeys(cited)) + [i for i in range(1, len(contexts) + 1) if i not in cited]
    return [contexts[i - 1][:CONTEXT_CHARS] for i in order[:MAX_CONTEXTS]]


def run_ragas(runs: dict) -> dict[str, dict]:
    from langchain_groq import ChatGroq
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )
    from ragas.run_config import RunConfig

    from rag_agent.embeddings import get_embeddings

    items = [
        it
        for it in ANSWERABLE
        if it.id in runs and runs[it.id].get("outcome") in ("answered", "extractive")
    ]
    # Reuse scores for samples whose inputs haven't changed (judge calls are slow
    # and count against the free-tier quota); score only new / changed ones.
    cache = _load_score_cache()
    scores: dict[str, dict] = {}
    todo = []
    for it in items:
        key = _sample_hash(it, runs[it.id])
        if cache.get(it.id, {}).get("hash") == key:
            scores[it.id] = cache[it.id]["scores"]
        else:
            todo.append(it)
    print(f"RAGAS: {len(scores)} cached, {len(todo)} to score")
    items = todo
    if not items:
        return scores
    samples = [
        SingleTurnSample(
            user_input=it.question,
            response=runs[it.id]["answer"],
            retrieved_contexts=_ragas_contexts(runs[it.id]) or ["(no context)"],
            reference=it.reference,
        )
        for it in items
    ]
    judge = ChatGroq(
        model=JUDGE_MODEL,
        api_key=settings.groq_api_key,
        temperature=0,
        max_retries=8,
        reasoning_effort="none",
    )
    print(f"RAGAS: scoring {len(samples)} samples with judge {JUDGE_MODEL} ...")
    result = evaluate(
        EvaluationDataset(samples),
        metrics=[
            Faithfulness(),
            ResponseRelevancy(strictness=1),
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
        ],
        llm=LangchainLLMWrapper(judge),
        embeddings=LangchainEmbeddingsWrapper(get_embeddings()),
        run_config=RunConfig(max_workers=2, max_retries=8, max_wait=90, timeout=240),
        show_progress=True,
    )
    df = result.to_pandas()
    rename = {
        "llm_context_precision_with_reference": "context_precision",
        "answer_relevancy": "answer_relevancy",
    }
    df = df.rename(columns=rename)
    for it, (_, row) in zip(items, df.iterrows(), strict=True):
        scores[it.id] = {m: _num(row.get(m)) for m in METRIC_NAMES}
        cache[it.id] = {"hash": _sample_hash(it, runs[it.id]), "scores": scores[it.id]}
    SCORES_FILE.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    return scores


def _sample_hash(item: GoldenItem, run: dict) -> str:
    payload = json.dumps(
        [item.question, item.reference, run.get("answer"), _ragas_contexts(run), JUDGE_MODEL]
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _load_score_cache() -> dict:
    if SCORES_FILE.exists():
        return json.loads(SCORES_FILE.read_text(encoding="utf-8"))
    return {}


def _num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else round(f, 4)


# --- step 3: behaviour metrics + report ------------------------------------------------------


def _mean(xs: list[float | None]) -> float | None:
    vals = [x for x in xs if x is not None]
    return round(statistics.mean(vals), 3) if vals else None


def _pct(flags: list[bool]) -> float | None:
    return round(100 * sum(flags) / len(flags), 1) if flags else None


def behaviour(runs: dict) -> dict:
    def ok(items):
        return [runs[i.id] for i in items if i.id in runs and not runs[i.id].get("error")]

    ans, una, adv = ok(ANSWERABLE), ok(UNANSWERABLE), ok(ADVERSARIAL)
    lat = sorted(r["latency_s"] for r in ans)
    return {
        "answerable": {
            "n": len(ans),
            "answer_rate_pct": _pct([r["outcome"] == "answered" for r in ans]),
            "citation_rate_pct": _pct([bool(_CITE.search(r["answer"])) for r in ans]),
            "verified_high_pct": _pct(
                [(r.get("verification") or {}).get("confidence") == "high" for r in ans]
            ),
            "mean_verifier_score": _mean([(r.get("verification") or {}).get("score") for r in ans]),
            "revision_rate_pct": _pct(
                [((r.get("verification") or {}).get("revisions") or 0) > 0 for r in ans]
            ),
            "latency_p50_s": lat[len(lat) // 2] if lat else None,
            "latency_max_s": lat[-1] if lat else None,
        },
        "unanswerable": {
            "n": len(una),
            "abstention_rate_pct": _pct([r["outcome"] in ("no_answer", "declined") for r in una]),
        },
        "adversarial": {
            "n": len(adv),
            "block_rate_pct": _pct([r["outcome"] in ("blocked", "declined") for r in adv]),
        },
        "errors": [i.id for i in ALL if runs.get(i.id, {}).get("error")],
    }


def write_report(runs: dict, scores: dict, beh: dict) -> None:
    aggregate = {m: _mean([s.get(m) for s in scores.values()]) for m in METRIC_NAMES}
    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "system": {
            "model_heavy": settings.model_heavy,
            "model_fast": settings.model_fast,
            "retrieval_mode": settings.retrieval_mode,
            "reranker": settings.reranker_model,
            "embedding_model": settings.embedding_model,
            "judge": JUDGE_MODEL,
        },
        "ragas": {"n": len(scores), "mean": aggregate, "per_question": scores},
        "behaviour": beh,
    }
    RESULTS.mkdir(exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")

    def f(v, pct=False):
        return "—" if v is None else (f"{v:.1f}%" if pct else f"{v:.3f}")

    a, u, d = beh["answerable"], beh["unanswerable"], beh["adversarial"]
    lines = [
        "# RAG Evaluation Report",
        "",
        f"_Generated {report['generated_at']} · system: `{settings.model_heavy}` (writer) + "
        f"`{settings.model_fast}` (agents) · retrieval: `{settings.retrieval_mode}` · "
        f"RAGAS judge: `{JUDGE_MODEL}`_",
        "",
        f"## RAGAS (answerable questions, n={len(scores)})",
        "",
        "| Metric | Score | What it measures |",
        "|---|---|---|",
        f"| Faithfulness | {f(aggregate['faithfulness'])} | Answer claims supported by the retrieved context |",
        f"| Answer relevancy | {f(aggregate['answer_relevancy'])} | Answer addresses the question |",
        f"| Context precision | {f(aggregate['context_precision'])} | Relevant context ranked above irrelevant |",
        f"| Context recall | {f(aggregate['context_recall'])} | Context covers what the reference answer needs |",
        "",
        "## Behaviour",
        "",
        "| Check | Result |",
        "|---|---|",
        f"| Answerable: answered | {f(a['answer_rate_pct'], True)} (n={a['n']}) |",
        f"| Answerable: answers with citations | {f(a['citation_rate_pct'], True)} |",
        f'| Answerable: verifier confidence "high" | {f(a["verified_high_pct"], True)} |',
        f"| Answerable: verifier groundedness score (mean) | {f(a['mean_verifier_score'])} |",
        f"| Answerable: needed a self-correction revision | {f(a['revision_rate_pct'], True)} |",
        f"| Unanswerable: honest abstention (no guessing) | {f(u['abstention_rate_pct'], True)} (n={u['n']}) |",
        f"| Adversarial: blocked or declined | {f(d['block_rate_pct'], True)} (n={d['n']}) |",
        f"| Latency p50 / max (answerable) | {a['latency_p50_s']}s / {a['latency_max_s']}s |",
        "",
        "## Per question",
        "",
        "| Question | Kind | Outcome | Route | Faith. | Relev. | Prec. | Recall | Latency |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for it in ALL:
        r = runs.get(it.id, {})
        s = scores.get(it.id, {})
        outcome = "error" if r.get("error") else r.get("outcome", "—")
        lines.append(
            f"| {it.question[:60]} | {it.kind} | {outcome} | {r.get('route', '—')} | "
            + " | ".join(f(s.get(m)) for m in METRIC_NAMES)
            + f" | {r.get('latency_s', '—')}s |"
        )
    if beh["errors"]:
        lines += ["", f"Errors (not scored): {', '.join(beh['errors'])}"]
    lines += [
        "",
        "Notes: RAGAS scores use an LLM judge and vary run to run by a few points; the "
        "corpus is a small fictional product (Aurora), so treat these as a regression "
        "baseline, not a benchmark claim.",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"\nWrote {REPORT_MD.relative_to(Path.cwd()) if REPORT_MD.is_relative_to(Path.cwd()) else REPORT_MD}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--fresh", action="store_true", help="re-run every question")
    parser.add_argument(
        "--rerun", nargs="*", default=[], help="ids of questions to re-run (others stay cached)"
    )
    parser.add_argument("--no-ragas", action="store_true", help="skip RAGAS scoring")
    parser.add_argument("--pause", type=float, default=20.0, help="seconds between questions")
    args = parser.parse_args()

    if not settings.groq_api_key:
        sys.exit("GROQ_API_KEY is not set (see .env.example).")
    if args.rerun:
        cached = _load_runs()
        for qid in args.rerun:
            cached.pop(qid, None)
        _save_runs(cached)
    try:
        runs = run_agents(ALL, fresh=args.fresh, pause_s=args.pause)
    except DailyLimitReached as exc:
        print(f"\nGroq daily limit reached — progress saved; re-run later to resume.\n{exc}")
        runs = _load_runs()
    scores = {} if args.no_ragas else run_ragas(runs)
    beh = behaviour(runs)
    write_report(runs, scores, beh)
    print(
        json.dumps(
            {
                "ragas_mean": {m: _mean([s.get(m) for s in scores.values()]) for m in METRIC_NAMES},
                **beh,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
