"""Comparative (pairwise) experiment: heavy vs fast model for the answer writer.

Runs the full multi-agent graph twice — once with the Fact Verifier writing on
the heavy model, once on the fast model — then an LLM judge picks the better
answer per example. The result is a side-by-side comparison in LangSmith.

Run:  uv run python -m evals.run_pairwise
"""

from __future__ import annotations

from langsmith import Client
from langsmith.evaluation import evaluate_comparative

from evals.create_dataset import DATASET_NAME
from rag_agent.agent import build_agent, run_research
from rag_agent.config import settings
from rag_agent.llms import build_chat_model, override_models

HEAVY, FAST = settings.model_heavy, settings.model_fast


def _make_target(writer_model: str):
    writer = build_chat_model(writer_model, streaming=False)

    def target(inputs: dict) -> dict:
        with override_models({"writer": writer}):
            result = run_research(inputs["question"], graph=build_agent())
        return {"answer": result["answer"], "context": result["contexts"]}

    return target


_judge = build_chat_model(FAST, streaming=False)


def ranked_preference(runs: list, example) -> dict:
    """LLM judge picks the better of two answers; returns per-run scores."""
    question = example.inputs.get("question", "")
    a = (runs[0].outputs or {}).get("answer", "")
    b = (runs[1].outputs or {}).get("answer", "")
    reference = (example.outputs or {}).get("answer", "")

    prompt = (
        "You are judging two assistant answers to the same question.\n"
        f"Question: {question}\n"
        f"Reference answer: {reference}\n\n"
        f"Answer A:\n{a}\n\nAnswer B:\n{b}\n\n"
        "Which answer is more correct, grounded, and well-cited? "
        "Reply with exactly one character: A or B."
    )
    verdict = _judge.invoke(prompt).text.strip().upper()
    a_wins = verdict.startswith("A")
    return {
        "key": "ranked_preference",
        "scores": {runs[0].id: int(a_wins), runs[1].id: int(not a_wins)},
    }


def main() -> None:
    client = Client()

    heavy = client.evaluate(
        _make_target(HEAVY),
        data=DATASET_NAME,
        experiment_prefix="rag-agent-heavy",
        metadata={"model": HEAVY},
        max_concurrency=1,
    )
    fast = client.evaluate(
        _make_target(FAST),
        data=DATASET_NAME,
        experiment_prefix="rag-agent-fast",
        metadata={"model": FAST},
        max_concurrency=1,
    )

    comparison = evaluate_comparative(
        (heavy.experiment_name, fast.experiment_name),
        evaluators=[ranked_preference],
        randomize_order=True,
        max_concurrency=1,
    )
    print(comparison)


if __name__ == "__main__":
    main()
