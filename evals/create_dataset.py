"""Create / refresh the LangSmith evaluation dataset.

Run:  uv run python -m evals.create_dataset
"""

from __future__ import annotations

from langsmith import Client

from evals.golden_set import ANSWERABLE

# Loads .env into os.environ so LANGSMITH_* is available.
from rag_agent.config import settings  # noqa: F401

DATASET_NAME = "document-intelligence-qa"

# The answerable questions of the shared golden set (same questions the RAGAS
# evaluation uses), so LangSmith experiments and RAGAS reports are comparable.
EXAMPLES: list[dict] = [
    {"inputs": {"question": it.question}, "outputs": {"answer": it.reference}} for it in ANSWERABLE
]


def main() -> None:
    client = Client()

    if client.has_dataset(dataset_name=DATASET_NAME):
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
        print(f"Dataset already exists: {dataset.id}")
    else:
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description="Q/reference-answer pairs for the agentic RAG + web-search demo.",
        )
        print(f"Created dataset: {dataset.id}")

    # Only add questions that aren't already in the dataset, so re-running this
    # script is idempotent instead of duplicating every example.
    existing = {
        (ex.inputs or {}).get("question") for ex in client.list_examples(dataset_id=dataset.id)
    }
    new = [ex for ex in EXAMPLES if ex["inputs"]["question"] not in existing]
    if new:
        client.create_examples(dataset_id=dataset.id, examples=new)
    print(f"Added {len(new)} new examples ({len(existing)} already present).")


if __name__ == "__main__":
    main()
