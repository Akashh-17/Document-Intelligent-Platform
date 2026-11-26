"""Evaluation scripts.

They index the sample Aurora documents into a *separate* Chroma collection, so
running an evaluation never adds sample files to your own document library.
(Set here, before any rag_agent module reads its settings.)
"""

import os

os.environ.setdefault("CHROMA_COLLECTION", "eval_aurora_samples")
