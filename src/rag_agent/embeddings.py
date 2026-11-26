"""Local embeddings via fastembed (ONNX runtime, no PyTorch, no API key).

Groq does not offer an embeddings endpoint, so embeddings run locally. The
default model, BAAI/bge-small-en-v1.5, produces 384-dim vectors and is a strong
small retrieval model. The model weights (~130 MB) download once on first use.

BGE models are trained with an instruction prefix for *queries* (not passages);
fastembed's ``query_embed`` applies it for us, which measurably improves recall.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.embeddings import Embeddings

from rag_agent.config import settings


class FastEmbedEmbeddings(Embeddings):
    """Minimal LangChain ``Embeddings`` adapter around ``fastembed.TextEmbedding``."""

    def __init__(self, model_name: str, batch_size: int = 64) -> None:
        from fastembed import TextEmbedding

        self.model_name = model_name
        self.batch_size = batch_size
        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [
            vec.tolist() for vec in self._model.passage_embed(texts, batch_size=self.batch_size)
        ]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed(text))).tolist()


@lru_cache
def get_embeddings() -> FastEmbedEmbeddings:
    return FastEmbedEmbeddings(settings.embedding_model)
