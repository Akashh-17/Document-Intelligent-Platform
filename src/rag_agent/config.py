"""Application settings, loaded from environment / .env via pydantic-settings."""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Populate os.environ from .env so libraries that read the environment directly
# (langsmith tracing, langchain-tavily, langchain-groq) pick up the keys too —
# pydantic-settings alone only fills the Settings object, not os.environ.
load_dotenv(override=False)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- credentials ---
    # Optional at import time so health checks / tests can run without a key;
    # llms.py raises a clear error the first time a model is actually needed.
    groq_api_key: str | None = None
    tavily_api_key: str | None = None

    # --- LangSmith (tracing auto-enables when LANGSMITH_TRACING=true + key set) ---
    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str = "document-intelligence-platform"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    # --- LLMs (Groq) ---
    # Heavy and fast models live in separate Groq rate-limit buckets, so routing
    # cheap steps to the fast model keeps us inside the free-tier TPM budget.
    model_heavy: str = "openai/gpt-oss-120b"
    model_fast: str = "openai/gpt-oss-20b"
    reasoning_effort: str = "medium"  # gpt-oss: low | medium | high
    llm_temperature: float = 0.1
    llm_max_retries: int = 4  # exponential backoff on 429 / 5xx
    llm_timeout_s: float = 60.0
    # Which tier each agent role uses ("heavy" | "fast").
    planner_tier: str = "fast"  # routing + query rewriting: small structured task
    worker_tier: str = "fast"  # Document Analyst / Web Researcher tool loops
    writer_tier: str = "heavy"  # the cited answer the user reads
    verifier_tier: str = "fast"  # groundedness judging
    summarizer_tier: str = "fast"  # summarize_document tool
    # Hard caps that keep one question inside the free-tier token budget.
    worker_max_steps: int = 3  # tool-call rounds per research agent
    max_revisions: int = 1  # verifier-triggered rewrites of the answer
    web_fallback: bool = True  # documents found nothing -> ask the Web Researcher
    # Keep names specific to the library (e.g. an internal product name) out of
    # public web searches while documents are in play. See rag_agent/privacy.py.
    web_query_privacy: bool = True
    private_terms: str = ""  # extra comma-separated names to never send to web search

    # --- failure handling (Phase 3) ---
    model_fallback: bool = True  # on model failure, retry on the other tier
    turn_timeout_s: float = 180.0  # hard deadline for one chat turn
    # With no evidence from documents or web: False = say so (strict grounding);
    # True = answer from the model's general knowledge with a clear disclaimer.
    answer_without_evidence: bool = False
    near_miss_count: int = 3  # "closest passages" shown when nothing was relevant
    evidence_char_budget: int = 9000  # evidence text given to the writer (~2.3K tokens)

    # --- guardrails (Phase 4) ---
    guardrails_enabled: bool = True
    pii_masking: bool = True  # mask personal data in user input before any LLM/web call
    redact_contact_info: bool = False  # also redact emails/phones in answers
    prompt_guard_enabled: bool = True  # Llama Prompt Guard 2 on user input (Groq)
    prompt_guard_model: str = "meta-llama/llama-prompt-guard-2-86m"
    prompt_guard_threshold: float = 0.9
    rate_limit_per_minute: int = 20  # chat requests per client (0 = off)

    # --- embeddings (local, via fastembed / ONNX — Groq has no embeddings API) ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # --- storage / retrieval ---
    chroma_dir: str = "./chroma_db"
    # Collection name includes the embedding family so switching models never
    # mixes incompatible vector dimensions in one collection.
    chroma_collection: str = "documents_bge_small"
    sqlite_path: str = "./memory.sqlite"
    chunk_size: int = 700
    chunk_overlap: int = 100
    sample_docs_dir: str = "./data/sample_docs"
    # Index the sample Aurora docs on startup. Turn off to start with an empty
    # library and upload your own documents.
    seed_sample_docs: bool = True

    # --- hybrid retrieval (see rag_agent/retrieval) ---
    # dense | bm25 | hybrid | hybrid_rerank
    retrieval_mode: str = "hybrid_rerank"
    retriever_k: int = 4  # chunks handed to the LLM
    dense_k: int = 20  # candidates from vector search
    bm25_k: int = 20  # candidates from keyword search
    rrf_k: int = 60  # RRF damping constant (value from the original paper)
    rerank_enabled: bool = True
    # Chosen with evals/retrieval_benchmark.py on CPU: L-6 is ~4x faster than
    # L-12 (~0.8 s vs ~3 s per 10 passages) for a small MRR loss (0.797 vs 0.833).
    # Set RERANKER_MODEL=Xenova/ms-marco-MiniLM-L-12-v2 for max accuracy.
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    rerank_top_n: int = 10  # fused candidates scored by the cross-encoder
    # Relevance gates — chunks below these are dropped, so off-topic questions
    # get "nothing relevant" instead of k random chunks.
    min_rerank_score: float = 0.50  # sigmoid(cross-encoder logit): a confident vote
    min_relevance_score: float = 0.55  # cosine similarity (BGE-small)
    bm25_strong_rank: int = 2  # top-N keyword hits always pass (exact identifiers)

    # --- upload limits ---
    max_upload_mb: int = 20
    max_files_per_upload: int = 10

    @field_validator("groq_api_key", "tavily_api_key", "langsmith_api_key", mode="before")
    @classmethod
    def _placeholder_is_unset(cls, v):
        """Treat template values copied from .env.example (``tvly-...``) as unset.

        Otherwise a placeholder Tavily key would route every web search to Tavily
        and fail, instead of using the keyless DuckDuckGo fallback.
        """
        if isinstance(v, str) and (not v.strip() or v.strip().endswith("...")):
            return None
        return v

    @model_validator(mode="after")
    def _tracing_needs_a_key(self):
        if self.langsmith_tracing and not self.langsmith_api_key:
            self.langsmith_tracing = False
        return self

    @property
    def web_backend(self) -> str:
        return "tavily" if self.tavily_api_key else "duckduckgo"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

# Keep libraries that read os.environ directly consistent with the cleaned settings:
# no LangSmith tracing without a real key (it would log auth errors on every run),
# and no placeholder Tavily key in the environment.
if not settings.langsmith_tracing:
    os.environ["LANGSMITH_TRACING"] = "false"
if not settings.tavily_api_key:
    os.environ.pop("TAVILY_API_KEY", None)
