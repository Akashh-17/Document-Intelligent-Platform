"""Pydantic request/response models for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    thread_id: str | None = Field(default=None, max_length=100)


class FeedbackRequest(BaseModel):
    run_id: str
    score: float = Field(..., ge=0, le=1)
    comment: str | None = Field(default=None, max_length=2000)
    key: str = "user_score"


class IngestFileResult(BaseModel):
    name: str
    chunks: int
    replaced: int = 0
    error: str | None = None


class IngestResponse(BaseModel):
    chunks_added: int
    files: list[IngestFileResult]


class HealthResponse(BaseModel):
    status: str
    llm_provider: str
    llm_configured: bool
    model_fast: str
    model_heavy: str
    embedding_model: str
    web_backend: str
    langsmith_tracing: bool
    documents_indexed: int
