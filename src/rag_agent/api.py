"""FastAPI application exposing the LangGraph agent over HTTP + SSE."""

from __future__ import annotations

import logging
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import PurePath

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import run_in_threadpool

from rag_agent.agent import build_agent
from rag_agent.config import settings
from rag_agent.guardrails import RateLimiter
from rag_agent.ingest import ensure_seeded, ingest_files
from rag_agent.schemas import (
    ChatRequest,
    DocumentInfo,
    FeedbackRequest,
    HealthResponse,
    IngestFileResult,
    IngestResponse,
)
from rag_agent.streaming import agent_event_stream
from rag_agent.vectorstore import collection_count, delete_by_source, source_stats

log = logging.getLogger("rag_agent")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Auto-seed the vector store with the sample docs on first boot. Embedding
    # is CPU-bound, so keep it off the event loop.
    seeded = await run_in_threadpool(ensure_seeded) if settings.seed_sample_docs else 0
    if seeded:
        log.info("Seeded vector store with %d sample chunks", seeded)
    if not settings.groq_api_key:
        log.warning("GROQ_API_KEY is not set — chat requests will fail until it is.")

    # Open the async checkpointer for the whole app lifetime.
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    async with AsyncSqliteSaver.from_conn_string(settings.sqlite_path) as saver:
        app.state.checkpointer = saver
        app.state.agent = None  # built lazily so a missing key doesn't block boot
        yield


app = FastAPI(title="Document Intelligence Platform", version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo: tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_agent(request: Request):
    if request.app.state.agent is None:
        request.app.state.agent = build_agent(checkpointer=request.app.state.checkpointer)
    return request.app.state.agent


# Each chat turn costs several LLM calls on a shared quota: cap requests per client.
_rate_limiter = RateLimiter(settings.rate_limit_per_minute)

_SAFE_NAME = re.compile(r"[^\w.\- ()]+")


def safe_filename(raw: str | None) -> str:
    """Strip any directory part and odd characters from a client-supplied name."""
    name = PurePath((raw or "").replace("\\", "/")).name
    name = _SAFE_NAME.sub("_", name).strip(" .")
    return name[:120] or f"upload-{uuid.uuid4().hex[:8]}.txt"


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        llm_provider="groq",
        llm_configured=bool(settings.groq_api_key),
        model_fast=settings.model_fast,
        model_heavy=settings.model_heavy,
        embedding_model=settings.embedding_model,
        web_backend=settings.web_backend,
        langsmith_tracing=settings.langsmith_tracing,
        documents_indexed=collection_count(),
    )


@app.post("/api/ingest", response_model=IngestResponse)
async def ingest(files: list[UploadFile] = File(...)) -> IngestResponse:
    if len(files) > settings.max_files_per_upload:
        raise HTTPException(
            status_code=413,
            detail=f"Too many files: max {settings.max_files_per_upload} per upload.",
        )

    max_bytes = settings.max_upload_mb * 1024 * 1024
    payload: list[tuple[str, bytes]] = []
    rejected: list[IngestFileResult] = []
    for f in files:
        name = safe_filename(f.filename)
        raw = await f.read(max_bytes + 1)
        if len(raw) > max_bytes:
            rejected.append(
                IngestFileResult(
                    name=name, chunks=0, error=f"{name}: larger than {settings.max_upload_mb} MB."
                )
            )
            continue
        payload.append((name, raw))

    # Parsing + embedding is CPU-bound: run it in a worker thread so live chat
    # streams keep flowing while a large upload is being indexed.
    report = await run_in_threadpool(ingest_files, payload)
    results = [
        IngestFileResult(
            name=r.name, chunks=r.chunks, replaced=r.replaced, error=r.error, warnings=r.warnings
        )
        for r in report.files
    ] + rejected
    return IngestResponse(chunks_added=report.chunks_added, files=results)


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest, request: Request) -> EventSourceResponse:
    client = request.client.host if request.client else "unknown"
    wait = _rate_limiter.check(client)
    if wait is not None:
        raise HTTPException(
            status_code=429,
            detail=f"Too many requests — please wait {wait:.0f}s and try again.",
            headers={"Retry-After": str(int(wait) + 1)},
        )
    thread_id = req.thread_id or str(uuid.uuid4())
    try:
        agent = _get_agent(request)
    except Exception as exc:  # e.g. missing GROQ_API_KEY
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    generator = agent_event_stream(
        agent,
        message=req.message,
        thread_id=thread_id,
        is_disconnected=request.is_disconnected,
    )
    return EventSourceResponse(
        generator,
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/documents", response_model=list[DocumentInfo])
async def list_documents() -> list[DocumentInfo]:
    """The knowledge base, one entry per uploaded file."""
    stats = await run_in_threadpool(source_stats)
    return [DocumentInfo(**s) for s in stats]


@app.delete("/api/documents/{name}")
async def delete_document(name: str) -> dict:
    """Remove a file and all its chunks from the knowledge base."""
    removed = await run_in_threadpool(delete_by_source, name)
    if not removed:
        raise HTTPException(status_code=404, detail=f"No document named '{name}'.")
    return {"deleted": name, "chunks_removed": removed}


@app.post("/api/feedback")
async def feedback(req: FeedbackRequest) -> dict:
    """Forward end-user thumbs up/down into LangSmith."""
    if not settings.langsmith_api_key:
        return {"ok": False, "error": "LangSmith is not configured; feedback not recorded."}
    try:
        from langsmith import Client

        Client().create_feedback(
            run_id=req.run_id, key=req.key, score=req.score, comment=req.comment
        )
        return {"ok": True}
    except Exception as exc:
        log.warning("feedback failed: %s", exc)
        return {"ok": False, "error": str(exc)}
