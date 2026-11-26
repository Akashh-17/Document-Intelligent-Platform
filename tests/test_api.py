"""HTTP layer: health, upload limits/sanitising, missing-key behaviour."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rag_agent.api import app
from rag_agent.config import settings
from rag_agent.vectorstore import delete_by_source


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # runs lifespan (seeds sample docs)
        yield c


def test_health_reports_groq_and_missing_key(client):
    body = client.get("/api/health").json()
    assert body["llm_provider"] == "groq"
    assert body["llm_configured"] is False
    assert body["documents_indexed"] > 0  # sample docs were seeded


def test_chat_without_key_returns_actionable_503(client):
    res = client.post("/api/chat/stream", json={"message": "hi"})
    assert res.status_code == 503
    assert "GROQ_API_KEY" in res.json()["detail"]


def test_upload_sanitises_name_and_reports_per_file(client):
    files = [
        ("files", ("../../sneaky.md", b"Sneaky doc about the Vega pipeline.", "text/markdown")),
        ("files", ("slides.pptx", b"PK\x03\x04", "application/octet-stream")),
    ]
    body = client.post("/api/ingest", files=files).json()
    by_name = {f["name"]: f for f in body["files"]}
    assert by_name["sneaky.md"]["chunks"] == 1 and by_name["sneaky.md"]["error"] is None
    assert "unsupported" in by_name["slides.pptx"]["error"]
    delete_by_source("sneaky.md")


def test_upload_too_many_files_is_rejected(client):
    files = [
        ("files", (f"f{i}.txt", b"x", "text/plain"))
        for i in range(settings.max_files_per_upload + 1)
    ]
    assert client.post("/api/ingest", files=files).status_code == 413


def test_oversized_file_is_skipped(client, monkeypatch):
    monkeypatch.setattr(settings, "max_upload_mb", 0)
    body = client.post("/api/ingest", files=[("files", ("big.txt", b"hello", "text/plain"))]).json()
    assert body["chunks_added"] == 0
    assert "larger than" in body["files"][0]["error"]


def test_document_library_list_and_delete(client):
    client.post(
        "/api/ingest",
        files=[("files", ("lib-test.md", b"# Lib test\nZebra facts.", "text/markdown"))],
    )
    docs = {d["source"]: d for d in client.get("/api/documents").json()}
    assert docs["lib-test.md"]["chunks"] == 1 and docs["lib-test.md"]["title"] == "Lib test"
    assert client.delete("/api/documents/lib-test.md").json()["chunks_removed"] == 1
    assert "lib-test.md" not in {d["source"] for d in client.get("/api/documents").json()}
    assert client.delete("/api/documents/lib-test.md").status_code == 404
