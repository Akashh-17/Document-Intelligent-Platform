"""Ingestion: idempotency, clean source names, user-facing errors, safe filenames."""

from __future__ import annotations

import pytest

from rag_agent.api import safe_filename
from rag_agent.ingest import ingest_files
from rag_agent.vectorstore import collection_count, delete_by_source, get_vectorstore

TEXT = (
    "Orion is a billing service. Invoices are generated nightly at 02:00 UTC. "
    "Refunds over 500 USD require manager approval.\n\n" * 3
)


@pytest.fixture(autouse=True)
def _clean():
    delete_by_source("orion.md")
    yield
    delete_by_source("orion.md")


def test_reingesting_same_file_does_not_duplicate_chunks():
    before = collection_count()
    first = ingest_files([("orion.md", TEXT.encode())])
    after_first = collection_count()
    second = ingest_files([("orion.md", TEXT.encode())])

    assert first.chunks_added > 0
    assert after_first == before + first.chunks_added
    assert collection_count() == after_first  # replaced, not appended
    assert second.files[0].replaced == first.chunks_added


def test_chunks_carry_clean_source_and_index():
    ingest_files([("orion.md", TEXT.encode())])
    got = get_vectorstore()._collection.get(where={"source": "orion.md"}, include=["metadatas"])
    metas = got["metadatas"]
    assert metas and all(m["source"] == "orion.md" for m in metas)
    assert sorted(m["chunk_index"] for m in metas) == list(range(len(metas)))


@pytest.mark.parametrize(
    ("name", "raw", "expected"),
    [
        ("report.docx", b"PK\x03\x04", "unsupported file type"),
        ("empty.txt", b"   \n", "empty"),
        ("blob.txt", b"abc\x00def", "binary"),
        ("broken.pdf", b"definitely not a pdf", "not a readable PDF"),
    ],
)
def test_bad_files_are_reported_not_raised(name, raw, expected):
    report = ingest_files([(name, raw)])
    assert report.chunks_added == 0
    assert expected in (report.files[0].error or "")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("../../etc/passwd", "passwd"),
        ("..\\..\\windows\\evil.md", "evil.md"),
        ("my notes (v2).md", "my notes (v2).md"),
        ("weird<>|name.txt", "weird_name.txt"),
    ],
)
def test_safe_filename_strips_paths(raw, expected):
    assert safe_filename(raw) == expected


def test_safe_filename_fallback_for_empty():
    assert safe_filename("").startswith("upload-")
