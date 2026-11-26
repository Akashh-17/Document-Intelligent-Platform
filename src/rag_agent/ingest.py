"""Document ingestion: load -> split -> embed -> persist to Chroma.

Design notes
- Every chunk gets a deterministic ID (hash of source + chunk index + content),
  and re-ingesting a file first deletes that file's old chunks. Uploading the
  same file twice is therefore idempotent, and uploading an edited version
  replaces the old one instead of leaving stale duplicates.
- Each chunk records ``source`` (the clean file name), ``page`` (PDFs) and
  ``chunk_index`` so later tools can cite precisely and fetch neighbours.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_agent.config import settings
from rag_agent.vectorstore import add_chunks, delete_by_source, list_sources

TEXT_EXTS = {".txt", ".md", ".markdown"}
PDF_EXTS = {".pdf"}
SUPPORTED_EXTS = TEXT_EXTS | PDF_EXTS


class IngestError(ValueError):
    """A user-facing problem with an uploaded file (bad type, empty, unreadable)."""


@dataclass
class FileResult:
    name: str
    chunks: int = 0
    replaced: int = 0
    error: str | None = None


@dataclass
class IngestReport:
    files: list[FileResult] = field(default_factory=list)

    @property
    def chunks_added(self) -> int:
        return sum(f.chunks for f in self.files)


_splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
    add_start_index=True,
)


# --- loaders -----------------------------------------------------------------


def _decode_text(raw: bytes, name: str) -> str:
    if b"\x00" in raw[:8192]:
        raise IngestError(f"{name}: looks like a binary file, not plain text.")
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise IngestError(f"{name}: could not decode text.")  # pragma: no cover


def _load_pdf(raw: bytes, name: str) -> list[Document]:
    import io

    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(raw))
    except (PdfReadError, ValueError) as exc:
        raise IngestError(f"{name}: not a readable PDF ({exc}).") from exc
    if reader.is_encrypted:
        raise IngestError(f"{name}: PDF is password-protected.")

    docs = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            docs.append(Document(page_content=text, metadata={"source": name, "page": i + 1}))
    if not docs:
        raise IngestError(f"{name}: no extractable text (it may be a scanned/image-only PDF).")
    return docs


def load_bytes(raw: bytes, name: str) -> list[Document]:
    """Turn raw file bytes into Documents. ``name`` is the clean display name."""
    ext = Path(name).suffix.lower()
    if ext not in SUPPORTED_EXTS:
        raise IngestError(
            f"{name}: unsupported file type '{ext or 'none'}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTS))}."
        )
    if not raw.strip():
        raise IngestError(f"{name}: file is empty.")
    if ext in PDF_EXTS:
        return _load_pdf(raw, name)
    text = _decode_text(raw, name).strip()
    if not text:
        raise IngestError(f"{name}: file is empty.")
    return [Document(page_content=text, metadata={"source": name})]


# --- pipeline ----------------------------------------------------------------


def _chunk_id(source: str, index: int, content: str) -> str:
    h = hashlib.sha256(f"{source}\x1f{index}\x1f{content}".encode()).hexdigest()
    return h[:32]


def index_documents(docs: list[Document], source: str) -> FileResult:
    """Split, (re)place and embed one file's documents."""
    chunks = _splitter.split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata["source"] = source
        c.metadata["chunk_index"] = i
    replaced = delete_by_source(source)
    if chunks:
        add_chunks(chunks, ids=[_chunk_id(source, i, c.page_content) for i, c in enumerate(chunks)])
    return FileResult(name=source, chunks=len(chunks), replaced=replaced)


def ingest_files(files: list[tuple[str, bytes]]) -> IngestReport:
    """Ingest ``(display_name, raw_bytes)`` pairs. Bad files are reported, not raised."""
    report = IngestReport()
    for name, raw in files:
        try:
            docs = load_bytes(raw, name)
            report.files.append(index_documents(docs, name))
        except IngestError as exc:
            report.files.append(FileResult(name=name, error=str(exc)))
    return report


def _sample_files(directory: str | None = None) -> list[Path]:
    root = Path(directory or settings.sample_docs_dir)
    if not root.exists():
        return []
    return [
        p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    ]


def ingest_directory(directory: str | None = None) -> int:
    files = [(p.name, p.read_bytes()) for p in _sample_files(directory)]
    return ingest_files(files).chunks_added


def ensure_seeded() -> int:
    """Index any sample doc that isn't in the store yet (first boot, or new samples)."""
    indexed = set(list_sources())
    missing = [(p.name, p.read_bytes()) for p in _sample_files() if p.name not in indexed]
    return ingest_files(missing).chunks_added if missing else 0
