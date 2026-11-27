"""Original files behind the indexed documents.

The vector store keeps only text chunks. To show a cited passage *in its
document* (the PDF page, highlighted), the UI needs the original file, so
uploads are kept on disk next to the index. Sample documents are served from
the sample directory they were indexed from.

File names are already sanitized at upload (``api.safe_filename``); every
lookup still resolves the path and checks it stays inside its directory, so a
crafted name can never read files elsewhere on the server.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from rag_agent.config import settings

log = logging.getLogger(__name__)

MEDIA_TYPES = {
    ".pdf": "application/pdf",
    ".md": "text/markdown; charset=utf-8",
    ".markdown": "text/markdown; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
}
KINDS = {".pdf": "pdf", ".md": "markdown", ".markdown": "markdown", ".txt": "text"}


def _roots() -> list[Path]:
    return [Path(settings.uploads_dir), Path(settings.sample_docs_dir)]


def _inside(root: Path, name: str) -> Path | None:
    root = root.resolve()
    path = (root / name).resolve()
    return path if path.parent == root and path.is_file() else None


def path_for(name: str) -> Path | None:
    """The stored original of ``name`` (uploads first, then samples), if any."""
    for root in _roots():
        if root.exists() and (path := _inside(root, name)) is not None:
            return path
    return None


def store(name: str, raw: bytes) -> None:
    """Keep an uploaded file (replacing an older version with the same name)."""
    root = Path(settings.uploads_dir)
    root.mkdir(parents=True, exist_ok=True)
    target = (root.resolve() / name).resolve()
    if target.parent != root.resolve():
        raise ValueError(f"unsafe file name: {name!r}")
    target.write_bytes(raw)


def remove(name: str) -> None:
    """Delete a stored upload (sample files are left alone)."""
    path = (
        _inside(Path(settings.uploads_dir), name) if Path(settings.uploads_dir).exists() else None
    )
    if path is not None:
        path.unlink(missing_ok=True)


def file_info(name: str) -> dict:
    """Details shown in the library: type, size, when it was added, preview available."""
    suffix = Path(name).suffix.lower()
    info: dict = {"kind": KINDS.get(suffix, "text"), "has_file": False}
    path = path_for(name)
    if path is not None:
        stat = path.stat()
        info.update(
            has_file=True,
            size_bytes=stat.st_size,
            added_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
        )
    return info


def media_type(name: str) -> str:
    return MEDIA_TYPES.get(Path(name).suffix.lower(), "application/octet-stream")
