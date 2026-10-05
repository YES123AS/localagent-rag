"""Metadata enrichment and checksum-based document version decisions."""

import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def enrich_document_metadata(
    documents: Iterable[Any],
    filename: str,
    *,
    document_id: str | None = None,
    upload_time: str | None = None,
    checksum: str | None = None,
    version: int = 1,
) -> list[Any]:
    document_id = document_id or uuid.uuid4().hex
    upload_time = upload_time or datetime.now(timezone.utc).isoformat()
    suffix = Path(filename).suffix.lower().lstrip(".") or "unknown"
    enriched = list(documents)
    for index, document in enumerate(enriched):
        metadata = document.metadata
        metadata.update(
            {
                "document_id": document_id,
                "filename": filename,
                "source": filename,
                "file_type": suffix,
                "chunk_index": metadata.get("chunk_index", index),
                "chunk_id": metadata.get("chunk_id", f"{document_id}:{index}"),
                "upload_time": upload_time,
                "checksum": checksum,
                "version": max(1, int(version)),
            }
        )
        if metadata.get("page") is not None:
            metadata["page_number"] = int(metadata["page"]) + 1
    return enriched


def document_checksum(content: bytes) -> str:
    """Return a stable content checksum before parsing or chunking."""
    return hashlib.sha256(content).hexdigest()


def ingestion_decision(
    existing_documents: Iterable[dict[str, Any]], filename: str, checksum: str
) -> dict[str, Any]:
    """Classify an upload as new, duplicate, or a replacement version."""
    same_name = [item for item in existing_documents if str(item.get("filename")) == filename]
    duplicate = next((item for item in same_name if item.get("checksum") == checksum), None)
    if duplicate:
        return {"action": "duplicate", "version": int(duplicate.get("version") or 1), "replace_ids": []}
    versions = [int(item.get("version") or 1) for item in same_name]
    return {
        "action": "replace" if same_name else "new",
        "version": max(versions, default=0) + 1,
        "replace_ids": [str(item["document_id"]) for item in same_name],
    }
