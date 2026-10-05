"""Qdrant-backed document inventory and true vector deletion."""

from collections import OrderedDict
from typing import Any

from qdrant_client import models


def list_documents(client: Any, collection_name: str, *, page_size: int = 256) -> list[dict[str, Any]]:
    if not client.collection_exists(collection_name):
        return []
    documents: OrderedDict[str, dict[str, Any]] = OrderedDict()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            limit=page_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            metadata = payload.get("metadata", payload)
            name = str(metadata.get("filename") or metadata.get("source") or "未知文档")
            document_id = str(metadata.get("document_id") or name)
            record = documents.setdefault(
                document_id,
                {
                    "document_id": document_id,
                    "filename": name,
                    "file_type": metadata.get("file_type"),
                    "upload_time": metadata.get("upload_time"),
                    "checksum": metadata.get("checksum"),
                    "version": metadata.get("version", 1),
                    "chunk_count": 0,
                },
            )
            record["chunk_count"] += 1
        if offset is None:
            break
    return list(documents.values())


def delete_document_vectors(client: Any, collection_name: str, document_id: str) -> None:
    if not document_id:
        raise ValueError("document_id is required")
    selector = models.FilterSelector(
        filter=models.Filter(
            should=[
                models.FieldCondition(
                    key="metadata.document_id",
                    match=models.MatchValue(value=document_id),
                ),
                # V2 payloads predate document_id; their inventory key is the
                # original source name, so keep deletion real for migrated data.
                models.FieldCondition(
                    key="metadata.filename",
                    match=models.MatchValue(value=document_id),
                ),
                models.FieldCondition(
                    key="metadata.source",
                    match=models.MatchValue(value=document_id),
                ),
            ]
        )
    )
    client.delete(collection_name=collection_name, points_selector=selector, wait=True)
