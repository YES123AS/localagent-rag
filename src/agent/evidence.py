"""Evidence normalization, citation labels, and deterministic grounding checks."""

import re
from typing import Any, Iterable, Mapping

from .models import Evidence


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def local_evidence(
    content: str,
    metadata: Mapping[str, Any] | None,
    tool_call_id: str,
    *,
    retrieval_score: float | None = None,
    rerank_score: float | None = None,
) -> Evidence:
    metadata = dict(metadata or {})
    page = _optional_int(metadata.get("page"))
    # PyPDFLoader pages are zero-based; ingestion adds the human-readable page.
    display_page = _optional_int(metadata.get("page_number"))
    if display_page is None and page is not None:
        display_page = page + 1
    name = str(metadata.get("filename") or metadata.get("source") or "本地资料")
    return {
        "source_type": "knowledge_base",
        "content": content,
        "title": name,
        "url": None,
        "published_at": None,
        "document_id": metadata.get("document_id"),
        "document_name": name,
        "file_type": metadata.get("file_type"),
        "page": display_page,
        "chunk_id": str(metadata.get("chunk_id") or "") or None,
        "chunk_index": _optional_int(metadata.get("chunk_index")),
        "retrieval_score": retrieval_score,
        "rerank_score": rerank_score,
        "tool_call_id": tool_call_id,
    }


def web_evidence(result: Mapping[str, Any], tool_call_id: str) -> Evidence:
    return {
        "source_type": "web",
        "content": str(result.get("content", "")),
        "title": str(result.get("title") or "网页来源"),
        "url": str(result.get("url") or "") or None,
        "published_at": str(result.get("date") or "") or None,
        "document_id": None,
        "document_name": None,
        "file_type": None,
        "page": None,
        "chunk_id": None,
        "chunk_index": None,
        "retrieval_score": None,
        "rerank_score": None,
        "tool_call_id": tool_call_id,
    }


def calculator_evidence(expression: str, result: str, tool_call_id: str) -> Evidence:
    return {
        "source_type": "calculator",
        "content": f"{expression} = {result}",
        "title": "计算器",
        "url": None,
        "published_at": None,
        "document_id": None,
        "document_name": None,
        "file_type": None,
        "page": None,
        "chunk_id": None,
        "chunk_index": None,
        "retrieval_score": None,
        "rerank_score": None,
        "tool_call_id": tool_call_id,
    }


def assign_evidence_ids(evidence: Iterable[Evidence]) -> list[Evidence]:
    normalized: list[Evidence] = []
    for index, item in enumerate(evidence, start=1):
        copy = dict(item)
        copy["evidence_id"] = str(index)
        normalized.append(copy)  # type: ignore[arg-type]
    return normalized


def citation_label(item: Mapping[str, Any]) -> str:
    if item.get("source_type") == "knowledge_base":
        label = str(item.get("document_name") or "本地资料")
        if item.get("page") is not None:
            label += f" · Page {item['page']}"
        return label
    if item.get("source_type") == "web":
        return str(item.get("title") or item.get("url") or "网页来源")
    return str(item.get("title") or "计算器")


def citation_report(answer: str, evidence: list[Evidence]) -> dict[str, Any]:
    cited = {int(value) for value in re.findall(r"\[(\d+)\]", answer)}
    valid = set(range(1, len(evidence) + 1))
    invalid = sorted(cited - valid)
    valid_citations = sorted(cited & valid)
    return {
        "citation_count": len(cited),
        "valid_citation_count": len(valid_citations),
        "invalid_citations": invalid,
        "citation_coverage": 1.0 if evidence and valid_citations else 0.0,
        "citation_correctness": (
            len(valid_citations) / len(cited) if cited else (1.0 if not evidence else 0.0)
        ),
        "grounded": not invalid and (bool(valid_citations) or not evidence),
    }
