"""Safe Qdrant knowledge-base collection lifecycle helpers."""

import re
from typing import Any

from qdrant_client import models


COLLECTION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def validate_collection_name(name: str) -> str:
    normalized = name.strip()
    if not COLLECTION_PATTERN.fullmatch(normalized):
        raise ValueError("知识库名称只能包含字母、数字、下划线和连字符，且不超过 64 个字符")
    return normalized


def list_collection_names(client: Any) -> list[str]:
    response = client.get_collections()
    return sorted(collection.name for collection in response.collections)


def create_collection(client: Any, name: str, vector_size: int) -> str:
    name = validate_collection_name(name)
    if vector_size <= 0:
        raise ValueError("vector_size must be positive")
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )
    return name
