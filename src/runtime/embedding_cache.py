"""TTL caching proxy for local embedding models."""

from __future__ import annotations

from typing import Any

from .cache import SQLiteTTLCache


class CachedEmbeddings:
    def __init__(self, delegate: Any, cache: SQLiteTTLCache, *, model_name: str, ttl_seconds: float = 604800):
        self.delegate = delegate
        self.cache = cache
        self.model_name = model_name
        self.ttl_seconds = ttl_seconds

    def _key(self, text: str) -> str:
        return self.cache.key({"model": self.model_name, "text": text})

    def embed_query(self, text: str) -> list[float]:
        key = self._key(text)
        cached = self.cache.get("embedding", key)
        if isinstance(cached, list):
            return [float(value) for value in cached]
        result = self.delegate.embed_query(text)
        self.cache.set("embedding", key, result, self.ttl_seconds)
        return result

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float] | None] = [None] * len(texts)
        missing_indexes: list[int] = []
        missing_texts: list[str] = []
        for index, text in enumerate(texts):
            cached = self.cache.get("embedding", self._key(text))
            if isinstance(cached, list):
                results[index] = [float(value) for value in cached]
            else:
                missing_indexes.append(index)
                missing_texts.append(text)
        if missing_texts:
            computed = self.delegate.embed_documents(missing_texts)
            for index, text, vector in zip(missing_indexes, missing_texts, computed):
                results[index] = vector
                self.cache.set("embedding", self._key(text), vector, self.ttl_seconds)
        return [vector or [] for vector in results]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)
