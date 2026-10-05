"""Bocha Web Search client used by the agent's web-search route.

This module only talks to the search API and normalizes its web-page results.
Agent state, retries, answer synthesis, and fallback behavior remain in
``src.app``.
"""

from __future__ import annotations

import json
import os
import re
import socket
from datetime import date
from pathlib import Path
from typing import Any
from urllib import error, request

try:
    from src.runtime.cache import SQLiteTTLCache
    from src.runtime.errors import AuthenticationError, RateLimitError, ToolTimeoutError, ToolUnavailableError
    from src.runtime.resilience import RateLimiter
except ModuleNotFoundError:
    from runtime.cache import SQLiteTTLCache
    from runtime.errors import AuthenticationError, RateLimitError, ToolTimeoutError, ToolUnavailableError
    from runtime.resilience import RateLimiter


DEFAULT_BOCHA_SEARCH_URL = "https://api.bocha.cn/v1/web-search"
BOCHA_FRESHNESS_VALUES = {"noLimit", "oneDay", "oneWeek", "oneMonth", "oneYear"}


class WebSearchError(RuntimeError):
    """Raised for a safe, user-actionable search failure."""


class WebSearchRateLimitError(WebSearchError, RateLimitError):
    pass


class WebSearchAuthenticationError(WebSearchError, AuthenticationError):
    pass


class WebSearchTimeoutError(WebSearchError, ToolTimeoutError):
    pass


class WebSearchUnavailableError(WebSearchError, ToolUnavailableError):
    pass


_RATE_LIMITER = RateLimiter(float(os.getenv("BOCHA_RATE_LIMIT_PER_SECOND", "2")))
_CACHE: SQLiteTTLCache | None = None


def _cache() -> SQLiteTTLCache:
    global _CACHE
    if _CACHE is None:
        default = Path(__file__).resolve().parents[2] / "data" / "runtime-cache.db"
        _CACHE = SQLiteTTLCache(os.getenv("RUNTIME_CACHE_PATH", str(default)))
    return _CACHE


def _api_key() -> str:
    key = os.getenv("BOCHA_SEARCH_API_KEY", "").strip()
    if not key:
        raise WebSearchError("BOCHA_SEARCH_API_KEY 未配置。")
    return key


def _timeout_seconds() -> float:
    raw_value = os.getenv("BOCHA_SEARCH_TIMEOUT_SECONDS", "15")
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise WebSearchError("BOCHA_SEARCH_TIMEOUT_SECONDS 必须是数字。") from exc
    return min(max(value, 1.0), 60.0)


def _boolean_setting(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise WebSearchError(f"{name} 必须是 true 或 false。")


def _freshness() -> str:
    value = os.getenv("BOCHA_SEARCH_FRESHNESS", "noLimit").strip() or "noLimit"
    if value in BOCHA_FRESHNESS_VALUES:
        return value

    date_parts = value.split("..")
    if len(date_parts) not in {1, 2} or not all(
        re.fullmatch(r"\d{4}-\d{2}-\d{2}", part) for part in date_parts
    ):
        raise WebSearchError(
            "BOCHA_SEARCH_FRESHNESS 必须是 noLimit、oneDay、oneWeek、"
            "oneMonth、oneYear、YYYY-MM-DD 或 YYYY-MM-DD..YYYY-MM-DD。"
        )
    try:
        parsed_dates = [date.fromisoformat(part) for part in date_parts]
    except ValueError as exc:
        raise WebSearchError("BOCHA_SEARCH_FRESHNESS 包含无效日期。") from exc
    if len(parsed_dates) == 2 and parsed_dates[0] > parsed_dates[1]:
        raise WebSearchError("BOCHA_SEARCH_FRESHNESS 的开始日期不能晚于结束日期。")
    return value


def _domain_filter(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        return ""
    domains = [domain.strip() for domain in re.split(r"[|,]", value) if domain.strip()]
    if len(domains) > 100:
        raise WebSearchError(f"{name} 最多允许 100 个域名。")
    return value


def _clean_text(value: Any) -> str:
    """Remove control separators sometimes present in search snippets."""
    text = str(value or "")
    return " ".join(
        "".join(character if character.isprintable() else " " for character in text).split()
    )


def _clean_result(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    url = str(item.get("url") or "").strip()
    content = _clean_text(item.get("summary") or item.get("snippet"))
    if not url.startswith(("https://", "http://")) or not content:
        return None

    title = _clean_text(item.get("name") or item.get("displayUrl"))
    result: dict[str, Any] = {
        "title": title or url,
        "url": url,
        "content": content,
        "type": "web",
    }
    published_at = _clean_text(
        item.get("datePublished") or item.get("dateLastCrawled")
    )
    if published_at:
        result["date"] = published_at
    site_name = _clean_text(item.get("siteName"))
    if site_name:
        result["site_name"] = site_name
    return result


def _response_error(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return "博查搜索 API 返回的数据结构不正确。"
    code = str(payload.get("code", "")).strip()
    if code in {"", "200"}:
        return None
    message = _clean_text(payload.get("msg") or payload.get("message"))
    detail = f"：{message}" if message else "。"
    return f"博查搜索 API 返回错误 {code}{detail}"


def _http_error_message(exc: error.HTTPError) -> str:
    response_message = ""
    try:
        payload = json.loads(exc.read().decode("utf-8"))
        if isinstance(payload, dict):
            response_message = _clean_text(payload.get("message") or payload.get("msg"))
    except (AttributeError, UnicodeDecodeError, json.JSONDecodeError):
        pass

    if exc.code == 401:
        return "博查搜索 API 密钥无效。"
    if exc.code == 403:
        return "博查搜索 API 余额不足或当前密钥无访问权限。"
    if exc.code == 429:
        return "博查搜索 API 请求频率已达到限制。"
    if exc.code == 400:
        suffix = f"：{response_message}" if response_message else "。"
        return f"博查搜索 API 拒绝了请求参数{suffix}"
    return f"博查搜索 API 返回 HTTP {exc.code}。"


def search_web(query: str, *, max_results: int = 5) -> list[dict[str, Any]]:
    """Search Bocha and return normalized web-page summaries."""
    cleaned_query = query.strip()
    if not cleaned_query:
        raise WebSearchError("搜索关键词不能为空。")

    bounded_max_results = min(max(int(max_results), 1), 50)
    cache_key = SQLiteTTLCache.key(
        {
            "query": cleaned_query.casefold(), "max_results": bounded_max_results,
            "freshness": _freshness(), "include": _domain_filter("BOCHA_SEARCH_INCLUDE"),
            "exclude": _domain_filter("BOCHA_SEARCH_EXCLUDE"),
        }
    )
    cache_enabled = float(os.getenv("WEB_SEARCH_CACHE_TTL_SECONDS", "300")) > 0
    if cache_enabled:
        cached = _cache().get("web_search", cache_key)
        if isinstance(cached, list) and cached:
            return cached[:bounded_max_results]
    payload_data: dict[str, Any] = {
        "query": cleaned_query,
        "freshness": _freshness(),
        "summary": _boolean_setting("BOCHA_SEARCH_SUMMARY", True),
        "count": bounded_max_results,
    }
    include = _domain_filter("BOCHA_SEARCH_INCLUDE")
    exclude = _domain_filter("BOCHA_SEARCH_EXCLUDE")
    if include:
        payload_data["include"] = include
    if exclude:
        payload_data["exclude"] = exclude

    search_request = request.Request(
        os.getenv("BOCHA_SEARCH_URL", DEFAULT_BOCHA_SEARCH_URL).strip()
        or DEFAULT_BOCHA_SEARCH_URL,
        data=json.dumps(payload_data, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {_api_key()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        _RATE_LIMITER.acquire()
        with request.urlopen(search_request, timeout=_timeout_seconds()) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        if exc.code == 429:
            raise WebSearchRateLimitError(_http_error_message(exc)) from exc
        if exc.code in {401, 403}:
            raise WebSearchAuthenticationError(_http_error_message(exc)) from exc
        raise WebSearchError(_http_error_message(exc)) from exc
    except (socket.timeout, TimeoutError) as exc:
        raise WebSearchTimeoutError("博查联网搜索超时。") from exc
    except error.URLError as exc:
        if isinstance(getattr(exc, "reason", None), (socket.timeout, TimeoutError)):
            raise WebSearchTimeoutError("博查联网搜索超时。") from exc
        raise WebSearchUnavailableError("博查联网搜索网络不可用。") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WebSearchError("博查搜索 API 返回了无法解析的数据。") from exc

    api_error = _response_error(response_payload)
    if api_error:
        raise WebSearchError(api_error)

    data = response_payload.get("data")
    web_pages = data.get("webPages") if isinstance(data, dict) else None
    raw_results = web_pages.get("value") if isinstance(web_pages, dict) else None
    if not isinstance(raw_results, list):
        raise WebSearchError("博查搜索 API 返回的数据结构不正确。")

    results = [result for item in raw_results if (result := _clean_result(item))]
    if not results:
        raise WebSearchError("博查联网搜索没有返回可用网页结果。")
    results = results[:bounded_max_results]
    if cache_enabled:
        ttl = min(max(float(os.getenv("WEB_SEARCH_CACHE_TTL_SECONDS", "300")), 1), 3600)
        _cache().set("web_search", cache_key, results, ttl)
    return results
