"""Deterministic and external tools used by the LocalRAG agent."""

from .calculator import CalculatorError, calculate_expression
from .web_search import WebSearchError, search_web

__all__ = [
    "CalculatorError",
    "WebSearchError",
    "calculate_expression",
    "search_web",
]
