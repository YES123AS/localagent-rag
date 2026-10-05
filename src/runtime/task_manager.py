"""Compatibility export for the V4.5 persistent background worker."""

from .task_manager_v45 import TaskManager

__all__ = ["TaskManager"]
