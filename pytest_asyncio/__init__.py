"""The main point for importing pytest-asyncio items."""

from __future__ import annotations

from importlib.metadata import version

from .plugin import PytestAsyncioWarning, fixture, is_async_test

__version__ = version(__name__)

__all__ = ("PytestAsyncioWarning", "fixture", "is_async_test")
