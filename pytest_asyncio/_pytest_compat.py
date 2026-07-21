"""Narrow compatibility boundary for pytest internals without public equivalents."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias, cast

import pytest
from _pytest.fixtures import FuncFixtureInfo

FixtureInfo: TypeAlias = FuncFixtureInfo


def get_fixture_info(node: object) -> FixtureInfo:
    """Return pytest's statically resolved fixture graph for a function node."""
    return cast(FixtureInfo, node._fixtureinfo)  # type: ignore[attr-defined]


def get_requesting_item(request: pytest.FixtureRequest) -> pytest.Function:
    """
    Return the test item that initiated a fixture request.

    ``request.node`` normally identifies a fixture's cache node, but package-scoped
    fixtures can fall back to the session node based on where they were defined.
    The requesting item is therefore required to identify the test's package.
    """
    return cast(pytest.Function, request._pyfuncitem)


def replace_fixture_function(
    fixturedef: pytest.FixtureDef[Any],
    function: Callable[..., Any],
    argnames: tuple[str, ...],
) -> None:
    """
    Replace a fixture implementation before its first setup.

    Pytest has no public hook for converting a ``@pytest.fixture`` definition.
    Auto mode needs this one-time adaptation so pytest sees the wrapper's formal
    dependencies and applies normal fixture caching and teardown semantics.
    """
    fixturedef.func = function  # type: ignore[misc]
    fixturedef.argnames = argnames  # type: ignore[misc]
