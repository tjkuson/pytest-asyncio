"""Scoped asyncio runner lifecycle, independent of pytest fixtures."""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import dataclasses
import functools
import sys
import warnings
from asyncio import AbstractEventLoop
from collections.abc import Callable, Coroutine, Iterator
from typing import Any, Literal, TypeAlias

import pytest
from _pytest.nodes import Node

if sys.version_info >= (3, 11):
    from asyncio import Runner
    from typing import assert_never
else:
    from backports.asyncio.runner import Runner
    from typing_extensions import assert_never

_ScopeName = Literal["session", "package", "module", "class", "function"]
LoopFactory: TypeAlias = Callable[[], AbstractEventLoop]


@dataclasses.dataclass(frozen=True, eq=False, slots=True)
class _LoopFactoryVariant:
    """A stable loop-factory identity shared by parametrized test items."""

    factory: LoopFactory | None


_DEFAULT_LOOP_FACTORY = _LoopFactoryVariant(factory=None)


class _ManagedRunner:
    def __init__(self, *, debug: bool, factory: LoopFactory | None) -> None:
        self._runner = Runner(debug=debug, loop_factory=factory)

    @property
    def loop(self) -> AbstractEventLoop:
        return self._runner.get_loop()

    def run(
        self,
        awaitable: Coroutine[Any, Any, Any],
        *,
        context: contextvars.Context | None = None,
    ) -> Any:
        return self._runner.run(awaitable, context=context)

    def close(self) -> None:
        if self.loop.is_closed():
            raise RuntimeError(
                "pytest-asyncio's event loop was closed before pytest-asyncio "
                "could clean it up"
            )
        self._runner.close()


def _scope_root(node: Node, scope: _ScopeName) -> Node:
    if scope == "function":
        return node
    node_type: type[Node]
    if scope == "class":
        node_type = pytest.Class
    elif scope == "module":
        node_type = pytest.Module
    elif scope == "package":
        node_type = pytest.Package
    elif scope == "session":
        return node.session
    else:
        assert_never(scope)
    root = node.getparent(node_type)
    if root is not None:
        return root
    # Match pytest's scope-node behavior outside a class or package.
    return node if scope == "class" else node.session


class _LoopManager:
    """Own one active runner for each pytest scope root."""

    def __init__(self, *, debug: bool) -> None:
        self._debug = debug
        self._active: dict[
            tuple[_ScopeName, Node],
            tuple[_LoopFactoryVariant, _ManagedRunner],
        ] = {}

    def get_runner(
        self,
        node: Node,
        scope: _ScopeName,
        variant: _LoopFactoryVariant,
    ) -> _ManagedRunner:
        root = _scope_root(node, scope)
        key = (scope, root)
        active = self._active.get(key)
        if active is not None:
            active_variant, runner = active
            if active_variant is variant:
                return runner
            del self._active[key]
            runner.close()
        runner = _ManagedRunner(debug=self._debug, factory=variant.factory)
        self._active[key] = (variant, runner)
        root.addfinalizer(functools.partial(self._finalize_runner, key, runner))
        return runner

    def _finalize_runner(
        self,
        key: tuple[_ScopeName, Node],
        runner: _ManagedRunner,
    ) -> None:
        active = self._active.get(key)
        if active is None or active[1] is not runner:
            return
        del self._active[key]
        runner.close()

    def close_all(self) -> None:
        for _, runner in reversed(tuple(self._active.values())):
            runner.close()
        self._active.clear()


@contextlib.contextmanager
def _temporary_event_loop(loop: AbstractEventLoop) -> Iterator[None]:
    try:
        old_loop = _get_event_loop_no_warn()
    except RuntimeError:
        old_loop = None
    if old_loop is loop:
        yield
        return
    _set_event_loop(loop)
    try:
        yield
    finally:
        _set_event_loop(old_loop)


def _get_event_loop_no_warn() -> AbstractEventLoop:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return asyncio.get_event_loop()


def _set_event_loop(loop: AbstractEventLoop | None) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        asyncio.set_event_loop(loop)
