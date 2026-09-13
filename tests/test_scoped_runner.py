"""The runner fixture of a loop scope: its lifetime as a context manager."""

from __future__ import annotations

import asyncio
import sys

import pytest
from pytest_asyncio.plugin import PytestAsyncioError, _opened_runner

_REQUIRES_311 = pytest.mark.skipif(
    sys.version_info < (3, 11), reason="asyncio.TaskGroup needs Python 3.11"
)


def _loop_factory(first=None):
    """A loop factory whose task factory does something to the first task."""
    loops = []

    def loop_factory() -> asyncio.AbstractEventLoop:
        loop = asyncio.new_event_loop()
        loops.append(loop)
        seen_first = False

        def task_factory(loop, coro, **kwargs):
            nonlocal seen_first
            task = asyncio.Task(coro, loop=loop, **kwargs)
            if first is not None and not seen_first:
                seen_first = True
                first(loop, task)
            return task

        loop.set_task_factory(task_factory)
        return loop

    return loop_factory, loops


@pytest.mark.parametrize("exit_with", [None, KeyboardInterrupt, GeneratorExit])
def test_the_runner_closes_however_its_scope_ends(exit_with):
    """A normal end, an interruption or a closed generator: the loop closes."""
    loop_factory, loops = _loop_factory()
    with (
        pytest.raises(exit_with) if exit_with else _no_error(),
        _opened_runner(None, loop_factory) as runner,
    ):
        assert runner.get_loop() is loops[0]
        if exit_with:
            raise exit_with
    assert loops[0].is_closed()


@_REQUIRES_311
def test_a_failed_opening_is_a_fixture_error_and_opens_nothing():
    loop_factory, loops = _loop_factory(lambda loop, task: task.cancel())
    with (
        pytest.raises(PytestAsyncioError, match="could not open this event loop"),
        _opened_runner(None, loop_factory),
    ):
        pytest.fail("the runner opened")
    assert loops[0].is_closed()


class _no_error:
    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False
