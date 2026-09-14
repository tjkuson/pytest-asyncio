"""The runner fixture of a loop scope: its lifetime as a context manager."""

from __future__ import annotations

import asyncio
import contextlib
import sys
from textwrap import dedent

import pytest
from pytest import Pytester
from pytest_asyncio._runner import TaskRunner
from pytest_asyncio.plugin import PytestAsyncioError, _opened

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
        pytest.raises(exit_with) if exit_with else contextlib.nullcontext(),
        _opened(TaskRunner(loop_factory=loop_factory)) as runner,
    ):
        assert runner.get_loop() is loops[0]
        if exit_with:
            raise exit_with
    assert loops[0].is_closed()


@_REQUIRES_311
def test_a_failed_opening_is_a_fixture_error_and_opens_nothing():
    loop_factory, loops = _loop_factory(lambda loop, task: task.cancel())
    with (
        pytest.raises(PytestAsyncioError, match="initialization of pytest-asyncio"),
        _opened(TaskRunner(loop_factory=loop_factory)),
    ):
        pytest.fail("the runner opened")
    assert loops[0].is_closed()


def test_the_loop_is_current_until_the_runner_has_closed(pytester: Pytester):
    """A loop from a loop factory is the current loop through the runner's close."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makeconftest(dedent("""\
            import asyncio

            closed_while_current = []

            class Loop(asyncio.SelectorEventLoop):
                def close(self):
                    if not self.is_closed():
                        try:
                            current = asyncio.get_event_loop()
                        except RuntimeError:
                            current = None
                        closed_while_current.append(current is self)
                    super().close()

            def pytest_asyncio_loop_factories(config, item):
                return {"custom": Loop}
            """))
    pytester.makepyfile(dedent("""\
            import pytest
            from conftest import closed_while_current

            @pytest.mark.asyncio
            async def test_uses_the_loop():
                pass

            def test_the_loop_was_current_while_it_closed():
                assert closed_while_current == [True]
            """))
    result = pytester.runpytest("--asyncio-mode=strict", "-W", "error")
    result.assert_outcomes(passed=2)
