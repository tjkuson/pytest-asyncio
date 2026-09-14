"""
Interruptions of the runner's waits, phase by phase.

An interruption (SIGINT, or a callback of the loop raising) cancels the
task pytest is waiting for and waits for it to end; what the task's
cleanup does decides what pytest gets: a failure of the cleanup, or the
interruption. A second interruption stops waiting; the task ends on its
own, an error it ends with is reported to the loop's exception handler,
and the loop goes on for pytest's teardowns.
"""

from __future__ import annotations

import asyncio
import contextvars
import sys
from collections.abc import AsyncGenerator

import pytest
from pytest_asyncio._runner import TaskRunner

_CLEANUPS = ["propagates", "returns", "fails"]


def _interrupt(loop: asyncio.AbstractEventLoop) -> None:
    def raise_interrupt() -> None:
        raise KeyboardInterrupt("interrupted")

    loop.call_soon(raise_interrupt)


def _copy() -> contextvars.Context:
    return contextvars.copy_context()


def _recover() -> None:
    """Go on after a cancellation, as asyncio's guidance says to."""
    task = asyncio.current_task()
    assert task is not None
    if sys.version_info >= (3, 11):
        task.uncancel()


async def _cleanup(cleanup: str, log: list[str], *, again: bool) -> None:
    """The body of a cleanup, once cancelled: it ends as `cleanup` says."""
    if again:
        _interrupt(asyncio.get_running_loop())
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            pass  # the second interruption cancelled the task once more
    log.append("cleaned up")
    if cleanup == "propagates":
        raise asyncio.CancelledError
    if cleanup == "fails":
        raise ValueError("cleanup failed") from None


def _expected(cleanup: str) -> type[BaseException]:
    return ValueError if cleanup == "fails" else KeyboardInterrupt


@pytest.mark.parametrize("cleanup", _CLEANUPS)
def test_an_interrupted_test(cleanup: str):
    log: list[str] = []

    async def test() -> None:
        _interrupt(asyncio.get_running_loop())
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await _cleanup(cleanup, log, again=False)

    with TaskRunner() as runner:
        with pytest.raises(_expected(cleanup)):
            runner.run(test(), context=_copy())
        assert log == ["cleaned up"]
        assert not runner.teardown_only
        runner.run(asyncio.sleep(0), context=_copy())


@pytest.mark.parametrize("cleanup", _CLEANUPS)
def test_an_interrupted_fixture_setup(cleanup: str):
    """
    A setup that ends without yielding: pytest gets the interruption, or a
    failure of the cleanup. Returning without a yield is not one: cancelled,
    the fixture ended as asked.
    """
    log: list[str] = []

    async def fixture() -> AsyncGenerator[None]:
        _interrupt(asyncio.get_running_loop())
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await _cleanup(cleanup, log, again=False)
            return
        yield

    with TaskRunner() as runner:
        with pytest.raises(_expected(cleanup)):
            runner.start_fixture(fixture(), context=_copy())
        assert log == ["cleaned up"]
        runner.run(asyncio.sleep(0), context=_copy())


@pytest.mark.parametrize("teardown", ["returns", "fails"])
def test_an_interrupted_fixture_setup_that_recovers_and_yields(teardown: str):
    """
    The fixture yielded, but pytest will not tear it down: the runner does,
    before the setup call raises, while what the fixture depends on is
    alive. A failure of that teardown is raised instead of the interruption.
    """
    log: list[str] = []

    async def parent() -> AsyncGenerator[dict[str, bool]]:
        state = {"alive": True}
        yield state
        state["alive"] = False

    async def child(state: dict[str, bool]) -> AsyncGenerator[str]:
        _interrupt(asyncio.get_running_loop())
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            _recover()
        yield "fallback"
        log.append(f"torn down, parent alive: {state['alive']}")
        if teardown == "fails":
            raise ValueError("teardown failed")

    with TaskRunner() as runner:
        started = runner.start_fixture(parent(), context=_copy())
        state = started.setup_result().value
        expected = ValueError if teardown == "fails" else KeyboardInterrupt
        with pytest.raises(expected):
            runner.start_fixture(child(state), context=_copy())
        assert log == ["torn down, parent alive: True"]
        runner.finish_fixture(started)
        assert not state["alive"]


@pytest.mark.parametrize("cleanup", _CLEANUPS)
def test_an_interrupted_fixture_teardown(cleanup: str):
    log: list[str] = []

    async def fixture() -> AsyncGenerator[None]:
        yield
        _interrupt(asyncio.get_running_loop())
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await _cleanup(cleanup, log, again=False)

    with TaskRunner() as runner:
        started = runner.start_fixture(fixture(), context=_copy())
        with pytest.raises(_expected(cleanup)):
            runner.finish_fixture(started)
        assert log == ["cleaned up"]
        runner.run(asyncio.sleep(0), context=_copy())


@pytest.mark.parametrize("phase", ["test", "setup", "teardown"])
@pytest.mark.parametrize("cleanup", _CLEANUPS)
def test_a_second_interruption_abandons_the_cleanup(phase: str, cleanup: str):
    """
    The second interruption is raised at once; the cleanup ends later, when
    the loop next runs; a failure of it is reported to the loop's exception
    handler, once; nothing is left pending at close.
    """
    log: list[str] = []
    reported: list[dict[str, object]] = []

    async def body() -> None:
        _interrupt(asyncio.get_running_loop())
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await _cleanup(cleanup, log, again=True)

    async def test() -> None:
        await body()

    async def fixture_setup() -> AsyncGenerator[None]:
        await body()
        yield

    async def fixture_teardown() -> AsyncGenerator[None]:
        yield
        await body()

    with TaskRunner() as runner:
        runner.get_loop().set_exception_handler(lambda loop, ctx: reported.append(ctx))
        with pytest.raises(KeyboardInterrupt):
            if phase == "test":
                runner.run(test(), context=_copy())
            elif phase == "setup":
                runner.start_fixture(fixture_setup(), context=_copy())
            else:
                started = runner.start_fixture(fixture_teardown(), context=_copy())
                runner.finish_fixture(started)
        assert log == []
        runner.run(asyncio.sleep(0), context=_copy())  # the abandoned cleanup ends
        assert log == ["cleaned up"]
    failures = [str(ctx["exception"]) for ctx in reported]
    assert failures == (["cleanup failed"] if cleanup == "fails" else [])


def test_a_fixture_cancelled_while_the_runner_joins_an_interrupted_test():
    """
    The interrupted test stays the active task while the runner joins it,
    so a fixture cancelled at its yield meanwhile cancels the test again:
    a cleanup that suppressed the first cancellation ends with the second,
    and the loop accepts no new tests.
    """
    log: list[str] = []

    async def fixture() -> AsyncGenerator[None]:
        yield

    async def test(fixture_task: asyncio.Task[None]) -> None:
        _interrupt(asyncio.get_running_loop())
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            fixture_task.cancel()  # as a task group whose child failed would
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                log.append("ended by the second cancellation")
                raise

    with TaskRunner() as runner:
        started = runner.start_fixture(fixture(), context=_copy())
        with pytest.raises(KeyboardInterrupt):
            runner.run(test(started.task), context=_copy())
        assert log == ["ended by the second cancellation"]
        assert runner.teardown_only
        runner.finish_fixture(started)
