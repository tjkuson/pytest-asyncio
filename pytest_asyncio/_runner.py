"""Run the async fixtures and tests of an event loop scope, each in a task."""

from __future__ import annotations

import asyncio
import contextvars
import sys
from collections.abc import AsyncGenerator, Callable, Coroutine
from typing import Any, Generic, TypeVar

if sys.version_info >= (3, 11):
    from asyncio import Runner
    from typing import Self
else:
    from backports.asyncio.runner import Runner
    from typing_extensions import Self

_T = TypeVar("_T")

_REFUSED_MESSAGE = (
    "The coroutine was refused: an async generator fixture of this event loop "
    "was cancelled at its yield, by a task group, cancel scope or timeout "
    "spanning the yield. Only fixture teardowns run until the loop is closed."
)
_DID_NOT_STOP_MESSAGE = "Async generator fixture didn't stop. Yield only once."
_RUNNING_LOOP_MESSAGE = (
    "pytest-asyncio fixtures cannot be requested from a running event loop"
)


class TaskRunner:
    """
    Run the async fixtures and tests of an event loop scope, each in a task.

    A test or a coroutine fixture runs in a task of its own, in a copy of
    the given context, as with :meth:`asyncio.Runner.run`. An async
    generator fixture runs in a task of its own too, one that stays alive
    across the fixture's ``yield``: the task that entered a task group,
    cancel scope or timeout before the ``yield`` is the task that exits it
    at teardown (see :class:`FixtureTask`).

    A cancellation of a fixture's task while it waits at the ``yield``, e.g.
    by a task group whose child failed, ends the normal work of the loop.
    The test or fixture setup running at the time is cancelled, and from
    then on only fixture teardowns run, until the loop closes; anything
    else is refused. The fixture's own teardown exits the scope, which
    reports what happened.

    An interruption of a wait (SIGINT, or a BaseException raised by a
    callback of the loop) cancels the task being waited for and waits for
    it to end, so that it cleans up before pytest tears down what it may be
    using. A second interruption abandons it.
    """

    def __init__(
        self,
        *,
        debug: bool | None = None,
        loop_factory: Callable[[], asyncio.AbstractEventLoop] | None = None,
    ) -> None:
        self._runner = Runner(debug=debug, loop_factory=loop_factory)
        self._fixtures: set[FixtureTask[Any]] = set()
        # The task pytest waits for, if a fixture's cancellation is to end
        # it (see _fixture_cancelled): a test or a fixture setup.
        self._running: asyncio.Task[Any] | None = None
        self._teardown_only = False

    def __enter__(self) -> Self:
        self._runner.__enter__()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def get_loop(self) -> asyncio.AbstractEventLoop:
        return self._runner.get_loop()

    def run(
        self,
        coro: Coroutine[Any, Any, _T],
        *,
        context: contextvars.Context,
        name: str | None = None,
    ) -> _T:
        """Run the coroutine in a task of its own and return its result."""
        if _in_running_loop() or self._teardown_only:
            coro.close()
            if self._teardown_only:
                raise asyncio.CancelledError(_REFUSED_MESSAGE)
            raise RuntimeError(_RUNNING_LOOP_MESSAGE)
        task = _create_task(self.get_loop(), coro, context=context, name=name)
        return self._wait_for(task, _outcome_of(task), cancellable=True)

    def start_fixture(
        self,
        gen: AsyncGenerator[_T],
        *,
        context: contextvars.Context,
        name: str | None = None,
    ) -> FixtureTask[_T]:
        """Run the fixture up to its yield, in a task of its own."""
        if _in_running_loop():
            raise RuntimeError(_RUNNING_LOOP_MESSAGE)
        if self._teardown_only:
            raise asyncio.CancelledError(_REFUSED_MESSAGE)
        fixture: FixtureTask[_T] = FixtureTask(self, gen, context=context, name=name)
        self._wait_for(fixture.task, fixture.setup, cancellable=True)
        self._fixtures.add(fixture)
        return fixture

    def finish_fixture(self, fixture: FixtureTask[Any]) -> None:
        """Run the fixture from its yield to its end, in its task."""
        if _in_running_loop():
            raise RuntimeError(_RUNNING_LOOP_MESSAGE)
        self._fixtures.discard(fixture)
        fixture.resume()
        self._wait_for(fixture.task, fixture.teardown, cancellable=False)

    def close(self) -> None:
        try:
            # Fixtures never torn down (the session was aborted) are
            # finalised in their tasks as the runner cancels them.
            for fixture in self._fixtures:
                fixture.finalise()
        finally:
            self._runner.close()

    def _wait_for(
        self, task: asyncio.Task[Any], outcome: asyncio.Future[_T], *, cancellable: bool
    ) -> _T:
        """Run the loop until the task has produced the outcome; return it."""
        self._running = task if cancellable else None
        try:
            try:
                self._runner.run(_wait_until(outcome))
            except BaseException as interruption:
                self._interrupt(task, outcome, interruption)
        finally:
            self._running = None
        return outcome.result()

    def _interrupt(
        self, task: asyncio.Task[Any], outcome: asyncio.Future[Any], exc: BaseException
    ) -> None:
        """
        End the interrupted task, then raise the interruption.

        The task is cancelled and run until it has produced the outcome, so
        that it cleans up before pytest goes on. A failure of that cleanup,
        not the cancellation, is then reported instead of the interruption,
        as asyncio.Runner reports it and as for a synchronous test; an
        outcome that ended, or suppressed, the cancellation does not
        suppress the interruption. A second interruption cancels the task
        again and abandons the outcome.
        """
        if not outcome.done():
            task.cancel()
            try:
                self._runner.run(_wait_until(outcome))
            except BaseException:
                outcome.cancel()
                task.cancel()
                raise
        if _failure(outcome) is None:
            raise exc

    def _fixture_cancelled(self) -> None:
        """A fixture's task was cancelled at its yield (see FixtureTask)."""
        self._teardown_only = True
        if self._running is not None:
            self._running.cancel()


class FixtureTask(Generic[_T]):
    """
    An async generator fixture, run in a task of its own.

    The task runs the fixture up to its ``yield``, waits there until pytest
    tears the fixture down, and runs it to its end. A cancellation of the
    task while it waits is a scope's (a task group whose child failed, a
    cancel scope, a timeout): the runner is told, once, and the task keeps
    waiting, so that the scope is exited by its own task at teardown.
    """

    def __init__(
        self,
        runner: TaskRunner,
        gen: AsyncGenerator[_T],
        *,
        context: contextvars.Context,
        name: str | None,
    ) -> None:
        loop = runner.get_loop()
        self._runner = runner
        self._gen = gen
        self._resume = asyncio.Event()
        self._closing = False
        # The outcomes of the two phases, for the runner to wait for.
        self.setup: asyncio.Future[_T] = loop.create_future()
        self.teardown: asyncio.Future[None] = loop.create_future()
        # A copy of the task's context once the fixture was set up.
        self.context_after: contextvars.Context | None = None
        self.task = _create_task(loop, self._live(), context=context, name=name)

    @property
    def value(self) -> _T:
        return self.setup.result()

    def resume(self) -> None:
        self._resume.set()

    def finalise(self) -> None:
        """Have the task close the fixture when it is next cancelled."""
        self._closing = True
        self.task.cancel()

    async def _live(self) -> None:
        try:
            value = await self._gen.__anext__()
        except BaseException as exc:
            _publish(self.setup, exc)
            return
        self.context_after = contextvars.copy_context()
        _publish(self.setup, value=value)
        told = False
        while not self._resume.is_set():
            try:
                await self._resume.wait()
            except asyncio.CancelledError:
                if self._closing:
                    await self._gen.aclose()
                    return
                if not told:
                    told = True
                    self._runner._fixture_cancelled()
        try:
            await self._gen.__anext__()
        except StopAsyncIteration:
            _publish(self.teardown, value=None)
        except BaseException as exc:
            _publish(self.teardown, exc)
        else:
            _publish(self.teardown, ValueError(_DID_NOT_STOP_MESSAGE))


def _create_task(
    loop: asyncio.AbstractEventLoop,
    coro: Coroutine[Any, Any, Any],
    *,
    context: contextvars.Context,
    name: str | None,
) -> asyncio.Task[Any]:
    # The task copies the current context, as any task does; here that is a
    # copy of the given one.
    return context.run(loop.create_task, coro, name=name)


def _outcome_of(task: asyncio.Task[_T]) -> asyncio.Future[_T]:
    """A future holding the task's result or exception once it is done."""
    outcome: asyncio.Future[_T] = task.get_loop().create_future()

    def publish(task: asyncio.Task[_T]) -> None:
        if outcome.done():
            return
        if task.cancelled():
            outcome.cancel()
        elif (exc := task.exception()) is not None:
            outcome.set_exception(exc)
        else:
            outcome.set_result(task.result())

    task.add_done_callback(publish)
    return outcome


def _publish(
    future: asyncio.Future[Any], exc: BaseException | None = None, *, value: Any = None
) -> None:
    if future.done():
        return
    if exc is not None:
        future.set_exception(exc)
    else:
        future.set_result(value)


async def _wait_until(future: asyncio.Future[Any]) -> None:
    # Unlike awaiting the future directly, this neither cancels it when the
    # waiter is cancelled nor raises its exception.
    await asyncio.wait([future])


def _failure(future: asyncio.Future[Any]) -> BaseException | None:
    """The exception of the done future, unless it is a cancellation."""
    exc = None if future.cancelled() else future.exception()
    return None if isinstance(exc, asyncio.CancelledError) else exc


def _in_running_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True
