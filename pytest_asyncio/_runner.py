"""
Run the async fixtures and tests of an event loop scope, each in a task.

The loop scope opens a loop driver (:class:`asyncio.Runner`) and a task
group that owns every fixture and test task; pytest then runs each fixture
setup, test and fixture teardown as one synchronous call, which drives the
loop until that work has an outcome; closing the scope finishes the group,
which joins its tasks, and closes the loop.
"""

from __future__ import annotations

import asyncio
import contextvars
import enum
import sys
from collections.abc import AsyncGenerator, Callable, Coroutine
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

if sys.version_info >= (3, 11):
    from asyncio import Runner
    from typing import Self, assert_never
else:
    from backports.asyncio.runner import Runner
    from typing_extensions import Self, assert_never

_T = TypeVar("_T")

REFUSED_MESSAGE = (
    "This event loop no longer accepts new tests or fixture setups: a task of "
    "pytest-asyncio's on it was cancelled. Fixture teardowns still run."
)
_NESTED_MESSAGE = (
    "pytest-asyncio cannot start an async fixture or test while the event loop "
    "is running: it drives them from synchronous code"
)


class TaskRunner:
    """
    The event loop scope: a loop driver, and the task group of its tasks.

    Pytest sets up, runs and tears down within its synchronous calls, so
    each of :meth:`run`, :meth:`start_fixture` and :meth:`finish_fixture`
    drives the loop until the work has an outcome. An async generator
    fixture's task outlives the call that started it (see FixtureTask).

    The loop refuses new tests and fixture setups once one of the runner's
    own tasks was cancelled (a fixture's, waiting at its ``yield``, by a
    task group whose child failed, say): the test or setup running at the
    time is cancelled, and only teardowns run until the loop closes.
    """

    def __init__(
        self,
        *,
        debug: bool | None = None,
        loop_factory: Callable[[], asyncio.AbstractEventLoop] | None = None,
    ) -> None:
        self._driver = Runner(debug=debug, loop_factory=loop_factory)
        self._teardown_only = False
        # The test or fixture setup being waited for, to cancel with the
        # runner's own task (see _end_normal_work).
        self._cancellable: asyncio.Task[object] | None = None
        # The tasks pytest stopped waiting for, to cancel once more at close.
        self._abandoned: set[asyncio.Task[object]] = set()

    def open(self) -> None:
        """Start the loop and the task group; on failure, close the loop."""
        try:
            self._group = _TaskGroupHost.start(self._driver, self._end_normal_work)
        except BaseException:
            self._driver.close()
            raise

    def close(self) -> None:
        """Finish the group, which joins every task, then close the loop."""
        try:
            for task in self._abandoned:
                task.cancel()
            self._group.finish()
        finally:
            self._driver.close()

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def get_loop(self) -> asyncio.AbstractEventLoop:
        return self._driver.get_loop()

    @property
    def teardown_only(self) -> bool:
        return self._teardown_only

    def run(
        self,
        coro: Coroutine[Any, Any, _T],
        *,
        context: contextvars.Context,
        name: str | None = None,
    ) -> _T:
        """Run the coroutine in a task of its own and return its result."""
        self._admit(coro)
        outcome: _Outcome[_T] = _Outcome(self.get_loop())
        task = self._group.create_task(
            outcome.capture(coro), context=context, name=name
        )
        task.add_done_callback(outcome.task_ended)
        # A task cancelled before its first step never started the coroutine.
        task.add_done_callback(lambda _: coro.close())
        return self._wait_for(task, outcome, cancellable=True)

    def start_fixture(
        self,
        gen: AsyncGenerator[_T],
        *,
        context: contextvars.Context,
        name: str | None = None,
    ) -> FixtureTask[_T]:
        """Run the fixture up to its yield, in a task of its own."""
        self._admit(gen)
        fixture = self._group.start_fixture(gen, context=context, name=name)
        self._wait_for(fixture.task, fixture.setup_outcome, cancellable=True)
        return fixture

    def finish_fixture(self, fixture: FixtureTask[Any]) -> None:
        """Run the fixture from its yield to its end, in its task."""
        fixture.finish()
        self._wait_for(fixture.task, fixture.teardown_outcome, cancellable=False)

    def _admit(self, work: Coroutine[Any, Any, Any] | AsyncGenerator[Any]) -> None:
        """Refuse new work before it has a task, or a place as the cancellable."""
        if _a_loop_is_running():
            _close(work)
            raise RuntimeError(_NESTED_MESSAGE)
        if self._teardown_only:
            _close(work)
            raise RuntimeError(REFUSED_MESSAGE)

    def _wait_for(
        self, task: asyncio.Task[object], outcome: _Outcome[_T], *, cancellable: bool
    ) -> _T:
        """Drive the loop until the outcome is settled, and return it."""
        self._cancellable = task if cancellable else None
        try:
            try:
                self._driver.run(asyncio.wait([outcome.settled]))
            except BaseException as interruption:
                self._interrupt(task, outcome, interruption)
        finally:
            self._cancellable = None
        return outcome.settled.result()

    def _interrupt(
        self, task: asyncio.Task[object], outcome: _Outcome[_T], exc: BaseException
    ) -> None:
        """
        The wait was interrupted (SIGINT, or a callback raising).

        The task is cancelled and driven until it has an outcome, so that it
        cleans up before pytest tears down what it may be using; a failure
        of that cleanup is reported instead of the interruption, as
        asyncio.Runner reports it. A second interruption gives up waiting:
        the task is cancelled again and left to end on its own (see
        _Outcome.abandon), cancelled once more at close.
        """
        if not outcome.settled.done():
            task.cancel()
            try:
                self._driver.run(asyncio.wait([outcome.settled]))
            except BaseException:
                outcome.abandon()
                task.cancel()
                self._abandoned.add(task)
                task.add_done_callback(self._abandoned.discard)
                raise
        if _failure(outcome.settled) is None:
            raise exc

    def _end_normal_work(self) -> None:
        """One of the runner's own tasks was cancelled."""
        self._teardown_only = True
        if self._cancellable is not None:
            self._cancellable.cancel()
            self._cancellable = None


if sys.version_info >= (3, 11):

    class _TaskGroupHost:
        """
        A task of the runner's own that keeps a task group entered.

        The host task enters an :class:`asyncio.TaskGroup` when the runner
        opens and exits it when the runner closes; the fixtures' and tests'
        tasks are its children, so the group joins them. A child never ends
        with an exception (see _Outcome and FixtureTask): what its coroutine
        raises goes to pytest, so one failure cannot cancel the others. If a
        child does fail, the runner has a bug, and the group raises it when
        the runner closes.

        A cancellation of the host ends the normal work of the loop; the
        group cancels its children and waits for them, as any task group
        does, and the fixtures cancelled at their yield keep waiting for
        pytest to tear them down.
        """

        @classmethod
        def start(cls, driver: Runner, cancelled: Callable[[], None]) -> Self:
            """Start the host and drive the loop until the group is entered."""
            loop = driver.get_loop()
            closing = asyncio.Event()
            entered: _Outcome[asyncio.TaskGroup] = _Outcome(loop)
            keep_entered = cls._keep_group_entered(entered, closing, cancelled)
            host = loop.create_task(keep_entered, name="pytest-asyncio")
            host.add_done_callback(entered.task_ended)
            driver.run(asyncio.wait([entered.settled]))
            return cls(driver, host, entered.settled.result(), closing, cancelled)

        def __init__(
            self,
            driver: Runner,
            host: asyncio.Task[None],
            group: asyncio.TaskGroup,
            closing: asyncio.Event,
            cancelled: Callable[[], None],
        ) -> None:
            self._driver = driver
            self._host = host
            self._group = group
            self._closing = closing
            self._cancelled = cancelled
            # The fixtures whose tasks are alive, to close at finish.
            self._fixtures: set[FixtureTask[Any]] = set()

        def create_task(
            self,
            coro: Coroutine[Any, Any, _T],
            *,
            context: contextvars.Context,
            name: str | None,
        ) -> asyncio.Task[_T]:
            # Created in the given context: the task copies it, as any does.
            return context.run(self._group.create_task, coro, name=name)

        def start_fixture(
            self,
            gen: AsyncGenerator[_T],
            *,
            context: contextvars.Context,
            name: str | None,
        ) -> FixtureTask[_T]:
            fixture = FixtureTask(
                gen,
                self._driver.get_loop(),
                lambda live: self.create_task(live, context=context, name=name),
                cancelled=self._cancelled,
            )
            self._fixtures.add(fixture)
            fixture.task.add_done_callback(lambda _: self._fixtures.discard(fixture))
            return fixture

        def finish(self) -> None:
            """Close the fixtures still alive; exit the group, joining every task."""
            for fixture in list(self._fixtures):
                fixture.close()
            self._closing.set()
            self._driver.run(asyncio.wait([self._host]))
            failure = _failure(self._host)
            if failure is not None:
                raise failure

        @staticmethod
        async def _keep_group_entered(
            entered: _Outcome[asyncio.TaskGroup],
            closing: asyncio.Event,
            cancelled: Callable[[], None],
        ) -> None:
            async with asyncio.TaskGroup() as group:
                entered.set_result(group)
                try:
                    await closing.wait()
                except asyncio.CancelledError:
                    cancelled()
                    raise

else:

    class _TaskGroupHost:
        """
        The tasks of the loop scope, without :class:`asyncio.TaskGroup`.

        The fixtures' tasks are joined by hand when the runner closes; the
        loop's own close joins the others.
        """

        @classmethod
        def start(cls, driver: Runner, cancelled: Callable[[], None]) -> Self:
            return cls(driver, cancelled)

        def __init__(self, driver: Runner, cancelled: Callable[[], None]) -> None:
            self._driver = driver
            self._cancelled = cancelled
            self._fixtures: set[FixtureTask[Any]] = set()

        def create_task(
            self,
            coro: Coroutine[Any, Any, _T],
            *,
            context: contextvars.Context,
            name: str | None,
        ) -> asyncio.Task[_T]:
            loop = self._driver.get_loop()
            return context.run(loop.create_task, coro, name=name)

        def start_fixture(
            self,
            gen: AsyncGenerator[_T],
            *,
            context: contextvars.Context,
            name: str | None,
        ) -> FixtureTask[_T]:
            fixture = FixtureTask(
                gen,
                self._driver.get_loop(),
                lambda live: self.create_task(live, context=context, name=name),
                cancelled=self._cancelled,
            )
            self._fixtures.add(fixture)
            fixture.task.add_done_callback(lambda _: self._fixtures.discard(fixture))
            return fixture

        def finish(self) -> None:
            fixtures = list(self._fixtures)
            for fixture in fixtures:
                fixture.close()
            if fixtures:
                self._driver.run(asyncio.wait([f.task for f in fixtures]))


@dataclass(frozen=True)
class _Returned(Generic[_T]):
    """A coroutine's return value, which may be None or an exception."""

    value: _T


class _Outcome(Generic[_T]):
    """
    A result of async work, for synchronous pytest to wait for and read.

    Settled by the work itself (:meth:`set_result`, :meth:`set_exception`),
    or by the end of its task if that comes first (:meth:`task_ended`), so
    that the wait cannot outlive the task. :meth:`capture` runs a coroutine
    as the body of a task whose end settles the outcome with what the
    coroutine returned or raised.

    Once pytest has stopped waiting (:meth:`abandon`), an exception is
    reported to the loop's exception handler instead: nobody else would.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.settled: asyncio.Future[_T] = loop.create_future()
        self._abandoned = False
        self._returned: _Returned[_T] | None = None
        self._raised: BaseException | None = None

    async def capture(self, coro: Coroutine[Any, Any, _T]) -> None:
        # What the coroutine raises is kept for pytest rather than ending
        # the task with it: in a task group, that would cancel the other
        # tasks. A cancellation ends the task as it does any task.
        try:
            self._returned = _Returned(await coro)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            self._raised = exc

    def task_ended(self, task: asyncio.Task[object]) -> None:
        """Settle from the task's end: what was captured, or how the task ended."""
        if self._raised is not None:
            # An exception raised while a cancellation was pending wins, as
            # it does in any task.
            self.set_exception(self._raised)
            return
        try:
            task.result()
        except BaseException as exc:
            self.set_exception(exc)
            return
        if self._returned is not None:
            self.set_result(self._returned.value)
        elif not self.settled.done():
            self.set_exception(RuntimeError("The task ended without a result"))

    def set_result(self, value: _T) -> None:
        if not self.settled.done():
            self.settled.set_result(value)

    def set_exception(self, exc: BaseException) -> None:
        if not self.settled.done():
            self.settled.set_exception(exc)
        elif self._abandoned:
            self._report_unobserved(exc)

    def abandon(self) -> None:
        """Stop waiting: an exception, settled or to come, is reported instead."""
        if self._abandoned:
            return
        self._abandoned = True
        if not self.settled.done():
            self.settled.cancel()
        elif (exc := _failure(self.settled)) is not None:
            self._report_unobserved(exc)

    def _report_unobserved(self, exc: BaseException) -> None:
        if not isinstance(exc, asyncio.CancelledError):
            self.settled.get_loop().call_exception_handler(
                {
                    "message": "Exception from an async fixture or test after "
                    "pytest stopped waiting for it",
                    "exception": exc,
                }
            )


@dataclass(frozen=True)
class FixtureSetup(Generic[_T]):
    """A successful fixture setup: the value yielded, and the context then."""

    value: _T
    context: contextvars.Context


class _TeardownAction(enum.Enum):
    FINISH = enum.auto()  # run the fixture from its yield to its end
    CLOSE = enum.auto()  # close the generator: the runner is closing


class FixtureTask(Generic[_T]):
    """
    An async generator fixture, run in a task of its own.

    The task runs the setup, waits at the ``yield`` for pytest's decision,
    then runs the teardown (:meth:`finish`) or closes the generator
    (:meth:`close`). Setup and teardown therefore share a task and a
    context, and a task group, cancel scope or timeout entered before the
    ``yield`` is exited by its own task.

    A cancellation of the task while it waits ends the normal work of the
    loop (``cancelled``); the task keeps waiting, so that the teardown, in
    this task, is what exits the scope. The setup and teardown outcomes go
    to pytest; nothing escapes the task.
    """

    def __init__(
        self,
        gen: AsyncGenerator[_T],
        loop: asyncio.AbstractEventLoop,
        create_task: Callable[[Coroutine[Any, Any, None]], asyncio.Task[None]],
        *,
        cancelled: Callable[[], None],
    ) -> None:
        self._gen = gen
        self._cancelled = cancelled
        self._action: asyncio.Future[_TeardownAction] = loop.create_future()
        self.setup_outcome: _Outcome[FixtureSetup[_T]] = _Outcome(loop)
        self.teardown_outcome: _Outcome[None] = _Outcome(loop)
        self.task = create_task(self._live())
        self.task.add_done_callback(self._task_ended)

    @property
    def setup(self) -> FixtureSetup[_T]:
        return self.setup_outcome.settled.result()

    def finish(self) -> None:
        self._action.set_result(_TeardownAction.FINISH)

    def close(self) -> None:
        """Nobody waits for the outcome; an exception is reported instead."""
        self.teardown_outcome.abandon()
        if not self._action.done():
            self._action.set_result(_TeardownAction.CLOSE)
        else:
            # The teardown pytest asked for was abandoned: ask it to end.
            self.task.cancel()

    async def _live(self) -> None:
        try:
            value = await anext(self._gen)
        except BaseException as exc:
            self.setup_outcome.set_exception(exc)
            return
        self.setup_outcome.set_result(FixtureSetup(value, contextvars.copy_context()))
        while not self._action.done():
            try:
                # asyncio.wait, so that a cancellation of this task neither
                # cancels the pending action nor ends the wait for it.
                await asyncio.wait([self._action])
            except asyncio.CancelledError:
                self._cancelled()
        action = self._action.result()
        if action is _TeardownAction.FINISH:
            try:
                await anext(self._gen)
            except StopAsyncIteration:
                self.teardown_outcome.set_result(None)
            except BaseException as exc:
                self.teardown_outcome.set_exception(exc)
            else:
                self.teardown_outcome.set_exception(
                    ValueError("Async generator fixture yielded more than once")
                )
        elif action is _TeardownAction.CLOSE:
            try:
                await self._gen.aclose()
            except BaseException as exc:
                self.teardown_outcome.set_exception(exc)
            else:
                self.teardown_outcome.set_result(None)
        else:
            assert_never(action)

    def _task_ended(self, task: asyncio.Task[None]) -> None:
        # The phase that owed a result and did not publish it ended with
        # the task: the setup, or the teardown once it was asked for.
        if not self.setup_outcome.settled.done():
            self.setup_outcome.task_ended(task)
        elif self._action.done():
            self.teardown_outcome.task_ended(task)


def _a_loop_is_running() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def _close(work: Coroutine[Any, Any, Any] | AsyncGenerator[Any]) -> None:
    """Close work that will never run, so nothing warns that it never ran."""
    if isinstance(work, Coroutine):
        work.close()
    # An async generator not started needs no closing.


def _failure(future: asyncio.Future[Any]) -> BaseException | None:
    """The exception of the done future, unless it is a cancellation."""
    exc = None if future.cancelled() else future.exception()
    return None if isinstance(exc, asyncio.CancelledError) else exc
