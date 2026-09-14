"""
Run the async fixtures and tests of an event loop scope, each in a task.

The scope opens a loop driver (:class:`asyncio.Runner`) and a task group
that owns every fixture and test task. Pytest then runs each fixture setup,
test and fixture teardown as one synchronous call, which drives the loop
until that work is done and reads its result: a fixture's setup is ready
while its task lives on; a test, and a fixture's teardown, are done when
their task is. Closing the scope finishes the group, which joins its
tasks, and closes the loop.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import enum
import sys
from collections.abc import AsyncGenerator, Callable, Coroutine, Iterator
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
    "This event loop no longer accepts new tests or fixture setups: an async "
    "generator fixture was cancelled while it waited at its yield, or "
    "pytest-asyncio's own task was. Fixture teardowns still run."
)
_NESTED_MESSAGE = (
    "pytest-asyncio cannot start an async fixture or test while the event loop "
    "is running: it drives them from synchronous code"
)


class TaskRunner:
    """
    The event loop scope: a loop driver, and the task group of its tasks.

    The loop refuses new tests and fixture setups once an async generator
    fixture was cancelled while it waited at its ``yield`` (by a task group
    whose child failed, say), or the runner's own task was: the test or
    setup running at the time is cancelled, and only teardowns run until
    the loop closes (see FixtureTask, and _stop_normal_work).

    An interruption of a wait (SIGINT, or a callback raising) cancels the
    task waited for and waits for it to end, so that it cleans up before
    pytest tears down the fixtures it may use, and raises what the cleanup
    raised instead of the interruption, as asyncio.Runner does. A second
    interruption stops waiting (see _join_or_abandon).
    """

    def __init__(
        self,
        *,
        debug: bool | None = None,
        loop_factory: Callable[[], asyncio.AbstractEventLoop] | None = None,
    ) -> None:
        self._driver = Runner(debug=debug, loop_factory=loop_factory)
        self._teardown_only = False
        # The test or fixture setup being waited for, cancelled with a
        # fixture waiting at its yield (see _stop_normal_work).
        self._active: asyncio.Task[object] | None = None
        # The generator fixtures whose tasks are alive: closed at close.
        self._fixtures: set[FixtureTask[Any]] = set()
        # The tasks pytest stopped waiting for: cancelled once more at close.
        self._abandoned: set[asyncio.Task[object]] = set()

    def open(self) -> None:
        """Start the loop and the task group; on failure, close the loop."""
        try:
            self._group = _TaskGroupHost.start(self._driver, self._stop_normal_work)
        except BaseException:
            self._driver.close()
            raise

    def close(self) -> None:
        """Finish the group, which joins every task, then close the loop."""
        try:
            for fixture in list(self._fixtures):
                fixture.request_close()
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
        captured: _Captured[_T] = _Captured()
        task = self._group.create_task(captured.run(coro), context=context, name=name)
        # A task cancelled before its first step never started the coroutine.
        task.add_done_callback(lambda _: coro.close())
        with self._as_active(task):
            interruption = self._drive_until(task)
            if interruption is None:
                return captured.read(task)
            task.cancel()
            self._join_or_abandon(task, captured.report_unobserved)
            raise captured.failure() or interruption

    def start_fixture(
        self,
        gen: AsyncGenerator[_T],
        *,
        context: contextvars.Context,
        name: str | None = None,
    ) -> FixtureTask[_T]:
        """Run the fixture up to its yield, in a task of its own."""
        self._admit(gen)
        fixture = FixtureTask(
            gen,
            lambda live: self._group.create_task(live, context=context, name=name),
            stop_normal_work=self._stop_normal_work,
        )
        self._fixtures.add(fixture)
        fixture.task.add_done_callback(lambda _: self._fixtures.discard(fixture))
        with self._as_active(fixture.task):
            interruption = self._drive_until(fixture.task, fixture.ready)
            if interruption is None:
                fixture.setup_result()  # raises if the setup failed
                return fixture
            if not fixture.ready.done():
                fixture.task.cancel()
                self._join_or_abandon(
                    fixture.task, fixture.report_unobserved, fixture.ready
                )
        # Interrupted, the fixture is not handed to pytest, which will not
        # tear it down: it ends here, before pytest tears down the fixtures
        # it depends on. One that yielded all the same is torn down, as
        # pytest would have.
        if fixture.yielded:
            fixture.request_teardown()
            self._join_or_abandon(fixture.task, fixture.report_unobserved)
            raise fixture.teardown_failure() or interruption
        raise fixture.setup_failure() or interruption

    def finish_fixture(self, fixture: FixtureTask[Any]) -> None:
        """Run the fixture from its yield to its end, in its task."""
        fixture.request_teardown()
        interruption = self._drive_until(fixture.task)
        if interruption is None:
            fixture.teardown_result()
            return
        fixture.task.cancel()
        self._join_or_abandon(fixture.task, fixture.report_unobserved)
        raise fixture.teardown_failure() or interruption

    def _admit(self, work: Coroutine[Any, Any, Any] | AsyncGenerator[Any]) -> None:
        """Refuse new work before it has a task, or a place as the active one."""
        if _a_loop_is_running():
            _close(work)
            raise RuntimeError(_NESTED_MESSAGE)
        if self._teardown_only:
            _close(work)
            raise RuntimeError(REFUSED_MESSAGE)

    @contextlib.contextmanager
    def _as_active(self, task: asyncio.Task[object]) -> Iterator[None]:
        """The test or setup pytest waits for, cancelled by _stop_normal_work."""
        self._active = task
        try:
            yield
        finally:
            self._active = None

    def _join_or_abandon(
        self,
        task: asyncio.Task[object],
        unobserved: Callable[[asyncio.Task[object]], None],
        *also: asyncio.Future[Any],
    ) -> None:
        """
        Drive the loop until the task ends, or one of the other futures is done.

        A second interruption stops waiting: the task is cancelled again and
        left to end on its own, ``unobserved`` reports what it ends with, and
        close cancels it once more.
        """
        interruption = self._drive_until(task, *also)
        if interruption is not None:
            task.add_done_callback(unobserved)
            task.cancel()
            self._abandoned.add(task)
            task.add_done_callback(self._abandoned.discard)
            raise interruption

    def _drive_until(self, *waiters: asyncio.Future[Any]) -> BaseException | None:
        """Run the loop until a waiter is done; return the interruption, if any."""
        # The driver runs the wait in a task of its own, which a callback
        # raising leaves behind, pending: released ends it with this call.
        released: asyncio.Future[None] = self.get_loop().create_future()
        try:
            self._driver.run(
                asyncio.wait([*waiters, released], return_when=asyncio.FIRST_COMPLETED)
            )
        except BaseException as interruption:
            return interruption
        finally:
            released.cancel()
        return None

    def _stop_normal_work(self) -> None:
        """A fixture waiting at its yield, or the runner's task, was cancelled."""
        self._teardown_only = True
        if self._active is not None:
            self._active.cancel()
            self._active = None


if sys.version_info >= (3, 11):

    class _TaskGroupHost:
        """
        A task of the runner's own that keeps a task group entered.

        The host enters an :class:`asyncio.TaskGroup` when the runner opens
        and exits it at :meth:`finish`; the fixtures' and tests' tasks are
        its children, so the group joins them. A child only ends with an
        exception if the runner has a bug (see _Captured and FixtureTask),
        and the group raises it at finish. A cancellation of the host stops
        the loop's normal work; the group then cancels its children and waits
        for them, as any task group does.
        """

        @classmethod
        def start(cls, driver: Runner, stop_normal_work: Callable[[], None]) -> Self:
            """Start the host and drive the loop until the group is entered."""
            loop = driver.get_loop()
            closing = asyncio.Event()
            entered: asyncio.Future[asyncio.TaskGroup] = loop.create_future()
            keep_entered = cls._keep_group_entered(entered, closing, stop_normal_work)
            host = loop.create_task(keep_entered, name="pytest-asyncio")
            startup: list[asyncio.Future[Any]] = [entered, host]
            driver.run(asyncio.wait(startup, return_when=asyncio.FIRST_COMPLETED))
            if not entered.done():
                host.result()  # the host ended before entering: raises how
                raise RuntimeError("The task group host ended before entering")
            return cls(driver, host, entered.result(), closing)

        def __init__(
            self,
            driver: Runner,
            host: asyncio.Task[None],
            group: asyncio.TaskGroup,
            closing: asyncio.Event,
        ) -> None:
            self._driver = driver
            self._host = host
            self._group = group
            self._closing = closing

        def create_task(
            self,
            coro: Coroutine[Any, Any, _T],
            *,
            context: contextvars.Context,
            name: str | None,
        ) -> asyncio.Task[_T]:
            # Created in the given context: the task copies it, as any does.
            return context.run(self._group.create_task, coro, name=name)

        def finish(self) -> None:
            """Exit the group, which joins its tasks; raise what the host ended with."""
            self._closing.set()
            self._driver.run(asyncio.wait([self._host]))
            failure = _failure(self._host)
            if failure is not None:
                raise failure

        @staticmethod
        async def _keep_group_entered(
            entered: asyncio.Future[asyncio.TaskGroup],
            closing: asyncio.Event,
            stop_normal_work: Callable[[], None],
        ) -> None:
            async with asyncio.TaskGroup() as group:
                entered.set_result(group)
                try:
                    await closing.wait()
                except asyncio.CancelledError:
                    stop_normal_work()
                    raise

else:

    class _TaskGroupHost:
        """The tasks of the loop scope, without a task group: joined at finish."""

        @classmethod
        def start(cls, driver: Runner, stop_normal_work: Callable[[], None]) -> Self:
            return cls(driver)

        def __init__(self, driver: Runner) -> None:
            self._driver = driver
            self._tasks: set[asyncio.Task[object]] = set()

        def create_task(
            self,
            coro: Coroutine[Any, Any, _T],
            *,
            context: contextvars.Context,
            name: str | None,
        ) -> asyncio.Task[_T]:
            loop = self._driver.get_loop()
            task = context.run(loop.create_task, coro, name=name)
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
            return task

        def finish(self) -> None:
            if self._tasks:
                self._driver.run(asyncio.wait(list(self._tasks)))


@dataclass(frozen=True)
class _Returned(Generic[_T]):
    """A coroutine's return value, which may be None or an exception."""

    value: _T


class _Captured(Generic[_T]):
    """
    What a coroutine returned or raised, read once its task has ended.

    :meth:`run` is the coroutine as the body of a task: what it raises is
    kept for pytest instead of ending the task with it, which in a task
    group would cancel the other tasks. A cancellation ends the task as it
    does any task.
    """

    def __init__(self) -> None:
        self._result: _Returned[_T] | BaseException | None = None
        self._reported = False

    async def run(self, coro: Coroutine[Any, Any, _T]) -> None:
        try:
            self._result = _Returned(await coro)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            self._result = exc

    def read(self, task: asyncio.Task[object]) -> _T:
        """The coroutine's result, or how its task ended."""
        if isinstance(self._result, BaseException):
            # Raised while a cancellation was pending, it wins, as in any task.
            raise self._result
        task.result()  # raises if the task was cancelled
        if self._result is None:
            raise RuntimeError("The task ended without a result")
        return self._result.value

    def failure(self) -> BaseException | None:
        """What the coroutine raised, if anything."""
        return self._result if isinstance(self._result, BaseException) else None

    def report_unobserved(self, task: asyncio.Task[object]) -> None:
        """A done callback for a task nobody reads: report an exception to the loop."""
        if self._reported:
            return
        self._reported = True
        try:
            self.read(task)
        except asyncio.CancelledError:
            pass
        except BaseException as exc:
            _report_unobserved(task, exc)


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

    The task runs the setup, announces its result (:attr:`ready`), waits at
    the ``yield`` for pytest's request, then runs the teardown or closes the
    generator. Setup and teardown therefore share a task and a context, and
    a task group, cancel scope or timeout entered before the ``yield`` is
    exited by its own task. The setup is read while the task lives; the
    teardown is read once the task has ended.

    A cancellation of the task while it waits is consumed there: it stops
    the loop's normal work (``stop_normal_work``) and the task keeps
    waiting, so that pytest tears the fixtures down in its order and this
    fixture's teardown is what exits the scope. The cancellation is not
    replayed into the generator.
    """

    def __init__(
        self,
        gen: AsyncGenerator[_T],
        create_task: Callable[[Coroutine[Any, Any, None]], asyncio.Task[None]],
        *,
        stop_normal_work: Callable[[], None],
    ) -> None:
        self._gen = gen
        self._stop_normal_work = stop_normal_work
        self._teardown: _Captured[None] = _Captured()
        self._reported = False
        self.task = create_task(self._live())
        loop = self.task.get_loop()
        self.ready: asyncio.Future[FixtureSetup[_T]] = loop.create_future()
        self._action: asyncio.Future[_TeardownAction] = loop.create_future()

    def setup_result(self) -> FixtureSetup[_T]:
        """The setup's result; raises if it failed, or the task ended before it."""
        if self.ready.done():
            return self.ready.result()
        self.task.result()
        raise RuntimeError("The fixture task ended before its setup")

    @property
    def yielded(self) -> bool:
        """The setup reached the yield: the task waits there, or tears down."""
        return self.ready.done() and self.ready.exception() is None

    def setup_failure(self) -> BaseException | None:
        """
        What the setup raised, if an error.

        A cancellation is none, nor is a return before the yield: cancelled,
        the fixture ended as asked.
        """
        failure = _failure(self.ready) if self.ready.done() else None
        return None if isinstance(failure, StopAsyncIteration) else failure

    def teardown_failure(self) -> BaseException | None:
        """What the teardown raised, if anything."""
        return self._teardown.failure()

    def request_teardown(self) -> None:
        self._action.set_result(_TeardownAction.FINISH)

    def request_close(self) -> None:
        """Have the task close the generator; nobody reads the result."""
        if not self._action.done():
            self.task.add_done_callback(self.report_unobserved)
            self._action.set_result(_TeardownAction.CLOSE)

    def teardown_result(self) -> None:
        """Raises what the teardown raised, or how the task ended."""
        self._teardown.read(self.task)

    def report_unobserved(self, task: asyncio.Task[object]) -> None:
        """A done callback for a fixture nobody reads: report an exception."""
        if self._reported:
            return
        self._reported = True
        if self.yielded:
            self._teardown.report_unobserved(task)
        elif (failure := self.setup_failure()) is not None:
            _report_unobserved(task, failure)

    async def _live(self) -> None:
        try:
            value = await anext(self._gen)
        except BaseException as exc:
            self.ready.set_exception(exc)
            return
        self.ready.set_result(FixtureSetup(value, contextvars.copy_context()))
        while not self._action.done():
            try:
                await asyncio.wait([self._action])
            except asyncio.CancelledError:
                # Keep waiting: dependent fixtures are torn down first, and
                # this fixture's teardown, in this task, exits the scope.
                self._stop_normal_work()
        await self._teardown.run(self._tear_down(self._action.result()))

    async def _tear_down(self, action: _TeardownAction) -> None:
        if action is _TeardownAction.FINISH:
            try:
                await anext(self._gen)
            except StopAsyncIteration:
                return
            raise ValueError("Async generator fixture yielded more than once")
        elif action is _TeardownAction.CLOSE:
            await self._gen.aclose()
        else:
            assert_never(action)


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


def _report_unobserved(task: asyncio.Task[object], exc: BaseException) -> None:
    task.get_loop().call_exception_handler(
        {
            "message": "Exception from an async fixture or test after pytest "
            "stopped waiting for it",
            "exception": exc,
            "task": task,
        }
    )
