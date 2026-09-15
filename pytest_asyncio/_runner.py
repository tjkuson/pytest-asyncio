"""
Run the async fixtures and tests of an event loop scope, each in a task.

The test runner borrows an :class:`asyncio.Runner` and opens a task
group that owns every fixture and test task. Pytest runs each fixture setup,
test and fixture teardown as one synchronous call, which drives the loop
until that work is done and reads its result: a fixture's setup is ready
while its task lives on; a test, and a fixture's teardown, are done when
their task is. Closing the runner finishes the group and joins its tasks.
The separate pytest fixture owning the loop then closes it.

Two policies decide the exceptional paths. A cancellation reaching a
fixture as it waits at its yield (from a task group spanning the yield
whose child failed, say) ends the loop's normal work: the test or setup
running at the time is cancelled, and only teardowns run until the loop
closes. An interruption of a synchronous call (SIGINT, or a callback
raising) cancels the task waited for and waits for it to end before
raising, so that its cleanup runs while the fixtures it uses are alive
(asyncio.Runner does as much for SIGINT alone); what the cleanup raised is
raised instead of the interruption. A second interruption stops waiting.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import enum
import sys
from collections.abc import AsyncGenerator, Callable, Coroutine, Generator
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

if sys.version_info >= (3, 11):
    from asyncio import Runner
    from typing import Self, assert_never
else:
    from backports.asyncio.runner import Runner
    from typing_extensions import Self, assert_never

_T = TypeVar("_T")

_REFUSED_MESSAGE = (
    "This event loop no longer accepts new tests or fixture setups: an async "
    "generator fixture was cancelled while it waited at its yield, or "
    "pytest-asyncio's own task was. Fixture teardowns still run."
)


class TestRunner:
    """Run async fixtures and tests in a task group on a borrowed event loop."""

    def __init__(self, asyncio_runner: Runner) -> None:
        self._asyncio_runner = asyncio_runner
        self._teardown_only = False
        # The test or fixture setup being waited for, cancelled with a
        # fixture waiting at its yield (see _stop_normal_work).
        self._active: asyncio.Task[object] | None = None
        # The generator fixtures whose tasks are alive: closed at close.
        self._fixtures: set[FixtureTask[Any]] = set()
        # The tasks pytest stopped waiting for: cancelled once more at close.
        self._abandoned: set[asyncio.Task[object]] = set()
        self._task_group_host = _TaskGroupHost.start(
            asyncio_runner, self._stop_normal_work
        )

    def close(self) -> None:
        """Finish the group and join its tasks before the owner closes the loop."""
        for fixture in list(self._fixtures):
            fixture.request_close()
        for task in self._abandoned:
            task.cancel()
        self._task_group_host.finish()

    def get_loop(self) -> asyncio.AbstractEventLoop:
        return self._asyncio_runner.get_loop()

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
        captured_coro = captured.run(coro)
        try:
            task = self._task_group_host.create_task(
                captured_coro, context=context, name=name
            )
        except BaseException:
            captured_coro.close()
            coro.close()
            raise
        # A task cancelled before its first step never started the coroutine.
        task.add_done_callback(lambda _: coro.close())
        with self._as_active(task):
            interruption = self._drive_until(task)
            if interruption is None:
                return captured.read(task)
            task.cancel()
            self._join_or_abandon(task, captured.report_unobserved)
            failure = captured.failure()
            raise interruption if failure is None else failure

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
            lambda live: self._task_group_host.create_task(
                live, context=context, name=name
            ),
            loop=self.get_loop(),
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
                    fixture.task, fixture.report_unobserved, ready=fixture.ready
                )
        # Interrupted, the fixture is not handed to pytest, which will not
        # tear it down: it ends here, before pytest tears down the fixtures
        # it depends on. One that recovered and yielded is torn down, as
        # pytest would have.
        if fixture.yielded:
            fixture.request_teardown()
            self._join_or_abandon(fixture.task, fixture.report_unobserved)
            failure = fixture.teardown_failure()
        else:
            failure = fixture.interrupted_setup_failure()
        raise interruption if failure is None else failure

    def finish_fixture(self, fixture: FixtureTask[Any]) -> None:
        """Run the fixture from its yield to its end, in its task."""
        fixture.request_teardown()
        interruption = self._drive_until(fixture.task)
        if interruption is None:
            fixture.teardown_result()
            return
        fixture.task.cancel()
        self._join_or_abandon(fixture.task, fixture.report_unobserved)
        failure = fixture.teardown_failure()
        raise interruption if failure is None else failure

    def _admit(self, work: Coroutine[Any, Any, Any] | AsyncGenerator[Any]) -> None:
        """Refuse new work before it has a task, or a place as the active one."""
        if self._teardown_only:
            _close(work)
            raise RuntimeError(_REFUSED_MESSAGE)

    @contextlib.contextmanager
    def _as_active(self, task: asyncio.Task[object]) -> Generator[None]:
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
        *,
        ready: asyncio.Future[Any] | None = None,
    ) -> None:
        """
        Drive the loop until the cancelled task ends.

        A second interruption stops waiting: the task is cancelled again and
        left to end on its own, ``unobserved`` reports what it ends with, and
        close cancels it once more. A setup's ``ready`` also ends the wait:
        a cancelled setup that recovers and yields parks its task there.
        """
        waiters: list[asyncio.Future[Any]] = [task]
        if ready is not None:
            waiters.append(ready)
        interruption = self._drive_until(*waiters)
        if interruption is not None:
            task.add_done_callback(unobserved)
            task.cancel()
            self._abandoned.add(task)
            task.add_done_callback(self._abandoned.discard)
            raise interruption

    def _drive_until(self, *waiters: asyncio.Future[Any]) -> BaseException | None:
        """Run the loop until a waiter is done; return the interruption, if any."""
        # asyncio.Runner runs the wait in a task of its own, which a callback
        # raising leaves behind, pending: released ends it with this call.
        released: asyncio.Future[None] = self.get_loop().create_future()
        try:
            self._asyncio_runner.run(
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
        A task of the runner's own keeping a task group entered from
        :meth:`start` to :meth:`finish`, so that the fixtures' and tests'
        tasks are its children. A child only ends with an exception if the
        runner has a bug (see _Captured and FixtureTask); the group raises
        it at finish. A cancellation of the host stops the loop's normal
        work, then cancels the children, as in any group.
        """

        @classmethod
        def start(
            cls, asyncio_runner: Runner, stop_normal_work: Callable[[], None]
        ) -> Self:
            """Start the host and drive the loop until the group is entered."""
            loop = asyncio_runner.get_loop()
            closing = asyncio.Event()
            entered: asyncio.Future[asyncio.TaskGroup] = loop.create_future()
            keep_entered = cls._keep_group_entered(entered, closing, stop_normal_work)
            host = loop.create_task(keep_entered, name="pytest-asyncio")
            try:
                startup: list[asyncio.Future[Any]] = [entered, host]
                asyncio_runner.run(
                    asyncio.wait(startup, return_when=asyncio.FIRST_COMPLETED)
                )
                if not entered.done():
                    host.result()  # the host ended before entering: raises how
                    raise RuntimeError("The task group host ended before entering")
                return cls(asyncio_runner, host, entered.result(), closing)
            except BaseException:
                closing.set()
                try:
                    # Joining the existing task does not need a working task factory.
                    loop.run_until_complete(host)
                except asyncio.CancelledError:
                    # Keep the startup error when joining an already-cancelled host.
                    if not host.cancelled():
                        raise
                raise

        def __init__(
            self,
            asyncio_runner: Runner,
            host: asyncio.Task[None],
            group: asyncio.TaskGroup,
            closing: asyncio.Event,
        ) -> None:
            self._asyncio_runner = asyncio_runner
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
            self._asyncio_runner.run(asyncio.wait([self._host]))
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
        def start(
            cls, asyncio_runner: Runner, stop_normal_work: Callable[[], None]
        ) -> Self:
            return cls(asyncio_runner)

        def __init__(self, asyncio_runner: Runner) -> None:
            self._asyncio_runner = asyncio_runner
            self._tasks: set[asyncio.Task[object]] = set()

        def create_task(
            self,
            coro: Coroutine[Any, Any, _T],
            *,
            context: contextvars.Context,
            name: str | None,
        ) -> asyncio.Task[_T]:
            loop = self._asyncio_runner.get_loop()
            task = context.run(loop.create_task, coro, name=name)
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
            return task

        def finish(self) -> None:
            if self._tasks:
                self._asyncio_runner.run(asyncio.wait(list(self._tasks)))


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
    generator: a task group, cancel scope or timeout entered before the
    ``yield`` is exited by the task that entered it. The setup is read while
    the task lives; the teardown once the task has ended.
    """

    def __init__(
        self,
        gen: AsyncGenerator[_T],
        create_task: Callable[[Coroutine[Any, Any, None]], asyncio.Task[None]],
        *,
        loop: asyncio.AbstractEventLoop,
        stop_normal_work: Callable[[], None],
    ) -> None:
        self._gen = gen
        self._stop_normal_work = stop_normal_work
        self._teardown: _Captured[None] = _Captured()
        self._reported = False
        self.ready: asyncio.Future[FixtureSetup[_T]] = loop.create_future()
        self._action: asyncio.Future[_TeardownAction] = loop.create_future()
        self.task = create_task(self._live())

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

    def interrupted_setup_failure(self) -> BaseException | None:
        """
        What an interrupted setup raised, if a user error.

        A setup that ends without yielding gives pytest the interruption,
        not the generator's StopAsyncIteration, whether it returned after
        the cancellation or before it arrived; the cancellation itself is
        the interruption's own.
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
        elif (failure := self.interrupted_setup_failure()) is not None:
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
                # A cancellation at the yield (a task group's, say) is
                # consumed here, not replayed into the generator: the task
                # keeps waiting, so that pytest tears the fixtures down in
                # its order and this fixture's teardown exits the scope.
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
