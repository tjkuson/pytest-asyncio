"""
Run async fixtures and tests in native tasks owned by a TaskGroup.

Pytest calls setup, test and teardown synchronously. A fixture's task remains
alive between these calls so its contexts are entered and exited in one task.
Cancellation waits for dependent teardown before unwinding that fixture.
An interrupted call cancels and joins its task while its resources remain alive;
a second interruption returns control to pytest without releasing task ownership.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import enum
from asyncio import Runner
from collections.abc import AsyncGenerator, Callable, Coroutine, Generator
from dataclasses import dataclass
from typing import Any, Generic, Self, TypeVar, assert_never

_T = TypeVar("_T")


class TaskGroupLoopOwner:
    """Close the task group and its event loop under one owner."""

    def __init__(self, asyncio_runner: Runner) -> None:
        self._asyncio_runner = asyncio_runner
        self._task_group_runner: TaskGroupRunner | None = None

    def get_loop(self) -> asyncio.AbstractEventLoop:
        return self._asyncio_runner.get_loop()

    def get_task_group_runner(self) -> TaskGroupRunner:
        """Start on demand so pytest can cache a group startup failure."""
        if self._task_group_runner is None:
            self._task_group_runner = TaskGroupRunner(self._asyncio_runner)
        return self._task_group_runner

    def close(self) -> None:
        task_group_runner, self._task_group_runner = self._task_group_runner, None
        try:
            if task_group_runner is not None and not self.get_loop().is_closed():
                task_group_runner.close()
        finally:
            # An interrupt can stop pytest before another fixture finalizer runs.
            self._asyncio_runner.close()


class TaskGroupRunner:
    """Run async fixtures and tests in a task group on a borrowed event loop."""

    def __init__(self, asyncio_runner: Runner) -> None:
        self._asyncio_runner = asyncio_runner
        self._refusal_reason: str | None = None
        self._cancel_active_operation: Callable[[], object] | None = None
        self._live_fixtures: set[FixtureTask[Any]] = set()
        # Abandoned waits leave their tasks owned until group shutdown.
        self._abandoned: set[asyncio.Task[object]] = set()
        self._task_group_host = _TaskGroupHost.start(
            asyncio_runner, self._refuse_new_work
        )

    def close(self) -> None:
        """Finish the group and join its tasks before the owner closes the loop."""
        for fixture in list(self._live_fixtures):
            fixture.request_close()
        for task in self._abandoned:
            task.cancel()
        self._task_group_host.finish()

    def get_loop(self) -> asyncio.AbstractEventLoop:
        return self._asyncio_runner.get_loop()

    @property
    def refusal_reason(self) -> str | None:
        return self._refusal_reason

    def run(
        self,
        coro: Coroutine[object, object, _T],
        *,
        context: contextvars.Context,
        name: str,
    ) -> _T:
        """Run the coroutine in a task of its own and return its result."""
        if self._refusal_reason is not None:
            __tracebackhide__ = True
            coro.close()
            raise RuntimeError(self._refusal_reason)
        captured: _Captured[_T] = _Captured(self.get_loop())
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
        with self._cancellable_operation(task.cancel):
            interruption = self._drive_until(task)
            if interruption is None:
                return captured.read(task)
            task.cancel()
            self._join_or_abandon(task, captured.report_unobserved)
            failure = captured.failure()
            raise _interruption_or_cleanup_error(interruption, failure)

    def start_fixture(
        self,
        gen: AsyncGenerator[_T],
        *,
        context: contextvars.Context,
        name: str,
    ) -> tuple[FixtureTask[_T], FixtureSetup[_T]]:
        """Run the fixture up to its yield, in a task of its own."""
        if self._refusal_reason is not None:
            raise RuntimeError(self._refusal_reason)
        fixture = FixtureTask(
            gen,
            lambda live: self._task_group_host.create_task(
                live, context=context, name=name
            ),
            loop=self.get_loop(),
            refuse_new_work=lambda: self._refuse_new_work(
                f"Async fixture {name!r} was cancelled while waiting for teardown."
            ),
        )
        self._live_fixtures.add(fixture)
        fixture.task.add_done_callback(lambda _: self._live_fixtures.discard(fixture))
        with self._cancellable_operation(fixture.cancel_setup):
            interruption = self._drive_until(fixture.task, fixture.ready)
            if interruption is None:
                return fixture, fixture.setup_result()
            if not fixture.ready.done():
                fixture.task.cancel()
                self._join_or_abandon(
                    fixture.task, fixture.report_unobserved, ready=fixture.ready
                )
        # Pytest never receives an interrupted fixture. End it here while its
        # dependencies still exist, tearing down a recovered setup normally.
        if fixture.yielded:
            fixture.request_teardown()
            self._join_or_abandon(fixture.task, fixture.report_unobserved)
        raise _interruption_or_cleanup_error(interruption, fixture.failure())

    def finish_fixture(self, fixture: FixtureTask[_T]) -> None:
        """Run the fixture from its yield to its end, in its task."""
        fixture.request_teardown()
        interruption = self._drive_until(fixture.task)
        if interruption is None:
            fixture.teardown_result()
            return
        fixture.task.cancel()
        self._join_or_abandon(fixture.task, fixture.report_unobserved)
        raise _interruption_or_cleanup_error(interruption, fixture.failure())

    @contextlib.contextmanager
    def _cancellable_operation(self, cancel: Callable[[], object]) -> Generator[None]:
        """Cancel this test or pending setup if a live fixture fails."""
        self._cancel_active_operation = cancel
        try:
            yield
        finally:
            self._cancel_active_operation = None

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
        # An interrupt can leave Runner.run's wait task pending. Release it
        # when this call ends, without cancelling the work it was observing.
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

    def _refuse_new_work(self, reason: str) -> None:
        if self._refusal_reason is not None:
            return
        self._refusal_reason = (
            f"{reason} This event loop no longer accepts new async tests or fixture "
            "setups. Fixture teardowns still run."
        )
        if self._cancel_active_operation is not None:
            self._cancel_active_operation()


class _TaskGroupHost:
    """
    Keep a TaskGroup entered between pytest's synchronous calls.

    This root task owns the group. The loop owner joins it before closing
    the loop; fixture and test tasks are created only through the group.
    """

    @classmethod
    def start(
        cls,
        asyncio_runner: Runner,
        refuse_new_work: Callable[[str], None],
    ) -> Self:
        """Start the host and drive the loop until the group is entered."""
        loop = asyncio_runner.get_loop()
        closing = asyncio.Event()
        entered: asyncio.Future[asyncio.TaskGroup] = loop.create_future()
        keep_entered = cls._keep_group_entered(entered, closing, refuse_new_work)
        try:
            host = loop.create_task(keep_entered)
        except BaseException:
            keep_entered.close()
            raise
        host.set_name("pytest-asyncio")
        try:
            startup: list[asyncio.Future[Any]] = [entered, host]
            asyncio_runner.run(
                asyncio.wait(startup, return_when=asyncio.FIRST_COMPLETED)
            )
            if not entered.done():
                host.result()  # Propagate startup failure before reading entered.
                raise AssertionError("The task group host ended before entering")
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
        coro: Coroutine[object, object, _T],
        *,
        context: contextvars.Context,
        name: str,
    ) -> asyncio.Task[_T]:
        # Hypothesis reuses the supplied context for each example.
        task = context.copy().run(self._group.create_task, coro)
        task.set_name(name)
        return task

    def finish(self) -> None:
        """Join the host and report errors other than cancellation."""
        self._closing.set()
        self._asyncio_runner.run(asyncio.wait([self._host]))
        failure = _non_cancellation_exception(self._host)
        if failure is not None:
            raise failure

    @staticmethod
    async def _keep_group_entered(
        entered: asyncio.Future[asyncio.TaskGroup],
        closing: asyncio.Event,
        refuse_new_work: Callable[[str], None],
    ) -> None:
        async with asyncio.TaskGroup() as group:
            entered.set_result(group)
            try:
                await closing.wait()
            except asyncio.CancelledError:
                refuse_new_work(
                    "The task group running async fixtures and tests was cancelled."
                )
                raise


class _Captured(Generic[_T]):
    """
    Keep user errors for pytest instead of cancelling unrelated group children.

    Cancellation propagates normally. Read the outcome only after joining the task.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._outcome: asyncio.Future[_T] = loop.create_future()

    async def run(self, coro: Coroutine[object, object, _T]) -> None:
        try:
            result = await coro
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            self._outcome.set_exception(exc)
        else:
            self._outcome.set_result(result)

    def read(self, task: asyncio.Task[object]) -> _T:
        """The coroutine's result, or how its task ended."""
        failure = self.failure()
        if failure is not None:
            # Raised while a cancellation was pending, it wins, as in any task.
            raise failure
        task.result()  # raises if the task was cancelled
        return self._outcome.result()

    def failure(self) -> BaseException | None:
        """What the coroutine raised, if anything."""
        return self._outcome.exception() if self._outcome.done() else None

    def report_unobserved(self, task: asyncio.Task[object]) -> None:
        """Report a captured failure; the TaskGroup handles uncaught task errors."""
        if (failure := self.failure()) is not None:
            _report_unobserved(task, failure)


@dataclass(frozen=True)
class FixtureSetup(Generic[_T]):
    """A successful fixture setup: the value yielded, and the context then."""

    value: _T
    context: contextvars.Context


class _TeardownAction(enum.Enum):
    FINISH = enum.auto()  # run the fixture from its yield to its end
    CLOSE = enum.auto()  # close the generator: the test runner is closing


class FixtureTask(Generic[_T]):
    """
    Run setup and teardown in one task, waiting for pytest between them.

    A cancelled fixture stays alive until pytest has torn down its dependents.
    """

    def __init__(
        self,
        gen: AsyncGenerator[_T],
        create_task: Callable[[Coroutine[object, object, None]], asyncio.Task[None]],
        *,
        loop: asyncio.AbstractEventLoop,
        refuse_new_work: Callable[[], None],
    ) -> None:
        self._gen = gen
        self._refuse_new_work = refuse_new_work
        self._teardown: _Captured[None] = _Captured(loop)
        self._reported = False
        self.ready: asyncio.Future[FixtureSetup[_T]] = loop.create_future()
        self._action: asyncio.Future[_TeardownAction] = loop.create_future()
        run_fixture = self._run_fixture()
        try:
            self.task = create_task(run_fixture)
        except BaseException:
            run_fixture.close()
            raise

    def setup_result(self) -> FixtureSetup[_T]:
        """The setup's result; raises if it failed, or the task ended before it."""
        if self.ready.done():
            return self.ready.result()
        self.task.result()
        raise AssertionError("The fixture task ended before its setup")

    def cancel_setup(self) -> None:
        # Once setup is published, cancellation must leave teardown to pytest.
        if not self.ready.done():
            self.task.cancel()

    @property
    def yielded(self) -> bool:
        """The setup reached the yield: the task waits there, or tears down."""
        return self.ready.done() and self.ready.exception() is None

    def failure(self) -> BaseException | None:
        """What an interrupted setup or teardown raised, other than cancellation."""
        if self.yielded:
            return self._teardown.failure()
        failure = _non_cancellation_exception(self.ready) if self.ready.done() else None
        # A cancelled setup that returns without yielding keeps the interruption.
        return None if isinstance(failure, StopAsyncIteration) else failure

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
        if (failure := self.failure()) is not None:
            _report_unobserved(task, failure)

    async def _run_fixture(self) -> None:
        try:
            value = await anext(self._gen)
        except asyncio.CancelledError as exc:
            self.ready.set_exception(exc)
            raise
        except BaseException as exc:
            self.ready.set_exception(exc)
            return
        self.ready.set_result(FixtureSetup(value, contextvars.copy_context()))
        cancellation: asyncio.CancelledError | None = None
        while not self._action.done():
            try:
                # Directly awaiting _action would let cancellation cancel the
                # handoff itself, preventing pytest from requesting teardown.
                await asyncio.wait([self._action])
            except asyncio.CancelledError as exc:
                # Keep resources alive until pytest tears down their dependents.
                # Teardown delivers the exception to the fixture's own scope.
                if cancellation is None:
                    cancellation = exc.with_traceback(None)
                if not self._action.done():
                    self._refuse_new_work()
        await self._teardown.run(self._tear_down(self._action.result(), cancellation))

    async def _tear_down(
        self, action: _TeardownAction, cancellation: asyncio.CancelledError | None
    ) -> None:
        if action is _TeardownAction.FINISH:
            try:
                if cancellation is None:
                    await anext(self._gen)
                else:
                    await self._gen.athrow(cancellation)
            except StopAsyncIteration:
                return
            raise ValueError("Async generator fixture yielded more than once")
        elif action is _TeardownAction.CLOSE:
            await self._gen.aclose()
        else:
            assert_never(action)


def _interruption_or_cleanup_error(
    interruption: BaseException, failure: BaseException | None
) -> BaseException:
    if failure is None:
        return interruption
    failure.add_note(
        f"Raised during cleanup after pytest-asyncio received: {interruption!r}"
    )
    return failure


def _non_cancellation_exception(future: asyncio.Future[_T]) -> BaseException | None:
    """Read a finished future's exception, treating cancellation separately."""
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
