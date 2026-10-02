"""
Run async fixtures and tests in native tasks on a borrowed event loop.

Pytest calls setup, test and teardown synchronously. A fixture's task remains
alive between these calls so its contexts are entered and exited in one task.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import functools
from asyncio import Runner
from collections.abc import AsyncGenerator, Callable, Coroutine, Generator, Iterable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

_T = TypeVar("_T")

_WAIT_CANCELLED = (
    "pytest-asyncio's internal wait was cancelled, e.g. by code that cancels "
    "every task. Cancel only tasks that your code created."
)


class TaskRunner:
    """
    Own the tasks that run async fixtures and tests on an asyncio.Runner's loop.

    Every task created here is joined and its outcome read. The call that
    creates a task joins it, except that ``finish_fixture`` joins a fixture task
    pytest holds. ``close`` joins the rest; if ``close`` is interrupted, closing
    the asyncio.Runner cancels them.
    """

    def __init__(self, asyncio_runner: Runner) -> None:
        self._asyncio_runner = asyncio_runner
        self._loop = asyncio_runner.get_loop()
        self._cancel_active_operation: Callable[[], object] | None = None
        self._live_fixtures: set[FixtureTask[Any]] = set()
        self._tasks_to_join_on_close: set[asyncio.Task[None]] = set()
        # Whether the last run of the loop raised, which can leave a stale stop
        # queued; see _wait.
        self._run_cut_short = False

    def close(self) -> None:
        """Before the loop closes, cancel unfinalized fixtures and join what is left."""
        __tracebackhide__ = True
        if self._loop.is_closed():
            # A test closed the loop: nothing can run, and the loop fixture
            # reports it.
            return
        for fixture in self._live_fixtures:
            fixture.cancel_unfinished()
        unfinished = self._tasks_to_join_on_close.union(
            fixture.task for fixture in self._live_fixtures
        )
        if unfinished:
            self._wait(unfinished, asyncio.ALL_COMPLETED)

    @property
    def refusal_reason(self) -> str | None:
        """Why new work is refused: a fixture in use was cancelled at its yield."""
        for fixture in self._live_fixtures:
            if fixture.cancelled_while_held:
                return (
                    f"Async fixture {fixture.name!r} was cancelled while waiting for "
                    "teardown. Until it is torn down, this event loop does not accept "
                    "new async tests or fixture setups."
                )
        return None

    def run(
        self,
        func: Callable[[], Coroutine[object, object, _T]],
        *,
        context: contextvars.Context,
        name: str,
    ) -> _T:
        """Run the async function in a task of its own and return its result."""
        __tracebackhide__ = True
        if (refusal_reason := self.refusal_reason) is not None:
            raise RuntimeError(refusal_reason)
        outcome: _Outcome[_T] = _Outcome(self._loop)
        task = self._create_task(outcome.run(func), context=context, name=name)
        with self._cancel_on_held_fixture_cancellation(task.cancel):
            interruption = self._drive_until(task)
            if interruption is None:
                return outcome.result(task)
            task.cancel()
            self._join_after_interruption(task, outcome.report_unobserved, interruption)
            raise _interruption_or_cleanup_error(interruption, outcome.failure(task))

    def start_fixture(
        self,
        gen: AsyncGenerator[_T],
        *,
        context: contextvars.Context,
        name: str,
    ) -> tuple[FixtureTask[_T], FixtureSetup[_T]]:
        """Run the fixture up to its yield, in a task of its own."""
        __tracebackhide__ = True
        if (refusal_reason := self.refusal_reason) is not None:
            raise RuntimeError(refusal_reason)
        fixture = FixtureTask(
            gen,
            functools.partial(self._create_task, context=context, name=name),
            name=name,
            loop=self._loop,
            on_cancelled_while_held=self._on_held_fixture_cancelled,
        )
        self._live_fixtures.add(fixture)
        fixture.task.add_done_callback(lambda _: self._live_fixtures.discard(fixture))
        with self._cancel_on_held_fixture_cancellation(fixture.cancel_setup):
            interruption = self._drive_until(fixture.task, fixture.ready)
            if interruption is None:
                return fixture, fixture.setup_result()
            # Pytest never receives an interrupted fixture. If it yields, it is
            # torn down at once, while its dependencies still exist.
            fixture.request_teardown()
            fixture.cancel_setup()
            self._join_after_interruption(
                fixture.task, fixture.report_unobserved, interruption
            )
            raise _interruption_or_cleanup_error(interruption, fixture.failure())

    def finish_fixture(self, fixture: FixtureTask[_T]) -> None:
        """Run the fixture from its yield to its end, in its task."""
        __tracebackhide__ = True
        fixture.request_teardown()
        interruption = self._drive_until(fixture.task)
        if interruption is None:
            fixture.teardown_result()
            return
        fixture.task.cancel()
        self._join_after_interruption(
            fixture.task, fixture.report_unobserved, interruption
        )
        raise _interruption_or_cleanup_error(interruption, fixture.failure())

    def _create_task(
        self,
        coro: Coroutine[object, object, None],
        *,
        context: contextvars.Context,
        name: str,
    ) -> asyncio.Task[None]:
        __tracebackhide__ = True
        try:
            task = self._loop.create_task(coro, context=context)
        except BaseException:
            coro.close()  # A failing task factory need not close it.
            raise
        task.set_name(name)  # Existing task factories need not accept names.
        return task

    @contextlib.contextmanager
    def _cancel_on_held_fixture_cancellation(
        self, cancel: Callable[[], object]
    ) -> Generator[None]:
        """While the operation runs, cancel it once if a held fixture is cancelled."""
        self._cancel_active_operation = cancel
        try:
            yield
        finally:
            self._cancel_active_operation = None

    def _on_held_fixture_cancelled(self) -> None:
        if self._cancel_active_operation is not None:
            self._cancel_active_operation()

    def _join_after_interruption(
        self,
        task: asyncio.Task[None],
        report_unobserved: Callable[[asyncio.Task[None]], None],
        interruption: BaseException,
    ) -> None:
        """
        Wait for the interrupted task to end.

        After a second interruption, cancel it again and leave it to ``close``;
        until then it runs whenever the loop does.
        """
        __tracebackhide__ = True
        second_interruption = self._drive_until(task)
        if second_interruption is None:
            return
        task.add_done_callback(report_unobserved)
        task.cancel()
        self._tasks_to_join_on_close.add(task)
        task.add_done_callback(self._tasks_to_join_on_close.discard)
        if isinstance(second_interruption, asyncio.CancelledError):
            # A cancelled wait is no request to stop, so it must not replace the
            # first interruption: Ctrl-C would become a test failure. Note on the
            # first why the join stopped, unless _wait has noted it already.
            if not isinstance(interruption, asyncio.CancelledError):
                interruption.add_note(_WAIT_CANCELLED)
            raise interruption
        raise _interruption_or_cleanup_error(interruption, second_interruption)

    def _drive_until(self, *waiters: asyncio.Future[Any]) -> BaseException | None:
        """Run the loop until a waiter is done; return the interruption, if any."""
        __tracebackhide__ = True
        try:
            self._wait(waiters, asyncio.FIRST_COMPLETED)
        except BaseException as interruption:
            return interruption
        return None

    def _wait(self, waiters: Iterable[asyncio.Future[Any]], return_when: str) -> None:
        """
        Run the loop until the waiters are done, as asyncio.wait defines.

        Runner.run's SIGINT handling cancels this wait, not the test or fixture:
        Ctrl-C normally reaches the caller as KeyboardInterrupt, and the caller
        cancels its task. Every loop run by TaskRunner goes through here. The
        one later run, asyncio.Runner.close() after TaskRunner.close(), can
        still meet a stale stop, but only if the wait in TaskRunner.close() is
        interrupted.

        Runner.run creates the wait's task, which is not joined here: if an
        exception escapes the run before it finishes, it ends on its own, and an
        error from it, such as one from a task factory that wraps it, is
        reported only when it is garbage collected, as "Task exception was never
        retrieved".
        """
        __tracebackhide__ = True
        if self._run_cut_short:
            # Work around a CPython bug: run_until_complete() stops the loop
            # from a done callback of its future. If an exception escapes the
            # run after that callback is scheduled but before it runs, it stays
            # queued and stops the next run early. One loop iteration runs it,
            # along with any other ready callbacks, outside Runner.run's SIGINT
            # handler. Remove this once the oldest supported Python has a fix.
            self._loop.stop()
            self._loop.run_forever()
        self._run_cut_short = True
        try:
            self._asyncio_runner.run(asyncio.wait(waiters, return_when=return_when))
        except asyncio.CancelledError as cancelled:
            # Runner.run turns its own SIGINT cancellation into KeyboardInterrupt,
            # so other code cancelled the wait.
            cancelled.add_note(_WAIT_CANCELLED)
            raise
        self._run_cut_short = False


@dataclass(frozen=True)
class FixtureSetup(Generic[_T]):
    """A successful fixture setup: its value, and a copy of the context it ended in."""

    value: _T
    context: contextvars.Context


class FixtureTask(Generic[_T]):
    """
    Run setup and teardown in one task, waiting for pytest between them.

    Constructing it starts the task. A cancelled fixture stays alive until
    pytest has torn down its dependents.
    """

    def __init__(
        self,
        gen: AsyncGenerator[_T],
        create_task: Callable[[Coroutine[object, object, None]], asyncio.Task[None]],
        *,
        name: str,
        loop: asyncio.AbstractEventLoop,
        on_cancelled_while_held: Callable[[], None],
    ) -> None:
        __tracebackhide__ = True
        self._gen = gen
        self.name = name
        self._on_cancelled_while_held = on_cancelled_while_held
        # Set if cancelled after setup and before teardown is requested.
        self.cancelled_while_held = False
        self.ready: asyncio.Future[FixtureSetup[_T]] = loop.create_future()
        self._teardown_requested: asyncio.Future[None] = loop.create_future()
        self._outcome: _Outcome[None] = _Outcome(loop)
        self.task = create_task(self._outcome.run(self._run_fixture))

    def setup_result(self) -> FixtureSetup[_T]:
        """Return what the fixture yielded, or raise why it did not."""
        __tracebackhide__ = True
        if not self.ready.done():
            # The task ended before the fixture yielded, so it failed.
            self._outcome.result(self.task)
        return self.ready.result()

    def cancel_setup(self) -> None:
        # Once setup is published, cancellation must leave teardown to pytest.
        if not self.ready.done():
            self.task.cancel()

    def failure(self) -> BaseException | None:
        """What an interrupted setup or teardown raised, other than cancellation."""
        failure = self._outcome.failure(self.task)
        # Returning without yielding is a valid response to a cancelled setup.
        return None if isinstance(failure, StopAsyncIteration) else failure

    def request_teardown(self) -> None:
        self._teardown_requested.set_result(None)

    def cancel_unfinished(self) -> None:
        """Cancel a fixture pytest never finalized; nobody reads the result."""
        if not self._teardown_requested.done():
            self.task.add_done_callback(self.report_unobserved)
            self._teardown_requested.set_result(None)
            self.task.cancel()

    def teardown_result(self) -> None:
        """Raise the teardown's error or the task's cancellation, if any."""
        __tracebackhide__ = True
        self._outcome.result(self.task)

    def report_unobserved(self, task: asyncio.Task[None]) -> None:
        if (failure := self.failure()) is not None:
            _report_failure(task, failure)

    async def _run_fixture(self) -> None:
        value = await anext(self._gen)
        self.ready.set_result(FixtureSetup(value, contextvars.copy_context()))
        cancellation = await self._wait_for_teardown_request()
        try:
            if cancellation is None:
                await anext(self._gen)
            else:
                await self._gen.athrow(cancellation)
        except StopAsyncIteration:
            return
        try:
            raise ValueError(
                f"Async generator fixture {self.name!r} didn't stop. Yield only once."
            )
        finally:
            # Unwind the fixture's contexts in its own task, as contextlib does.
            await self._gen.aclose()

    async def _wait_for_teardown_request(self) -> asyncio.CancelledError | None:
        """Wait until teardown is requested; return the first cancellation, if any."""
        cancellation: asyncio.CancelledError | None = None
        while not self._teardown_requested.done():
            try:
                # Awaiting the Future directly would let a cancellation cancel it,
                # so pytest could not request teardown.
                await asyncio.wait([self._teardown_requested])
            except asyncio.CancelledError as exc:
                # Keep the first cancellation for the yield, as asyncio.TaskGroup
                # does while it waits for its tasks: delivering it now would
                # close resources that the cancelled test's cleanup and dependent
                # teardown still use. Later ones are caught without uncancel(),
                # so the cancellation count stays intact for the scopes unwinding
                # at the yield.
                if cancellation is not None:
                    continue
                # Thrown in at the yield, it should not carry this wait's frames.
                cancellation = exc.with_traceback(None)
                if not self._teardown_requested.done():
                    self.cancelled_while_held = True
                    self._on_cancelled_while_held()
        return cancellation


class _Outcome(Generic[_T]):
    """
    What an async function returned, or the KeyboardInterrupt or SystemExit it raised.

    asyncio re-raises those two out of the event loop before the task's done
    callbacks run, where they would pass for an interruption from outside the
    task. Stored here, they stay the function's own error.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._future: asyncio.Future[_T] = loop.create_future()

    async def run(self, func: Callable[[], Coroutine[object, object, _T]]) -> None:
        __tracebackhide__ = True
        try:
            result = await func()
        except (KeyboardInterrupt, SystemExit) as exc:
            self._future.set_exception(exc)
        else:
            self._future.set_result(result)

    def result(self, task: asyncio.Task[None]) -> _T:
        """Return the function's result, or raise what it or its task raised."""
        __tracebackhide__ = True
        if (failure := self.failure(task)) is not None:
            raise failure
        task.result()  # Raises a cancellation, even if the function returned.
        return self._future.result()

    def failure(self, task: asyncio.Task[None]) -> BaseException | None:
        """How the finished task failed, other than by cancellation."""
        raised = self._future.exception() if self._future.done() else None
        if task.cancelled():
            # run() returns after storing the exception, so a cancellation
            # pending then cancels the task; natively, the exception would win.
            return raised
        failure = task.exception()
        if failure is None:
            return raised
        if raised is not None and failure.__context__ is None:
            # A task factory's wrapper raised this after the stored exception,
            # which natively would be in its chain. Keep any chain it has.
            failure.__context__ = raised
        return failure

    def report_unobserved(self, task: asyncio.Task[None]) -> None:
        if (failure := self.failure(task)) is not None:
            _report_failure(task, failure)


def _interruption_or_cleanup_error(
    interruption: BaseException, failure: BaseException | None
) -> BaseException:
    if failure is None:
        return interruption
    failure.add_note(
        f"Raised during cleanup after pytest-asyncio received: {interruption!r}"
    )
    return failure


def _report_failure(task: asyncio.Task[object], exc: BaseException) -> None:
    task.get_loop().call_exception_handler(
        {
            "message": "Exception from an async fixture or test after "
            "pytest-asyncio stopped waiting for it",
            "exception": exc,
            "task": task,
        }
    )
