"""
Run async fixtures and tests in native tasks on a borrowed event loop.

Pytest calls setup, test and teardown synchronously. A fixture's task remains
alive between these calls so its contexts are entered and exited in one task.
Cancellation waits for dependent teardown before unwinding that fixture.
An interrupted call cancels and joins its task while its resources remain alive;
a second interruption returns control to pytest, and closing joins what is left.
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
    Run async fixtures and tests in tasks of their own on a borrowed loop.

    Every task created here is joined, reading the task's own outcome. The
    call that creates a task joins it, except that a fixture task pytest holds
    is joined by ``finish_fixture``. ``close`` joins the rest: tasks abandoned
    after a second interruption, kept in ``_abandoned``, and fixtures pytest
    never finalized, still in ``_live_fixtures``. If ``close`` is interrupted,
    closing the asyncio.Runner cancels what is left.

    asyncio.Runner.run creates and owns the task that runs each wait, so it is
    not joined here: adopting it could not cover one that has not started. If
    an interruption leaves that task pending, it ends on its own. An error from
    it, such as one from a task factory that wraps it, is then reported only
    when the task is garbage collected, as "Task exception was never retrieved".
    """

    def __init__(self, asyncio_runner: Runner) -> None:
        self._asyncio_runner = asyncio_runner
        self._loop = asyncio_runner.get_loop()
        self._cancel_active_operation: Callable[[], object] | None = None
        self._live_fixtures: set[FixtureTask[Any]] = set()
        self._abandoned: set[asyncio.Task[None]] = set()
        # Whether the last run of the loop was cut short; see _wait.
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
        unfinished = {fixture.task for fixture in self._live_fixtures} | self._abandoned
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
        with self._cancellable_operation(task.cancel):
            interruption = self._drive_until(task)
            if interruption is None:
                return outcome.result(task)
            task.cancel()
            self._join_or_abandon(task, outcome.report_unobserved, interruption)
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
            on_cancelled_while_held=self._on_fixture_cancelled,
        )
        self._live_fixtures.add(fixture)
        fixture.task.add_done_callback(lambda _: self._live_fixtures.discard(fixture))
        with self._cancellable_operation(fixture.cancel_setup):
            interruption = self._drive_until(fixture.task, fixture.ready)
            if interruption is None:
                return fixture, fixture.setup_result()
            # Pytest never receives an interrupted fixture. If it yields, it is
            # torn down at once, while its dependencies still exist.
            fixture.request_teardown()
            fixture.cancel_setup()
            self._join_or_abandon(fixture.task, fixture.report_unobserved, interruption)
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
        self._join_or_abandon(fixture.task, fixture.report_unobserved, interruption)
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
    def _cancellable_operation(self, cancel: Callable[[], object]) -> Generator[None]:
        """Cancel this test or pending setup if a fixture pytest holds is cancelled."""
        self._cancel_active_operation = cancel
        try:
            yield
        finally:
            self._cancel_active_operation = None

    def _join_or_abandon(
        self,
        task: asyncio.Task[None],
        report_unobserved: Callable[[asyncio.Task[None]], None],
        interruption: BaseException,
    ) -> None:
        """
        Drive the loop until the task, cancelled by ``interruption``, ends.

        A second interruption stops waiting and is raised, noting the first,
        unless it is a cancelled wait. That is never a request to stop, for
        example when code cancels every task, so it must not replace the first:
        Ctrl-C would become a test failure, and the next test would run. The
        task is cancelled again and left to end on its own, and
        ``report_unobserved`` reports its failure. Each interruption cancels
        the task once; close joins it without cancelling it again.
        """
        __tracebackhide__ = True
        second_interruption = self._drive_until(task)
        if second_interruption is not None:
            task.add_done_callback(report_unobserved)
            task.cancel()
            self._abandoned.add(task)
            task.add_done_callback(self._abandoned.discard)
            if isinstance(second_interruption, asyncio.CancelledError):
                # A cancelled wait carries this note already (see _wait).
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

        Every run of the loop by TaskRunner goes through here. Anything
        raised out of it interrupts the caller. That includes a
        CancelledError: Runner.run raises one when other code cancels the
        task running the wait, and turns its own SIGINT cancellation into
        KeyboardInterrupt.
        """
        __tracebackhide__ = True
        if self._run_cut_short:
            # Work around CPython gh-XXXXX. run_until_complete() stops the loop
            # from a done callback of its future. If an exception escapes the
            # run after that callback is scheduled but before it runs, it stays
            # queued and stops the next run early. One loop iteration runs it,
            # along with any other ready callbacks, I/O ones included, here
            # outside Runner.run's SIGINT handler. Remove this once the oldest
            # supported Python has the fix.
            self._loop.stop()
            self._loop.run_forever()
        self._run_cut_short = True
        try:
            self._asyncio_runner.run(asyncio.wait(waiters, return_when=return_when))
        except asyncio.CancelledError as cancelled:
            cancelled.add_note(_WAIT_CANCELLED)
            raise
        self._run_cut_short = False

    def _on_fixture_cancelled(self) -> None:
        if self._cancel_active_operation is not None:
            self._cancel_active_operation()


@dataclass(frozen=True)
class FixtureSetup(Generic[_T]):
    """A successful fixture setup: the value yielded, and the context then."""

    value: _T
    context: contextvars.Context


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
        name: str,
        loop: asyncio.AbstractEventLoop,
        on_cancelled_while_held: Callable[[], None],
    ) -> None:
        __tracebackhide__ = True
        self._gen = gen
        self.name = name
        self._on_cancelled_while_held = on_cancelled_while_held
        # Set once cancellation reaches the fixture before pytest's teardown.
        self.cancelled_while_held = False
        self.ready: asyncio.Future[FixtureSetup[_T]] = loop.create_future()
        self._teardown_requested: asyncio.Future[None] = loop.create_future()
        self._outcome: _Outcome[None] = _Outcome(loop)
        self.task = create_task(self._outcome.run(self._run_fixture))

    def setup_result(self) -> FixtureSetup[_T]:
        """What the fixture yielded; raises how its task ended if it never did."""
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
        # A cancelled setup that returns without yielding keeps the interruption.
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
        """Raises what the teardown raised, or how the task ended."""
        __tracebackhide__ = True
        self._outcome.result(self.task)

    def report_unobserved(self, task: asyncio.Task[None]) -> None:
        """A done callback for a fixture nobody reads: report an exception."""
        if (failure := self.failure()) is not None:
            _report_failure(task, failure)

    async def _run_fixture(self) -> None:
        value = await anext(self._gen)
        self.ready.set_result(FixtureSetup(value, contextvars.copy_context()))
        cancellation: asyncio.CancelledError | None = None
        while not self._teardown_requested.done():
            try:
                # Directly awaiting the future would let cancellation cancel
                # the handoff itself, preventing pytest from requesting teardown.
                await asyncio.wait([self._teardown_requested])
            except asyncio.CancelledError as exc:
                # Hold the cancellation until pytest tears the fixture down:
                # delivering it now would close resources that the cancelled
                # test's cleanup and dependent teardown still use. As
                # asyncio.TaskGroup does while it waits for its tasks, keep the
                # first one to throw in at the yield, and leave later ones,
                # such as AnyIO's repeats, pending rather than uncancelled, so
                # each scope unwinding at the yield sees whether another also
                # cancelled it.
                if cancellation is not None:
                    continue
                # Thrown in at the yield, it should not carry the frames of
                # the wait above into the fixture's error report.
                cancellation = exc.with_traceback(None)
                if not self._teardown_requested.done():
                    self.cancelled_while_held = True
                    self._on_cancelled_while_held()
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


class _Outcome(Generic[_T]):
    """
    What an async function returned, or the KeyboardInterrupt or SystemExit it raised.

    asyncio re-raises those two out of the event loop from the middle of an
    iteration, before the task's done callbacks run. There they would pass for
    an interruption from outside the task. Held here, they stay the function's
    own error. An exception that a task factory's wrapper raises after the
    function ends is reported instead, with the held one as its context. A
    cancellation of the task is not: once held, the exception cannot be told
    apart from one raised while a cancellation was pending, and natively such
    an exception wins over the pending cancellation.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._future: asyncio.Future[_T] = loop.create_future()

    async def run(self, func: Callable[[], Coroutine[object, object, _T]]) -> None:
        """The coroutine of the function's task."""
        __tracebackhide__ = True
        try:
            result = await func()
        except (KeyboardInterrupt, SystemExit) as exc:
            self._future.set_exception(exc)
        else:
            self._future.set_result(result)

    def result(self, task: asyncio.Task[None]) -> _T:
        """The result; raises what the function raised, or how its task ended."""
        __tracebackhide__ = True
        if (failure := self.failure(task)) is not None:
            raise failure
        task.result()  # A cancellation pending as the function returned wins.
        return self._future.result()

    def failure(self, task: asyncio.Task[None]) -> BaseException | None:
        """How the finished task failed, other than by cancellation."""
        held = self._future.exception() if self._future.done() else None
        if task.cancelled() or (failure := task.exception()) is None:
            return held
        # The task can fail after the function ends: a task factory may wrap
        # the function's coroutine. Chain what the function raised as it would
        # have been had it not been held here.
        if failure.__context__ is None:
            failure.__context__ = held
        return failure

    def report_unobserved(self, task: asyncio.Task[None]) -> None:
        """A done callback for a task nobody reads: report its failure."""
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
