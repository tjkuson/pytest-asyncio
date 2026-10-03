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
from collections.abc import AsyncGenerator, Callable, Coroutine, Generator
from dataclasses import dataclass
from typing import Any, Generic, NoReturn, TypeVar

_T = TypeVar("_T")

_WAIT_CANCELLED = (
    "pytest-asyncio's internal wait was cancelled, e.g. by code that cancels "
    "every task. Cancel only tasks that your code created."
)


class TaskRunner:
    """
    Run async fixtures and tests in tasks of their own on an asyncio.Runner's loop.

    Each call runs the loop until its task is done or, in ``start_fixture``,
    has reached the fixture's yield. A call interrupted by an exception that
    escapes the loop, such as Ctrl-C, cancels its task and waits for it once,
    so that the task's cleanup runs while its fixtures are still set up. A
    second interruption leaves the task to asyncio.Runner.close().
    """

    def __init__(self, asyncio_runner: Runner) -> None:
        self._asyncio_runner = asyncio_runner
        self._loop = asyncio_runner.get_loop()
        self._cancel_active_operation: Callable[[], object] | None = None
        self._live_fixtures: set[FixtureTask[Any]] = set()
        # Whether the last run of the loop raised; see _run_stale_stop.
        self._run_cut_short = False

    def close(self) -> None:
        """
        Prepare the loop for asyncio.Runner.close(), which cancels its tasks.

        A fixture that pytest never finalized would keep that cancellation
        for a teardown request, so request its teardown.
        """
        self._run_stale_stop()
        for fixture in self._live_fixtures:
            fixture.request_teardown()

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
        task = self._create_task(func(), context=context, name=name)
        with self._cancel_on_held_fixture_cancellation(task.cancel):
            try:
                self._wait(task)
            except BaseException as interruption:
                task.cancel()
                self._join_and_raise(task, interruption)
        return task.result()

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
            try:
                self._wait(fixture.task, fixture.ready)
            except BaseException as interruption:
                # Pytest never receives an interrupted fixture. If it yields, it
                # is torn down at once, while its dependencies still exist.
                fixture.request_teardown()
                fixture.cancel_setup()
                self._join_and_raise(fixture.task, interruption)
        return fixture, fixture.setup_result()

    def finish_fixture(self, fixture: FixtureTask[_T]) -> None:
        """Run the fixture from its yield to its end, in its task."""
        __tracebackhide__ = True
        fixture.request_teardown()
        try:
            self._wait(fixture.task)
        except BaseException as interruption:
            fixture.task.cancel()
            self._join_and_raise(fixture.task, interruption)
        fixture.task.result()

    def _create_task(
        self,
        coro: Coroutine[object, object, _T],
        *,
        context: contextvars.Context,
        name: str,
    ) -> asyncio.Task[_T]:
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

    def _join_and_raise(
        self, task: asyncio.Task[Any], interruption: BaseException
    ) -> NoReturn:
        """
        Wait for the cancelled task; raise its cleanup's error or the interruption.

        A second interruption is raised as it is.
        """
        __tracebackhide__ = True
        try:
            self._wait(task)
        except asyncio.CancelledError:
            # A cancelled wait is no request to stop, so it must not replace the
            # interruption: Ctrl-C would become a test failure.
            raise interruption from None
        except BaseException as second_interruption:
            if _error_raised_by(task) is not second_interruption:
                raise
        failure = _error_raised_by(task)
        if failure is None or failure is interruption:
            raise interruption
        failure.add_note(
            f"Raised during cleanup after pytest-asyncio received: {interruption!r}"
        )
        raise failure

    def _wait(self, *waiters: asyncio.Future[Any]) -> None:
        """
        Run the loop until a waiter is done.

        Runner.run's SIGINT handling cancels this wait, not the test or fixture:
        Ctrl-C reaches the caller as KeyboardInterrupt, and the caller cancels
        its task.
        """
        __tracebackhide__ = True
        self._run_stale_stop()
        self._run_cut_short = True
        try:
            self._asyncio_runner.run(
                asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
            )
        except asyncio.CancelledError as cancelled:
            # Runner.run turns its own SIGINT cancellation into KeyboardInterrupt,
            # so other code cancelled the wait.
            cancelled.add_note(_WAIT_CANCELLED)
            raise
        self._run_cut_short = False

    def _run_stale_stop(self) -> None:
        """
        Work around CPython gh-158406.

        run_until_complete() stops the loop from a done callback of its future.
        If an exception escapes the run after that callback is scheduled but
        before it runs, it stays queued and stops the next run early. One loop
        iteration runs it, along with any other ready callbacks, outside
        Runner.run's SIGINT handler. Remove this once the oldest supported
        Python has a fix.
        """
        if self._run_cut_short:
            self._loop.stop()
            self._loop.run_forever()


def _error_raised_by(task: asyncio.Task[Any]) -> BaseException | None:
    """
    Return the exception the task ended with, other than its own cancellation.

    asyncio raises a task's KeyboardInterrupt or SystemExit out of the loop as
    well as storing it, so the exception that interrupted a wait can be this one.
    """
    if not task.done() or task.cancelled():
        return None
    return task.exception()


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
        self._teardown_requested = asyncio.Event()
        self.task = create_task(self._run())

    def setup_result(self) -> FixtureSetup[_T]:
        """Return what the fixture yielded, or raise why it did not."""
        __tracebackhide__ = True
        if not self.ready.done():
            # The task ended before the fixture yielded, so it failed.
            self.task.result()
        return self.ready.result()

    def cancel_setup(self) -> None:
        # Once setup is published, cancellation must leave teardown to pytest.
        if not self.ready.done():
            self.task.cancel()

    def request_teardown(self) -> None:
        self._teardown_requested.set()

    async def _run(self) -> None:
        try:
            value = await anext(self._gen)
        except StopAsyncIteration:
            if self._teardown_requested.is_set():
                # The setup was interrupted, so pytest no longer expects a value.
                return
            raise
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
        while not self._teardown_requested.is_set():
            try:
                await self._teardown_requested.wait()
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
                if not self._teardown_requested.is_set():
                    self.cancelled_while_held = True
                    self._on_cancelled_while_held()
        return cancellation
