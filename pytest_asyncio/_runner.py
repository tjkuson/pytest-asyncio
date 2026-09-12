"""Run every coroutine of an event loop scope in one long-lived task."""

from __future__ import annotations

import asyncio
import contextvars
import dataclasses
import sys
import types
from collections.abc import Callable, Coroutine, Generator
from typing import Any, TypeVar

if sys.version_info >= (3, 11):
    from asyncio import Runner
    from typing import Self
else:
    from backports.asyncio.runner import Runner
    from typing_extensions import Self

_T = TypeVar("_T")

_REFUSED_MESSAGE = (
    "The coroutine was refused: the task running pytest-asyncio's fixtures and "
    "tests on this event loop has been cancelled, by a task group, cancel scope "
    "or timeout spanning the yield of a fixture, or to interrupt a test or "
    "fixture. Only fixture teardowns run on it until the loop is closed."
)
_ABANDONED_FAILURE_MESSAGE = (
    "Exception from a coroutine that pytest-asyncio abandoned after a second "
    "interruption"
)
_DIED_NOTE = (
    "The task running pytest-asyncio's fixtures and tests on this event loop "
    "died of this exception; nothing else runs on the loop."
)
_ENDED_MESSAGE = (
    "The task running pytest-asyncio's fixtures and tests on this event loop "
    "has ended."
)


@dataclasses.dataclass(eq=False)
class _Job:
    """A coroutine submitted to the task, and what becomes of it."""

    coro: Coroutine[Any, Any, Any]
    # The context the coroutine runs in (see _run_in).
    context: contextvars.Context
    # A fixture teardown runs even after the task was cancelled: the scope
    # that requested the cancellation needs it in order to exit.
    teardown: bool
    # The result or exception of the coroutine, set by the task once the
    # coroutine has ended (or by task_died, if the task ends first).
    outcome: asyncio.Future[Any]
    # Set by the task when it takes the job.
    started: bool = False
    # Set by abandon(): nobody waits for the outcome any more.
    abandoned: bool = False

    def abandon(self) -> None:
        """Stop waiting for the outcome (see TaskRunner._interrupt)."""
        self.abandoned = True
        # Whoever waited for the outcome so far (see _wait_for) is released.
        self.outcome.cancel()

    def refuse(self) -> None:
        self.coro.close()
        self.outcome.set_exception(asyncio.CancelledError(_REFUSED_MESSAGE))

    def succeed(self, result: object) -> None:
        if not self.abandoned:
            self.outcome.set_result(result)

    def fail(self, exc: BaseException) -> None:
        if not self.abandoned:
            self.outcome.set_exception(exc)
        elif not isinstance(exc, asyncio.CancelledError):
            # Reported like the exception of a task that nobody awaits.
            self.outcome.get_loop().call_exception_handler(
                {"message": _ABANDONED_FAILURE_MESSAGE, "exception": exc}
            )

    def task_died(self, task: asyncio.Task[None]) -> None:
        """Fail the job if the task ended while the job was waiting for it."""
        if self.outcome.done():
            return
        self.coro.close()
        if not self.abandoned:
            self.outcome.set_exception(_death(task))


@types.coroutine
def _run_in(
    context: contextvars.Context, coro: Coroutine[Any, Any, Any]
) -> Generator[Any, Any, Any]:
    """
    Await the coroutine in the current task, running it in the given context.

    Every step of the coroutine runs with the context entered, so the
    coroutine reads and writes that context as it would in a task created
    with ``context=context``. What it awaits is yielded on to the task as is,
    and what the task sends or throws in (a cancellation, say) is passed on
    to the coroutine.
    """
    __tracebackhide__ = True
    step: Callable[[Any], Any] = coro.send
    arg: Any = None
    while True:
        try:
            awaited = context.run(step, arg)
        except StopIteration as stop:
            return stop.value
        try:
            arg = yield awaited
            step = coro.send
        except BaseException as exc:
            step, arg = coro.throw, exc


class TaskRunner:
    """
    Run coroutines one after another in a single, long-lived task.

    pytest requests the setup of an async fixture, the tests using it and its
    teardown in separate synchronous calls. Running the coroutines of all of
    them in one task, the task of the event loop, lets a task group, cancel
    scope or timeout entered before a fixture's ``yield`` be exited by the
    task that entered it. Each coroutine runs in the context passed to
    :meth:`run`, as it would in a task of its own (see _run_in).

    A cancellation of the task, e.g. by a task group whose child failed, ends
    the coroutine running at the time: its CancelledError is what run()
    raises. The task carries on, because pytest still has to tear the fixtures
    down, in this task, which exits the scope. From then on only fixture
    teardowns run; any other coroutine is refused, until the loop closes.

    An interruption of run() (SIGINT, or a BaseException raised by a callback
    of the loop) cancels the task to end the coroutine, and waits for it to
    end, so that it cleans up before pytest tears down what it may be using;
    a second interruption abandons the coroutine. The runner cannot tell its
    own cancellation from a scope's, so an interruption ends the normal work
    of the loop too.
    """

    def __init__(
        self,
        *,
        debug: bool | None = None,
        loop_factory: Callable[[], asyncio.AbstractEventLoop] | None = None,
    ) -> None:
        self._runner = Runner(debug=debug, loop_factory=loop_factory)
        self._jobs: asyncio.Queue[_Job | None] = asyncio.Queue()
        # Whether the task was cancelled (see the class docstring).
        self._teardown_only = False

    def __enter__(self) -> Self:
        self._runner.__enter__()
        loop = self._runner.get_loop()
        self._task = loop.create_task(self._serve(), name="pytest-asyncio")
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def get_loop(self) -> asyncio.AbstractEventLoop:
        return self._runner.get_loop()

    def run(
        self,
        coro: Coroutine[Any, Any, _T],
        *,
        context: contextvars.Context | None = None,
        teardown: bool = False,
    ) -> _T:
        """
        Run the coroutine in the task and return its result.

        The coroutine runs in the given context, or in a copy of the current
        one, as with :meth:`asyncio.Runner.run`.
        """
        _check_no_running_loop(coro)
        if self._task.done():
            coro.close()
            raise _death(self._task)
        if context is None:
            context = contextvars.copy_context()
        job = _Job(coro, context, teardown, self.get_loop().create_future())
        self._jobs.put_nowait(job)
        self._task.add_done_callback(job.task_died)
        try:
            interruption = self._wait(job.outcome)
            if interruption is not None:
                self._interrupt(job, interruption)
        finally:
            self._task.remove_done_callback(job.task_died)
        return job.outcome.result()

    def _wait(self, future: asyncio.Future[Any]) -> BaseException | None:
        """Run the loop until the future is done, or the wait is interrupted."""
        try:
            self._runner.run(_wait_for(future))
        except BaseException as exc:
            return exc
        return None

    def _interrupt(self, job: _Job, interruption: BaseException) -> None:
        """
        Cancel the interrupted job and wait for it to end, then raise the
        interruption.

        The job cleans up before pytest goes on. A failure of that cleanup,
        not the cancellation, is then the job's outcome, reported by run()
        instead of the interruption, as asyncio.Runner reports it and as for a
        synchronous test. A job that ended, or suppressed the cancellation,
        does not suppress the interruption. A second interruption abandons
        the job: it ends at its next await, when the loop next runs.
        """
        if not job.outcome.done():
            if not job.started:
                job.abandon()
                job.coro.close()
                raise interruption
            self._teardown_only = True
            self._task.cancel()
            second_interruption = self._wait(job.outcome)
            if second_interruption is not None:
                job.abandon()
                self._task.cancel()
                raise second_interruption
        if _failure(job.outcome) is None:
            raise interruption

    def close(self) -> None:
        try:
            # A task that died (see _death) is not joined: awaiting a done
            # task whose exception is KeyboardInterrupt or SystemExit never
            # stops the loop (asyncio's _run_until_complete_cb).
            if not self._task.done():
                self._jobs.put_nowait(None)
                self.get_loop().run_until_complete(self._task)
        finally:
            self._runner.close()

    async def _serve(self) -> None:
        while (job := await self._take()) is not None:
            if job.abandoned:
                # Before it started (see _interrupt); its coroutine is closed.
                continue
            job.started = True
            await self._execute(job)

    async def _take(self) -> _Job | None:
        while True:
            try:
                return await self._jobs.get()
            except asyncio.CancelledError:
                # Cancelled while nothing was running, e.g. by a task group
                # spanning a fixture's yield whose child failed.
                self._teardown_only = True

    async def _execute(self, job: _Job) -> None:
        if self._teardown_only and not job.teardown:
            job.refuse()
            return
        try:
            result = await _run_in(job.context, job.coro)
        except BaseException as exc:
            # Cancelled while the job ran (see the class docstring).
            self._teardown_only |= isinstance(exc, asyncio.CancelledError)
            job.fail(exc)
        else:
            job.succeed(result)


def _check_no_running_loop(coro: Coroutine[Any, Any, Any]) -> None:
    # Runner.run() refuses to run inside a running event loop; checking first
    # keeps a refused job off the queue (it would run later, unrequested).
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    coro.close()
    msg = "pytest-asyncio fixtures cannot be requested from a running event loop"
    raise RuntimeError(msg)


async def _wait_for(future: asyncio.Future[Any]) -> None:
    # Unlike awaiting the future directly, this neither cancels it when the
    # waiter is cancelled nor raises its exception.
    await asyncio.wait([future])


def _failure(future: asyncio.Future[Any]) -> BaseException | None:
    """The exception of the done future, unless it is a cancellation."""
    exc = None if future.cancelled() else future.exception()
    return None if isinstance(exc, asyncio.CancelledError) else exc


def _death(task: asyncio.Task[None]) -> BaseException:
    """
    What the jobs a done task cannot run raise: the exception it died of.

    It is raised as it is, so that it keeps its meaning for pytest, which
    stops the session for a KeyboardInterrupt or pytest.exit() and reports
    anything else as a failure.
    """
    try:
        exc = task.exception()
    except asyncio.CancelledError as cancelled:
        # Cancelled before its first step, e.g. by a task factory.
        exc = cancelled
    if exc is None:
        return RuntimeError(_ENDED_MESSAGE)
    if sys.version_info >= (3, 11) and _DIED_NOTE not in getattr(exc, "__notes__", ()):
        exc.add_note(_DIED_NOTE)
    return exc
