"""Run every coroutine of an event loop scope in one long-lived task."""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import dataclasses
import sys
from collections.abc import Callable, Coroutine, Iterator
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
    "tests on this event loop has been cancelled, e.g. by a task group, cancel "
    "scope or timeout spanning the yield of a fixture. Only fixture teardowns "
    "run on it until the loop is closed."
)
_ABANDONED_FAILURE_MESSAGE = (
    "Exception from a coroutine that pytest-asyncio abandoned after a second "
    "interruption"
)
_DIED_MESSAGE = "The task running pytest-asyncio's fixtures and tests has died."


@dataclasses.dataclass(eq=False)
class _Job:
    """A coroutine submitted to the task, and what becomes of it."""

    coro: Coroutine[Any, Any, Any]
    # The context of the code that submitted the coroutine (see _applied).
    context: contextvars.Context
    # A fixture teardown runs even after the task was cancelled: the scope
    # that requested the cancellation needs it in order to exit.
    teardown: bool
    # The result or exception of the coroutine, set by the task once the
    # coroutine has ended and the task is done with it.
    outcome: asyncio.Future[Any]
    # Set by the task when it takes the job.
    started: bool = False
    # Set by the runner: it cancelled the task to end this job (see
    # TaskRunner._end).
    interrupted: bool = False
    # Set by the runner: nobody waits for the outcome any more.
    abandoned: bool = False

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


class TaskRunner:
    """
    Run coroutines one after another in a single, long-lived task.

    pytest requests the setup of an async fixture, the tests using it and its
    teardown in separate synchronous calls. Running the coroutines of all of
    them in one task, the task of the event loop, lets a task group, cancel
    scope or timeout entered before a fixture's ``yield`` be exited by the
    task that entered it. Each coroutine runs with the context variables of
    the code that submitted it applied to the task's context (see _applied).

    A cancellation of the task, e.g. by a task group whose child failed, ends
    the coroutine running at the time: its CancelledError is what run()
    raises. The task carries on, because pytest still has to tear the fixtures
    down, in this task, which exits the scope that requested the cancellation.
    From then on only fixture teardowns run; any other coroutine is refused,
    until the loop closes.

    An interruption of run() (SIGINT, or a BaseException raised by a callback
    of the loop) cancels the task to end the coroutine, and waits for it to
    end, so that it cleans up before pytest tears down what it may be using.
    A second interruption abandons the coroutine. The runner resolves the one
    cancellation request it made, and only that one: a request made in the
    meantime by a scope of the user's is taken to have cancelled the task.
    """

    def __init__(
        self,
        *,
        debug: bool | None = None,
        loop_factory: Callable[[], asyncio.AbstractEventLoop] | None = None,
    ) -> None:
        self._runner = Runner(debug=debug, loop_factory=loop_factory)
        self._jobs: asyncio.Queue[_Job | None] = asyncio.Queue()
        # Whether a cancellation not requested by the runner reached the task.
        self._cancelled = False

    def __enter__(self) -> Self:
        self._runner.__enter__()
        loop = self._runner.get_loop()
        # The task starts from an empty context; what it sees of the context
        # of the code submitting a coroutine is applied for that coroutine.
        self._task = contextvars.Context().run(
            loop.create_task, self._serve(), name="pytest-asyncio"
        )
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def get_loop(self) -> asyncio.AbstractEventLoop:
        return self._runner.get_loop()

    def run(self, coro: Coroutine[Any, Any, _T], *, teardown: bool = False) -> _T:
        """Run the coroutine in the task and return its result."""
        _check_no_running_loop(coro)
        if self._task.done():
            coro.close()
            raise _death(self._task)
        job = _Job(
            coro, contextvars.copy_context(), teardown, self.get_loop().create_future()
        )
        self._jobs.put_nowait(job)
        self._task.add_done_callback(job.task_died)
        try:
            interruption = self._wait(job.outcome)
            if interruption is not None:
                self._end(job, interruption)
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

    def _end(self, job: _Job, interruption: BaseException) -> None:
        """
        End the interrupted job, then raise the interruption.

        The job is cancelled and run until it has ended, so that it cleans up
        before pytest goes on. A failure of that cleanup, not the cancellation,
        is then the job's outcome, reported by run() instead of the
        interruption, as asyncio.Runner reports it and as for a synchronous
        test. A job that suppressed the cancellation does not suppress the
        interruption. A task that died meanwhile cannot end the job: the job
        fails with the death.
        """
        if self._task.done():
            job.task_died(self._task)
            return
        if not job.started:
            job.abandoned = True
            job.coro.close()
            raise interruption
        job.interrupted = True
        self._task.cancel()
        second_interruption = self._wait(job.outcome)
        if second_interruption is not None:
            # The job is abandoned: it ends at its next await, the next time
            # the loop runs. The runner's request stays a single one.
            job.abandoned = True
            _uncancel(self._task)
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
                # Before it started (see _end); its coroutine is closed.
                continue
            job.started = True
            await self._execute(job)

    async def _take(self) -> _Job | None:
        while True:
            try:
                return await self._jobs.get()
            except asyncio.CancelledError:
                # E.g. a task group spanning a fixture's yield whose child
                # failed while nothing was running.
                self._cancelled = True

    async def _execute(self, job: _Job) -> None:
        if self._cancelled and not job.teardown:
            job.coro.close()
            job.outcome.set_exception(asyncio.CancelledError(_REFUSED_MESSAGE))
            return
        try:
            with _applied(job.context):
                result = await job.coro
        except BaseException as exc:
            self._settle(job, cancelled=isinstance(exc, asyncio.CancelledError))
            job.fail(exc)
        else:
            self._settle(job, cancelled=False)
            job.succeed(result)

    def _settle(self, job: _Job, *, cancelled: bool) -> None:
        """
        Record whether a cancellation not requested by the runner reached the
        task, once the job has ended (``cancelled``: with a CancelledError).

        The request the runner made to end an interrupted job (see _end) is
        resolved now, whatever the job did with it, as asyncio.timeout does
        on exit. A request that remains was made by someone else meanwhile,
        e.g. by a task group whose child failed while the job was cleaning
        up, and is still to be delivered or was suppressed by the job.
        """
        if not job.interrupted:
            self._cancelled = self._cancelled or cancelled
        elif sys.version_info < (3, 11):
            # Python 3.10 cannot count requests: assume another one was made.
            self._cancelled = True
        elif self._task.uncancel() > 0:
            self._cancelled = True


_MISSING = object()


@contextlib.contextmanager
def _applied(context: contextvars.Context) -> Iterator[None]:
    """
    Apply the variables of the given context to the current one, for a while.

    The coroutine about to run in the task's context sees the variables set
    by the code that submitted it. Afterwards, each applied variable is
    restored to its previous value in the task's context, unless the
    coroutine left it with a different value.
    """
    applied = [
        (var, value, var.set(value))
        for var, value in context.items()
        if var.get(_MISSING) is not value
    ]
    try:
        yield
    finally:
        for var, value, token in reversed(applied):
            if var.get(_MISSING) is value:
                var.reset(token)


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


def _death(task: asyncio.Task[None]) -> RuntimeError:
    """The error reported for a job the done task cannot run."""
    try:
        cause: BaseException | None = task.exception()
    except asyncio.CancelledError as exc:
        # Cancelled before its first step, e.g. by a task factory.
        cause = exc
    error = RuntimeError(_DIED_MESSAGE)
    error.__cause__ = cause
    return error


def _uncancel(task: asyncio.Task[Any]) -> None:
    if sys.version_info >= (3, 11):
        task.uncancel()
