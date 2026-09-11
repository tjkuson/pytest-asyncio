"""Run every coroutine of an event loop scope in one long-lived task."""

from __future__ import annotations

import asyncio
import contextvars
import dataclasses
import functools
import sys
from collections.abc import Callable, Coroutine
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
    "tests on this event loop has an unresolved cancellation request. Only "
    "fixture teardowns run until the task group, cancel scope or timeout that "
    "requested the cancellation exits."
)
_ABANDONED_FAILURE_MESSAGE = (
    "Exception from a coroutine that pytest-asyncio abandoned after a second "
    "interruption"
)
_WORKER_DIED_MESSAGE = "The task running pytest-asyncio's fixtures and tests has died."


@dataclasses.dataclass(eq=False)
class _Job:
    coro: Coroutine[Any, Any, Any]
    future: asyncio.Future[Any]
    # The context of the code that submitted the coroutine.
    context: contextvars.Context
    # Fixture teardowns run even while the task is cancelling: the scope that
    # requested the cancellation needs them in order to exit.
    teardown: bool
    # Set by TaskRunner._cancel: the runner cancelled the worker task to end
    # this job.
    interrupted: bool = False


class TaskRunner:
    """
    Run coroutines one after another in a single, long-lived task.

    Every coroutine submitted to :meth:`run` is awaited by the same task, so the
    setup and teardown of an async generator fixture, the fixtures depending on
    it, and the test body all execute in the task that entered any task group,
    cancel scope or timeout spanning the fixture's ``yield``. Each coroutine sees
    the context variables set by the code that submitted it, and the values it
    sets itself persist in the task's context for the lifetime of the loop.

    Cancelling that task cancels the coroutine running at the time, or lands on
    the task while it waits for the next coroutine. Either way the task has seen
    a cancellation whose request may be unresolved: asyncio counts the requests
    made of a task (``Task.cancelling()``), and the task group, cancel scope or
    timeout that made one resolves it (``Task.uncancel()``) when it exits. Until
    then the task runs nothing but fixture teardowns, which the requesting scope
    needs in order to exit: any other coroutine is refused. Refusal is triggered
    by a cancellation the task actually saw and released when asyncio reports no
    unresolved request. Python 3.10 cannot report: there, the task stays in
    teardown-only mode until the loop closes. The runner resolves no request but
    its own, made by :meth:`run` to end a coroutine whose wait was interrupted.
    """

    def __init__(
        self,
        *,
        debug: bool | None = None,
        loop_factory: Callable[[], asyncio.AbstractEventLoop] | None = None,
    ) -> None:
        self._runner = Runner(debug=debug, loop_factory=loop_factory)
        self._jobs: asyncio.Queue[_Job | None] = asyncio.Queue()

    def __enter__(self) -> Self:
        self._runner.__enter__()
        loop = self._runner.get_loop()
        self._task = loop.create_task(self._serve(), name="pytest-asyncio")
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def get_loop(self) -> asyncio.AbstractEventLoop:
        return self._runner.get_loop()

    def run(self, coro: Coroutine[Any, Any, _T], *, teardown: bool = False) -> _T:
        """Run the coroutine in the worker task and return its result."""
        _check_no_running_loop(coro)
        loop = self._runner.get_loop()
        job = _Job(coro, loop.create_future(), contextvars.copy_context(), teardown)
        self._jobs.put_nowait(job)
        # A worker that ends without completing the job (a BaseException raised
        # in its own frames, e.g. KeyboardInterrupt from a signal handler) fails
        # the job instead of leaving the wait below to hang forever.
        worker_died = functools.partial(_fail_job, job)
        self._task.add_done_callback(worker_died)
        try:
            self._runner.run(_wait(job.future))
        except BaseException as exc:
            # The wait was interrupted (e.g. by SIGINT) before the job ended.
            # The job is dealt with outside this handler, so that what it
            # raises while being cancelled is not chained to the interruption.
            interruption: BaseException | None = exc
        else:
            interruption = None
        try:
            # The job is cancelled and run until it ends, so that it cleans up
            # before pytest tears down what it may be using.
            if interruption is not None and not job.future.done():
                self._cancel(job)
        finally:
            self._task.remove_done_callback(worker_died)
        if interruption is not None and not _failed(job.future):
            # The job ended with the cancellation, or suppressed it: Ctrl-C is
            # not the job's to suppress. A cleanup that failed instead is the
            # job's outcome, reported below as asyncio.Runner would (the
            # cancellation is its context), just as for a synchronous test.
            raise interruption
        return job.future.result()

    def _cancel(self, job: _Job) -> None:
        """Cancel the job and run the loop until it has ended."""
        # A job still queued never ran: the worker skips it (dequeueing and
        # the job's first step happen in one step of the worker, so an empty
        # queue means the job runs). A dead worker runs nothing: its done
        # callback fails the job (see run()).
        if not self._jobs.empty() or self._task.done():
            job.coro.close()
            job.future.cancel()
            return
        job.interrupted = True
        self._task.cancel()
        try:
            self._runner.run(_wait(job.future))
        except BaseException:
            # Interrupted again (e.g. a second SIGINT): the job is abandoned.
            # It is cancelled once more so that it ends at its next await, the
            # next time the loop runs (for a later job, or in close()). The
            # request above is resolved here; the worker resolves the new one
            # when the job ends (see _serve). Nobody waits for the outcome.
            _uncancel(self._task)
            self._task.cancel()
            job.future.cancel()
            raise

    def close(self) -> None:
        try:
            # A worker that already died (see run()) is not joined: awaiting a
            # done task whose exception is KeyboardInterrupt or SystemExit
            # never stops the loop (asyncio's _run_until_complete_cb).
            if not self._task.done():
                self._jobs.put_nowait(None)
                self._runner.get_loop().run_until_complete(self._task)
        finally:
            self._runner.close()

    async def _serve(self) -> None:
        task = asyncio.current_task()
        assert task is not None
        # Whether this task has seen a cancellation that may be unresolved (see
        # the class docstring).
        observed = False
        while True:
            try:
                job = await self._jobs.get()
            except asyncio.CancelledError:
                # E.g. a task group spanning a fixture's yield whose child
                # failed while nothing was running.
                observed = True
                continue
            if job is None:
                return
            if job.future.cancelled():
                job.coro.close()
                continue
            if observed and not job.teardown:
                if _may_be_cancelling(task):
                    job.coro.close()
                    job.future.set_exception(asyncio.CancelledError(_REFUSED_MESSAGE))
                    continue
                observed = False
            overlay = _apply_context(job.context)
            try:
                result = await job.coro
            except BaseException as exc:
                cancelled = isinstance(exc, asyncio.CancelledError)
                if not job.future.cancelled():
                    job.future.set_exception(exc)
                elif not cancelled:
                    # The job was abandoned (see _cancel) and failed: reported
                    # like the exception of a task that nobody awaits.
                    asyncio.get_running_loop().call_exception_handler(
                        {"message": _ABANDONED_FAILURE_MESSAGE, "exception": exc}
                    )
            else:
                cancelled = False
                if not job.future.cancelled():
                    job.future.set_result(result)
            finally:
                _remove_overlay(overlay)
            if job.interrupted:
                # The runner's request (see _cancel) is resolved now that the
                # job it was made for has ended, whatever the job did with it.
                _uncancel(task)
            elif cancelled:
                observed = True


_Overlay = list[tuple[contextvars.ContextVar[Any], Any, contextvars.Token[Any]]]


def _apply_context(context: contextvars.Context) -> _Overlay:
    """
    Set the variables of the given context in the current one, for a while.

    The coroutine about to run sees the values set by the code that submitted
    it; :func:`_remove_overlay` restores the current context's own values
    afterwards.
    """
    overlay = []
    for var, value in context.items():
        try:
            if var.get() is value:
                continue
        except LookupError:
            pass
        overlay.append((var, value, var.set(value)))
    return overlay


def _remove_overlay(overlay: _Overlay) -> None:
    # A variable the coroutine set itself keeps its new value.
    for var, value, token in reversed(overlay):
        if var.get() is value:
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


async def _wait(future: asyncio.Future[Any]) -> None:
    # Unlike awaiting the future directly, this neither cancels it nor raises
    # its exception; the outcome is retrieved by the caller.
    await asyncio.wait([future])


def _failed(future: asyncio.Future[Any]) -> bool:
    """Whether the done future holds an exception other than a cancellation."""
    exc = None if future.cancelled() else future.exception()
    return exc is not None and not isinstance(exc, asyncio.CancelledError)


def _fail_job(job: _Job, worker: asyncio.Task[None]) -> None:
    if job.future.done():
        return
    job.coro.close()
    try:
        cause = worker.exception()
    except asyncio.CancelledError as exc:
        # Cancelled before its first step, e.g. by a task factory.
        cause = exc
    error = RuntimeError(_WORKER_DIED_MESSAGE)
    error.__cause__ = cause
    job.future.set_exception(error)


def _may_be_cancelling(task: asyncio.Task[Any]) -> bool:
    if sys.version_info >= (3, 11):
        return task.cancelling() > 0
    # Python 3.10 cannot report whether a request is unresolved.
    return True


def _uncancel(task: asyncio.Task[Any]) -> None:
    if sys.version_info >= (3, 11):
        task.uncancel()
