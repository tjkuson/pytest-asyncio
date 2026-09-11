"""The task that runs every coroutine of an event loop scope."""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import sys
from collections.abc import AsyncGenerator, Iterator
from typing import TypeVar

import pytest
from pytest_asyncio._runner import TaskRunner

_INTERRUPTION_CANCELS_ON_310 = pytest.mark.xfail(
    sys.version_info < (3, 11),
    raises=asyncio.CancelledError,
    reason="Python 3.10 cannot count cancellation requests: an interrupted job "
    "leaves the task cancelled",
    strict=True,
)
_REFUSED = "The coroutine was refused"
_T = TypeVar("_T")
_var: contextvars.ContextVar[str] = contextvars.ContextVar("var")


def _task() -> asyncio.Task[object]:
    task = asyncio.current_task()
    assert task is not None
    return task


def _interrupt(loop: asyncio.AbstractEventLoop) -> None:
    """Interrupt the wait for the running job, as SIGINT does."""

    def raise_interrupt() -> None:
        raise KeyboardInterrupt("interrupted")

    loop.call_soon(raise_interrupt)


@contextlib.contextmanager
def _set_in_sync(value: str) -> Iterator[None]:
    token = _var.set(value)
    try:
        yield
    finally:
        _var.reset(token)


async def _cancelling() -> int:
    """The task's unresolved cancellation requests (Python 3.10 cannot count)."""
    return _task().cancelling() if sys.version_info >= (3, 11) else 0


async def _cancel_while_idle() -> None:
    """Request a cancellation that lands on the task as it waits for a job."""
    asyncio.get_running_loop().call_soon(_task().cancel)


async def _get() -> str:
    return _var.get("unset")


async def _set(value: str) -> None:
    _var.set(value)


async def _nothing() -> None:
    pass


async def _never_started() -> None:
    raise AssertionError("the coroutine was started")


async def _anext(gen: AsyncGenerator[_T]) -> _T:
    # As the plugin does: an async generator is first iterated in the loop.
    return await gen.__anext__()


def test_jobs_run_one_after_another_in_the_same_task():
    """The setup and teardown of a fixture and the tests share one task."""
    tasks = []

    async def fixture() -> AsyncGenerator[None]:
        tasks.append(_task())
        yield
        tasks.append(_task())

    async def test() -> None:
        tasks.append(_task())

    gen = fixture()
    with TaskRunner() as runner:
        runner.run(_anext(gen))
        runner.run(test())
        with pytest.raises(StopAsyncIteration):
            runner.run(_anext(gen), teardown=True)
    assert len(tasks) == 3
    assert len(set(tasks)) == 1
    assert tasks[0].get_name() == "pytest-asyncio"


@pytest.mark.parametrize("error_type", [ValueError, KeyboardInterrupt, SystemExit])
def test_run_raises_what_the_coroutine_raised(error_type: type[BaseException]):
    """Any exception of the coroutine is the outcome of run(); the task lives."""

    async def raise_it() -> None:
        raise error_type("from the coroutine")

    with TaskRunner() as runner:
        with pytest.raises(error_type, match="from the coroutine"):
            runner.run(raise_it())
        runner.run(_nothing())


def test_run_is_refused_from_a_running_loop():
    """A job cannot request another: run() refuses and closes the coroutine."""

    async def job() -> None:
        inner = _never_started()
        with pytest.raises(RuntimeError, match="from a running event loop"):
            runner.run(inner)
        assert inner.cr_frame is None

    with TaskRunner() as runner:
        runner.run(job())


def test_jobs_see_the_submitting_context_and_the_task_keeps_its_own():
    """
    The variables of the context submitting a job are applied to the task's
    context for that job. What a job sets stays in the task's context for the
    lifetime of the loop and never reaches the submitting context.
    """
    with TaskRunner() as runner:
        runner.run(_set("from job"))
        assert runner.run(_get()) == "from job"
        assert _var.get("unset") == "unset"
        with _set_in_sync("from sync"):
            assert runner.run(_get()) == "from sync"
        assert runner.run(_get()) == "from job"


def test_the_task_starts_from_an_empty_context():
    """A variable set before the loop's task exists is applied for a job at a time."""
    with _set_in_sync("from sync"):
        runner = TaskRunner().__enter__()
    with contextlib.closing(runner):
        assert runner.run(_get()) == "unset"
        with _set_in_sync("from sync"):
            assert runner.run(_get()) == "from sync"
        assert runner.run(_get()) == "unset"


def test_a_variable_the_job_changed_keeps_its_new_value():
    """An applied variable is restored after the job unless the job changed it."""
    with TaskRunner() as runner:
        with _set_in_sync("from sync"):
            runner.run(_set("from job"))
        assert runner.run(_get()) == "from job"


def test_a_variable_the_job_reset_stays_reset():
    """
    Resetting a token, as an async generator fixture does after its yield,
    unsets the variable, whatever a job set meanwhile and whatever value the
    submitting context applied.
    """

    async def set_with_token() -> contextvars.Token[str]:
        return _var.set("from fixture")

    async def reset(token: contextvars.Token[str]) -> None:
        _var.reset(token)

    with TaskRunner() as runner:
        token = runner.run(set_with_token())
        runner.run(_set("from test"))
        with _set_in_sync("from fixture"):
            runner.run(reset(token))
        assert runner.run(_get()) == "unset"


def test_setting_the_applied_value_again_is_not_a_change():
    """
    Whether a job changed a variable is known only from the value it left: a
    set to the value the submitting context applied is restored like any
    unchanged variable.
    """
    value = "shared"
    with TaskRunner() as runner:
        with _set_in_sync(value):
            runner.run(_set(value))
        assert runner.run(_get()) == "unset"


@pytest.mark.parametrize("seen", ["idle", "escaping"])
def test_cancellation_leaves_only_teardowns_until_the_loop_closes(seen: str):
    """
    A cancellation reaching the task, between two jobs or escaping one, is
    the end of the loop's normal work: only teardown jobs run from then on,
    for the requesting scope to exit; any other job is refused with a
    CancelledError saying so, its coroutine closed unawaited. Resolving the
    request, as the scope does on exit, changes nothing.
    """

    async def cancel_self() -> None:
        _task().cancel()
        await asyncio.sleep(0)

    async def resolve() -> None:
        if sys.version_info >= (3, 11):
            _task().uncancel()

    with TaskRunner() as runner:
        if seen == "idle":
            runner.run(_cancel_while_idle())
        else:
            with pytest.raises(asyncio.CancelledError):
                runner.run(cancel_self())
        refused = _never_started()
        with pytest.raises(asyncio.CancelledError, match=_REFUSED):
            runner.run(refused)
        assert refused.cr_frame is None
        runner.run(resolve(), teardown=True)
        assert runner.run(_cancelling(), teardown=True) == 0
        with pytest.raises(asyncio.CancelledError, match=_REFUSED):
            runner.run(_nothing())


def test_a_suppressed_cancellation_never_reaches_the_task():
    """A job that catches its own cancellation leaves the task serving."""

    async def swallow() -> None:
        _task().cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.sleep(0)

    with TaskRunner() as runner:
        runner.run(swallow())
        runner.run(_nothing())


@_INTERRUPTION_CANCELS_ON_310
@pytest.mark.parametrize("cleanup", ["propagate", "suppress", "fail"])
def test_interrupted_job_is_cancelled_and_joined(cleanup: str):
    """
    An interrupted job is cancelled and run until it has ended: its cleanup
    completes before run() raises. A cleanup that fails is the job's outcome
    (the cancellation its context); one that ends or suppresses the
    cancellation does not suppress the interruption. The runner's request is
    resolved, and the task goes on serving.
    """
    log = []

    async def job() -> None:
        _interrupt(asyncio.get_running_loop())
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await asyncio.sleep(0)
            log.append("cleaned up")
            if cleanup == "propagate":
                raise
            if cleanup == "fail":
                raise ValueError("cleanup failed") from None

    with TaskRunner() as runner:
        expected = ValueError if cleanup == "fail" else KeyboardInterrupt
        with pytest.raises(expected) as info:
            runner.run(job())
        assert log == ["cleaned up"]
        if cleanup == "fail":
            assert isinstance(info.value.__context__, asyncio.CancelledError)
        assert runner.run(_cancelling(), teardown=True) == 0
        runner.run(_nothing())


@_INTERRUPTION_CANCELS_ON_310
@pytest.mark.parametrize("then", ["propagate", "suppress", "fail"])
def test_second_interruption_abandons_the_job(then: str):
    """
    A wait for the job's cleanup that is interrupted too ends at once. The
    job ends at its next await, the next time the loop runs; a failure then
    goes to the loop's exception handler instead of being lost.
    """
    log = []
    reported = []

    async def job() -> None:
        loop = asyncio.get_running_loop()
        _interrupt(loop)
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            _interrupt(loop)
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                log.append("ended")
                if then == "propagate":
                    raise
                if then == "fail":
                    raise ValueError("late failure") from None

    with TaskRunner() as runner:
        runner.get_loop().set_exception_handler(lambda loop, ctx: reported.append(ctx))
        with pytest.raises(KeyboardInterrupt):
            runner.run(job())
        assert log == []
        runner.run(_nothing())
        assert log == ["ended"]
        assert runner.run(_cancelling(), teardown=True) == 0
    failures = [type(context["exception"]) for context in reported]
    assert failures == ([ValueError] if then == "fail" else [])


def test_interruption_does_not_hide_a_cancellation_requested_meanwhile():
    """
    While the interrupted job cleans up, a scope of the user's may cancel
    the task too (a task group whose child failed, say). Resolving the
    runner's own request leaves that one: the task counts as cancelled.
    """

    async def job() -> None:
        _interrupt(asyncio.get_running_loop())
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            _task().cancel()
            await asyncio.Event().wait()

    with TaskRunner() as runner:
        with pytest.raises(KeyboardInterrupt):
            runner.run(job())
        with pytest.raises(asyncio.CancelledError, match=_REFUSED):
            runner.run(_nothing())
        runner.run(_nothing(), teardown=True)


def test_interruption_before_the_job_started_skips_the_job():
    """A job the task had not taken when the wait was interrupted never runs."""
    with TaskRunner() as runner:
        runner.run(_nothing())
        _interrupt(runner.get_loop())
        skipped = _never_started()
        with pytest.raises(KeyboardInterrupt):
            runner.run(skipped)
        assert skipped.cr_frame is None
        runner.run(_nothing())


def test_task_cancelled_before_its_first_step_fails_the_job():
    """A task factory that cancels the task, say: the job fails, loudly."""

    def loop_factory() -> asyncio.AbstractEventLoop:
        loop = asyncio.new_event_loop()
        cancelled = []

        def task_factory(loop, coro, **kwargs):
            task = asyncio.Task(coro, loop=loop, **kwargs)
            if not cancelled:
                # The first task of the loop is the runner's.
                cancelled.append(task)
                task.cancel()
            return task

        loop.set_task_factory(task_factory)
        return loop

    with (
        TaskRunner(loop_factory=loop_factory) as runner,
        pytest.raises(RuntimeError, match="has died") as info,
    ):
        runner.run(_never_started())
    assert isinstance(info.value.__cause__, asyncio.CancelledError)


class _Failed(BaseException):
    """A BaseException asyncio keeps inside the loop (pytest-timeout's, say)."""


@pytest.mark.parametrize("error_type", [SystemExit, _Failed])
def test_task_dying_fails_every_job(
    monkeypatch: pytest.MonkeyPatch, error_type: type[BaseException]
):
    """
    A BaseException raised in the task's own frames, outside any job (by a
    signal handler, say), kills the task, whether asyncio lets it escape the
    loop (SystemExit here, standing in for KeyboardInterrupt) or keeps it in
    the task. The job and every later one fail with a RuntimeError carrying
    it as cause, their coroutines closed unawaited; the loop closes cleanly.
    """

    def die(self: TaskRunner, job: object, *, cancelled: bool) -> None:
        raise error_type("task killed")

    monkeypatch.setattr(TaskRunner, "_settle", die)
    with TaskRunner() as runner:
        for coro in (_nothing(), _never_started()):
            with pytest.raises(RuntimeError, match="has died") as info:
                runner.run(coro)
            assert isinstance(info.value.__cause__, error_type)


def test_close_finalises_an_unresumed_async_generator():
    """
    An async generator fixture whose teardown never ran (its test was
    interrupted, say) is finalised in the loop when the runner closes.
    """
    log = []

    async def fixture() -> AsyncGenerator[str]:
        try:
            yield "value"
        finally:
            await asyncio.sleep(0)
            log.append("finalised")

    gen = fixture()
    with TaskRunner() as runner:
        assert runner.run(_anext(gen)) == "value"
    assert log == ["finalised"]
    assert gen.ag_frame is None
