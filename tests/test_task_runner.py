"""The task that runs every coroutine of an event loop scope."""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import sys
from collections.abc import AsyncGenerator, Iterator
from typing import TypeVar

import pytest
import pytest_asyncio._runner
from pytest_asyncio._runner import TaskRunner

_REQUIRES_311 = pytest.mark.skipif(
    sys.version_info < (3, 11), reason="Task.cancelling() needs Python 3.11"
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


def _assert_teardown_only(runner: TaskRunner) -> None:
    refused = _never_started()
    with pytest.raises(asyncio.CancelledError, match=_REFUSED):
        runner.run(refused)
    assert refused.cr_frame is None
    runner.run(_nothing(), teardown=True)


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


def test_a_job_runs_in_the_given_context():
    """
    A job reads and writes the context passed to run(), and nothing else: not
    the submitting context, not the context of another job.
    """
    context = contextvars.copy_context()
    with TaskRunner() as runner:
        runner.run(_set("in job"), context=context)
        assert context[_var] == "in job"
        assert _var.get("unset") == "unset"
        assert runner.run(_get()) == "unset"
        assert runner.run(_get(), context=context) == "in job"


def test_a_job_runs_in_a_copy_of_the_current_context_by_default():
    """As with asyncio.Runner.run(): a job sees what was set when it was run."""
    with TaskRunner() as runner:
        with _set_in_sync("from sync"):
            assert runner.run(_get()) == "from sync"
        assert runner.run(_get()) == "unset"


def test_tasks_and_callbacks_of_a_job_copy_its_context():
    """Whatever a job starts inherits the job's context, as usual in asyncio."""

    async def job() -> tuple[str, str]:
        _var.set("in job")
        loop = asyncio.get_running_loop()
        from_callback = loop.create_future()
        loop.call_soon(lambda: from_callback.set_result(_var.get("unset")))
        return await asyncio.create_task(_get()), await from_callback

    with TaskRunner() as runner:
        assert runner.run(job()) == ("in job", "in job")


@pytest.mark.skipif(sys.version_info < (3, 12), reason="Task.get_context() is new")
def test_the_task_context_is_not_the_job_context():
    """
    The known difference from a task of the job's own: the task's context is
    not the job's, so passing it on explicitly passes on the wrong variables;
    copy_context() and the default of create_task() and call_soon() are right.
    """

    async def job() -> tuple[str, str, str]:
        _var.set("in job")
        loop = asyncio.get_running_loop()
        explicit = _task().get_context()
        child = asyncio.create_task(_get(), context=explicit)
        from_callback = loop.create_future()
        loop.call_soon(
            lambda: from_callback.set_result(_var.get("unset")), context=explicit
        )
        return await child, await from_callback, contextvars.copy_context()[_var]

    with TaskRunner() as runner:
        assert runner.run(job()) == ("unset", "unset", "in job")


@pytest.mark.parametrize("seen", ["idle", "escaping"])
def test_a_cancellation_of_the_task_leaves_only_teardowns(seen: str):
    """
    A cancellation of the task, landing on it between two jobs or escaping
    one, ends the loop's normal work: only teardown jobs run from then on,
    for the requesting scope to exit; any other job is refused with a
    CancelledError saying so, its coroutine closed unawaited. The scope
    resolving its request on exit changes nothing.
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
        _assert_teardown_only(runner)
        runner.run(resolve(), teardown=True)
        assert runner.run(_cancelling(), teardown=True) == 0
        _assert_teardown_only(runner)


def test_a_suppressed_cancellation_never_reaches_the_task():
    """A job that catches its own cancellation leaves the task serving."""

    async def swallow() -> None:
        _task().cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.sleep(0)

    with TaskRunner() as runner:
        runner.run(swallow())
        runner.run(_nothing())


@pytest.mark.parametrize("cleanup", ["propagate", "suppress", "fail"])
@pytest.mark.parametrize("suspended", ["at an await", "between steps"])
def test_interrupted_job_is_cancelled_and_joined(suspended: str, cleanup: str):
    """
    An interrupted job's task is cancelled, as asyncio.Runner cancels its
    task, and the job runs until it has ended: its cleanup completes before
    run() raises. A cleanup that fails is the job's outcome (the cancellation
    its context); one that ends or suppresses the cancellation does not
    suppress the interruption. The loop is teardown-only from then on.
    """
    log = []

    async def job() -> None:
        _interrupt(asyncio.get_running_loop())
        try:
            if suspended == "at an await":
                await asyncio.Event().wait()
            else:
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            if sys.version_info >= (3, 11):
                assert await _cancelling() == 1
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
        _assert_teardown_only(runner)


@_REQUIRES_311
def test_interrupted_job_tells_its_cancellation_from_a_child_s():
    """
    The interruption is a cancellation request on the task, which a job
    awaiting a child task can tell from the child's own cancellation.
    """
    log = []

    async def job() -> None:
        _interrupt(asyncio.get_running_loop())
        try:
            await asyncio.create_task(asyncio.Event().wait())
        except asyncio.CancelledError:
            if await _cancelling() == 0:
                log.append("child cancelled; carrying on")
                await asyncio.Event().wait()
            raise
        finally:
            log.append("cleaned up")

    with TaskRunner() as runner:
        with pytest.raises(KeyboardInterrupt):
            runner.run(job())
        assert log == ["cleaned up"]


def test_interrupted_job_awaiting_a_future_like_object():
    """Whatever the job awaits, asyncio's own cancellation reaches it."""

    class FutureLike:
        """A Future-like object (asyncio.isfuture), not an asyncio.Future."""

        _asyncio_future_blocking = False

        def __init__(self) -> None:
            self.inner = asyncio.get_running_loop().create_future()
            self.cancelled_with: list[object] = []

        def __await__(self):
            if not self.inner.done():
                self._asyncio_future_blocking = True
                yield self
            return self.inner.result()

        def get_loop(self):
            return self.inner.get_loop()

        def add_done_callback(self, callback, *, context=None):
            self.inner.add_done_callback(lambda _: callback(self), context=context)

        def remove_done_callback(self, callback):
            return 0

        def cancel(self, msg=None):
            self.cancelled_with.append(msg)
            return self.inner.cancel(msg)

        def done(self):
            return self.inner.done()

        def cancelled(self):
            return self.inner.cancelled()

        def result(self):
            return self.inner.result()

        def exception(self):
            return self.inner.exception()

    cancelled_with = []

    async def job() -> None:
        _interrupt(asyncio.get_running_loop())
        future = FutureLike()
        assert asyncio.isfuture(future)
        try:
            await future
        finally:
            cancelled_with.extend(future.cancelled_with)

    with TaskRunner() as runner, pytest.raises(KeyboardInterrupt):
        runner.run(job())
    assert cancelled_with == [None]


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
        runner.run(_nothing(), teardown=True)
        assert log == ["ended"]
    failures = [type(context["exception"]) for context in reported]
    assert failures == ([ValueError] if then == "fail" else [])


def test_interruption_after_the_job_ended_is_only_the_interruption():
    """A job that ended before the wait for it was interrupted is left alone."""

    async def job() -> None:
        _interrupt(asyncio.get_running_loop())

    with TaskRunner() as runner:
        with pytest.raises(KeyboardInterrupt):
            runner.run(job())
        assert runner.run(_cancelling()) == 0


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
        pytest.raises(asyncio.CancelledError) as info,
    ):
        runner.run(_never_started())
    if sys.version_info >= (3, 11):
        assert any("died" in note for note in info.value.__notes__)


class _Failed(BaseException):
    """A BaseException asyncio keeps inside the loop (pytest-timeout's, say)."""


@pytest.mark.parametrize(
    "error_type", [KeyboardInterrupt, SystemExit, _Failed, ValueError]
)
def test_task_dying_raises_its_exception_for_every_job(
    monkeypatch: pytest.MonkeyPatch, error_type: type[BaseException]
):
    """
    An exception raised in the task's own frames, outside any job, kills the
    task. The job's run() and every later one raise that exception as it is,
    so that pytest treats it as it would from a synchronous test (a
    KeyboardInterrupt stops the session, a SystemExit is a failure). The
    coroutines are closed unawaited, and the loop closes cleanly.
    """

    def die(self: object, result: object) -> None:
        raise error_type("task killed")

    monkeypatch.setattr(pytest_asyncio._runner._Job, "succeed", die)
    with TaskRunner() as runner:
        for coro in (_nothing(), _never_started()):
            with pytest.raises(error_type, match="task killed") as info:
                runner.run(coro)
            if sys.version_info >= (3, 11):
                assert sum("died" in note for note in info.value.__notes__) == 1


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
