"""The runner of an event loop scope: a task per fixture and test."""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import gc
import inspect
import sys
import weakref
from collections.abc import AsyncGenerator, Callable, Iterator
from typing import Any

import pytest
from pytest_asyncio._runner import FixtureTask, TaskRunner

_REQUIRES_311 = pytest.mark.skipif(
    sys.version_info < (3, 11), reason="asyncio.TaskGroup needs Python 3.11"
)
if sys.version_info >= (3, 11):
    from builtins import BaseExceptionGroup as _BaseExceptionGroup
else:
    _BaseExceptionGroup = BaseException
_REFUSED = "no longer accepts new tests or fixture setups"
_var: contextvars.ContextVar[str] = contextvars.ContextVar("var")


def _task() -> asyncio.Task[object]:
    task = asyncio.current_task()
    assert task is not None
    return task


def _interrupt(loop: asyncio.AbstractEventLoop) -> None:
    """Interrupt the wait for the running coroutine, as SIGINT does."""

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


def _copy() -> contextvars.Context:
    return contextvars.copy_context()


async def _get() -> str:
    return _var.get("unset")


async def _set(value: str) -> None:
    _var.set(value)


async def _nothing() -> None:
    pass


async def _never_started() -> None:
    raise AssertionError("the coroutine was started")


async def _plain_fixture() -> AsyncGenerator[str]:
    yield "value"


def _run(runner: TaskRunner, coro, **kwargs):
    return runner.run(coro, context=_copy(), **kwargs)


def _start(runner: TaskRunner, gen, **kwargs):
    return runner.start_fixture(gen, context=_copy(), **kwargs)


def _assert_teardown_only(runner: TaskRunner) -> None:
    refused = _never_started()
    with pytest.raises(RuntimeError, match=_REFUSED):
        _run(runner, refused)
    assert inspect.getcoroutinestate(refused) == inspect.CORO_CLOSED
    with pytest.raises(RuntimeError, match=_REFUSED):
        _start(runner, _plain_fixture())


def test_a_generator_fixture_lives_in_one_task_across_its_yield():
    """The task that set the fixture up is the task that tears it down."""
    tasks = []

    async def fixture() -> AsyncGenerator[str]:
        tasks.append(_task())
        yield "value"
        tasks.append(_task())

    with TaskRunner() as runner:
        started = _start(runner, fixture(), name="fixture")
        assert started.setup_result().value == "value"
        assert started.task.get_name() == "fixture"
        runner.finish_fixture(started)
    assert len(tasks) == 2
    assert tasks[0] is tasks[1] is started.task


def test_tests_and_coroutine_fixtures_run_in_tasks_of_their_own():
    """Each coroutine is a task of its own, named as asked."""
    seen = []

    async def record() -> None:
        seen.append(_task())

    with TaskRunner() as runner:
        _run(runner, record(), name="first")
        _run(runner, record(), name="second")
    assert seen[0] is not seen[1]
    assert [task.get_name() for task in seen] == ["first", "second"]


@pytest.mark.parametrize("error_type", [ValueError, KeyboardInterrupt, SystemExit])
def test_run_raises_what_the_coroutine_raised(error_type: type[BaseException]):
    async def raise_it() -> None:
        raise error_type("from the coroutine")

    with TaskRunner() as runner:
        with pytest.raises(error_type, match="from the coroutine"):
            _run(runner, raise_it())
        _run(runner, _nothing())


def test_a_fixture_that_fails_to_set_up_raises_from_start_fixture():
    async def broken() -> AsyncGenerator[None]:
        raise ValueError("setup failed")
        yield

    with TaskRunner() as runner:
        with pytest.raises(ValueError, match="setup failed"):
            _start(runner, broken())
        _run(runner, _nothing())


def test_a_fixture_that_yields_twice_or_fails_at_teardown_raises_from_finish():
    async def yields_twice() -> AsyncGenerator[None]:
        yield
        yield

    async def fails() -> AsyncGenerator[None]:
        yield
        raise ValueError("teardown failed")

    with TaskRunner() as runner:
        with pytest.raises(ValueError, match="yielded more than once"):
            runner.finish_fixture(_start(runner, yields_twice()))
        with pytest.raises(ValueError, match="teardown failed"):
            runner.finish_fixture(_start(runner, fails()))


def test_run_is_refused_from_a_running_loop():
    async def job() -> None:
        inner = _never_started()
        with pytest.raises(RuntimeError, match="while the event loop is running"):
            _run(runner, inner)
        assert inspect.getcoroutinestate(inner) == inspect.CORO_CLOSED
        with pytest.raises(RuntimeError, match="while the event loop is running"):
            _start(runner, _plain_fixture())

    with TaskRunner() as runner:
        _run(runner, job())


def test_a_coroutine_runs_in_a_copy_of_the_given_context():
    """It sees the context's variables; what it sets goes nowhere else."""
    context = _copy()
    with TaskRunner() as runner:
        with _set_in_sync("from sync"):
            assert _run(runner, _get()) == "from sync"
        runner.run(_set("in coroutine"), context=context)
        assert context.get(_var, "unset") == "unset"
        assert _var.get("unset") == "unset"
        assert runner.run(_get(), context=context) == "unset"


def test_a_fixture_keeps_its_context_from_setup_to_teardown():
    """Setup and teardown share the fixture task's context; tokens reset."""
    seen = []

    async def fixture() -> AsyncGenerator[None]:
        token = _var.set("from fixture")
        yield
        seen.append(_var.get())
        _var.reset(token)
        seen.append(_var.get("unset"))

    with TaskRunner() as runner:
        started = _start(runner, fixture())
        assert started.setup_result().context[_var] == "from fixture"
        assert _run(runner, _get()) == "unset"
        runner.finish_fixture(started)
    assert seen == ["from fixture", "unset"]


def test_the_task_of_a_coroutine_owns_its_context():
    """
    A native task: its context is the coroutine's, and its await chain, seen
    while the coroutine is suspended, leads to the coroutine's own frame.
    """

    def chain_of(task: asyncio.Task[Any]) -> list[str]:
        names = []
        coro: Any = task.get_coro()
        while coro is not None and hasattr(coro, "cr_frame"):
            names.append(coro.cr_frame.f_code.co_name)
            coro = coro.cr_await
        return names

    async def job() -> tuple[str, list[str]]:
        _var.set("in job")
        task = _task()
        loop = asyncio.get_running_loop()
        seen = loop.create_future()
        loop.call_soon(
            lambda: seen.set_result((_var.get("unset"), chain_of(task))),
            context=task.get_context() if sys.version_info >= (3, 12) else None,
        )
        return await seen

    with TaskRunner() as runner:
        value, chain = _run(runner, job())
    assert value == "in job"
    assert chain[-1] == "job"


@_REQUIRES_311
@pytest.mark.parametrize(
    "when", ["during a teardown", "during a test", "during a setup"]
)
def test_a_fixture_cancelled_at_its_yield_ends_the_normal_work_of_the_loop(
    when: str,
):
    """
    A scope cancelling the fixture's task at its yield (a task group whose
    child failed, say): the test or fixture setup running at the time is
    cancelled, a teardown is not, nothing but fixture teardowns runs from
    then on, and the fixture's teardown exits the scope, which reports what
    happened.
    """
    log = []

    async def fixture() -> AsyncGenerator[asyncio.Event]:
        trigger = asyncio.Event()

        async def fail() -> None:
            await trigger.wait()
            raise ValueError("child failed")

        async with asyncio.TaskGroup() as group:
            group.create_task(fail())
            yield trigger
        log.append("group exited")

    async def test(trigger: asyncio.Event) -> None:
        trigger.set()
        try:
            await asyncio.Event().wait()
        finally:
            log.append("test cleaned up")

    async def setup(trigger: asyncio.Event) -> AsyncGenerator[None]:
        trigger.set()
        try:
            await asyncio.Event().wait()
        finally:
            log.append("setup cleaned up")
        yield

    async def teardown(trigger: asyncio.Event) -> AsyncGenerator[None]:
        yield
        trigger.set()
        await asyncio.sleep(0.01)
        log.append("teardown completed")

    with TaskRunner() as runner:
        started = _start(runner, fixture())
        if when == "during a teardown":
            runner.finish_fixture(
                _start(runner, teardown(started.setup_result().value))
            )
            assert log == ["teardown completed"]
        elif when == "during a test":
            with pytest.raises(asyncio.CancelledError):
                _run(runner, test(started.setup_result().value))
            assert log == ["test cleaned up"]
        else:
            with pytest.raises(asyncio.CancelledError):
                _start(runner, setup(started.setup_result().value))
            assert log == ["setup cleaned up"]
        _assert_teardown_only(runner)
        with pytest.raises(_BaseExceptionGroup) as info:
            runner.finish_fixture(started)
        assert [str(exc) for exc in info.value.exceptions] == ["child failed"]
        _assert_teardown_only(runner)


@_REQUIRES_311
def test_a_dependent_fixture_is_torn_down_in_its_own_task_untouched():
    """A cancelled scope in one fixture does not cancel another's teardown."""
    log = []

    async def service() -> AsyncGenerator[asyncio.Event]:
        trigger = asyncio.Event()

        async def fail() -> None:
            await trigger.wait()
            raise ValueError("child failed")

        async with asyncio.TaskGroup() as group:
            group.create_task(fail())
            yield trigger

    async def dependent() -> AsyncGenerator[None]:
        try:
            yield
        finally:
            await asyncio.sleep(0.01)
            log.append("dependent cleaned up")

    async def test(trigger: asyncio.Event) -> None:
        trigger.set()
        await asyncio.Event().wait()

    with TaskRunner() as runner:
        started = _start(runner, service())
        depends = _start(runner, dependent())
        with pytest.raises(asyncio.CancelledError):
            _run(runner, test(started.setup_result().value))
        runner.finish_fixture(depends)
        assert log == ["dependent cleaned up"]
        with pytest.raises(_BaseExceptionGroup):
            runner.finish_fixture(started)


def test_a_test_cancelling_its_own_task_affects_nothing_else():
    async def cancel_self() -> None:
        _task().cancel()
        await asyncio.sleep(0)

    with TaskRunner() as runner:
        started = _start(runner, _plain_fixture())
        with pytest.raises(asyncio.CancelledError):
            _run(runner, cancel_self())
        _run(runner, _nothing())
        runner.finish_fixture(started)


@pytest.mark.parametrize("cleanup", ["propagate", "suppress", "fail"])
def test_interrupted_coroutine_is_cancelled_and_joined(cleanup: str):
    """
    The task of an interrupted coroutine is cancelled and run until it has
    ended: its cleanup completes before run() raises. A cleanup that fails
    is the coroutine's outcome (the cancellation its context); one that ends
    or suppresses the cancellation does not suppress the interruption. Other
    tasks are untouched: the loop goes on serving.
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
        started = _start(runner, _plain_fixture())
        expected = ValueError if cleanup == "fail" else KeyboardInterrupt
        with pytest.raises(expected) as info:
            _run(runner, job())
        assert log == ["cleaned up"]
        if cleanup == "fail":
            assert isinstance(info.value.__context__, asyncio.CancelledError)
        _run(runner, _nothing())
        runner.finish_fixture(started)


def test_interrupted_fixture_teardown_is_cancelled_and_joined():
    log = []

    async def fixture() -> AsyncGenerator[None]:
        yield
        _interrupt(asyncio.get_running_loop())
        try:
            await asyncio.Event().wait()
        finally:
            log.append("teardown cleaned up")

    with TaskRunner() as runner:
        started = _start(runner, fixture())
        with pytest.raises(KeyboardInterrupt):
            runner.finish_fixture(started)
        assert log == ["teardown cleaned up"]
        _run(runner, _nothing())


@pytest.mark.parametrize("then", ["propagate", "suppress", "fail"])
def test_second_interruption_abandons_the_coroutine(then: str):
    """
    A wait for the coroutine's cleanup that is interrupted too ends at
    once; the task is cancelled again and left to end on its own when the
    loop next runs. What it ends with then goes to the loop's exception
    handler, as the error of a task nobody awaits.
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
            _run(runner, job())
        assert log == []
        _run(runner, _nothing())
        assert log == ["ended"]
    failures = [type(context.get("exception")) for context in reported]
    assert failures == ([ValueError] if then == "fail" else [])


def test_interruption_after_the_coroutine_ended_is_only_the_interruption():
    async def job() -> None:
        _interrupt(asyncio.get_running_loop())

    with TaskRunner() as runner:
        with pytest.raises(KeyboardInterrupt):
            _run(runner, job())
        _run(runner, _nothing())


def test_a_fixture_task_cancelled_before_its_first_step_fails_its_setup():
    """
    The fixture's task ended before running the fixture (a task factory
    cancelled it, say): its setup fails with the cancellation instead of
    waiting for a result the task can no longer produce.
    """

    def loop_factory() -> asyncio.AbstractEventLoop:
        loop = asyncio.new_event_loop()

        def task_factory(loop, coro, **kwargs):
            task = asyncio.Task(coro, loop=loop, **kwargs)
            code = getattr(coro, "cr_code", None)
            if code is not None and code.co_name == "_live":
                task.cancel()
            return task

        loop.set_task_factory(task_factory)
        return loop

    reported = []
    with TaskRunner(loop_factory=loop_factory) as runner:
        runner.get_loop().set_exception_handler(lambda loop, ctx: reported.append(ctx))
        with pytest.raises(asyncio.CancelledError):
            _start(runner, _plain_fixture())
        _run(runner, _nothing())
        gc.collect()
    assert reported == []


def test_interruption_before_the_fixture_task_started_fails_its_setup():
    """A queued interruption cancels the task before its first step."""
    with TaskRunner() as runner:
        _interrupt(runner.get_loop())
        with pytest.raises(KeyboardInterrupt):
            _start(runner, _plain_fixture())
        _run(runner, _nothing())


def test_a_fixture_interrupted_after_its_yield_is_torn_down_at_once():
    """
    An interruption of the wait for the setup after the fixture yielded:
    pytest never receives the fixture, so the runner tears it down in its
    task before the setup call raises, as pytest would have later.
    """
    log = []

    async def fixture() -> AsyncGenerator[None]:
        _interrupt(asyncio.get_running_loop())
        yield
        await asyncio.sleep(0)
        log.append("torn down in its task" if _task().get_name() == "f" else "?")

    with TaskRunner() as runner:
        with pytest.raises(KeyboardInterrupt):
            _start(runner, fixture(), name="f")
        assert log == ["torn down in its task"]
        _run(runner, _nothing())


def test_an_abandoned_fixture_teardown_reports_its_late_error():
    """
    A teardown abandoned after a second interruption ends later; what it
    ends with goes to the loop's exception handler, as the error of a task
    nobody awaits, instead of being lost.
    """
    reported = []

    async def fixture() -> AsyncGenerator[None]:
        loop = asyncio.get_running_loop()
        yield
        _interrupt(loop)
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            _interrupt(loop)
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                raise ValueError("late teardown failure") from None

    with TaskRunner() as runner:
        runner.get_loop().set_exception_handler(lambda loop, ctx: reported.append(ctx))
        started = _start(runner, fixture())
        with pytest.raises(KeyboardInterrupt):
            runner.finish_fixture(started)
        assert reported == []
        _run(runner, _nothing())
    assert [str(ctx["exception"]) for ctx in reported] == ["late teardown failure"]
    assert "stopped waiting" in reported[0]["message"]


def test_close_finalises_fixtures_never_torn_down_in_their_tasks():
    """An aborted session: the runner closes each fixture in its own task."""
    log = []

    async def fixture() -> AsyncGenerator[None]:
        task = _task()
        try:
            yield
        finally:
            await asyncio.sleep(0)
            log.append(_task() is task)

    with TaskRunner() as runner:
        _start(runner, fixture())
        _start(runner, fixture())
    assert log == [True, True]


@_REQUIRES_311
def test_close_finalises_a_fixture_cancelled_at_its_yield():
    """
    Even one whose scope was cancelled: its task closes the generator, as
    asyncio does for the async generators left at shutdown, so the scope
    exits and its errors are reported.
    """

    async def fixture() -> AsyncGenerator[asyncio.Event]:
        trigger = asyncio.Event()

        async def fail() -> None:
            await trigger.wait()
            raise ValueError("child failed")

        async with asyncio.TaskGroup() as group:
            group.create_task(fail())
            yield trigger

    reported = []
    with TaskRunner() as runner:
        runner.get_loop().set_exception_handler(lambda loop, ctx: reported.append(ctx))
        started = _start(runner, fixture())
        started.setup_result().value.set()
        with pytest.raises(asyncio.CancelledError):
            _run(runner, asyncio.Event().wait(), name="cancelled as the child fails")
        _assert_teardown_only(runner)
    (context,) = reported
    assert [type(exc) for exc in context["exception"].exceptions] == [
        ValueError,
        GeneratorExit,
    ]


def _root_of(loop: asyncio.AbstractEventLoop) -> asyncio.Task[Any]:
    (root,) = (t for t in asyncio.all_tasks(loop) if t.get_name() == "pytest-asyncio")
    return root


@_REQUIRES_311
def test_the_task_owning_the_generator_fixtures_lives_as_long_as_the_runner():
    """
    A task of the runner's own, named pytest-asyncio, owns the task group
    of the generator fixtures, from the runner's opening to its close.
    """
    seen = {}

    def names() -> set[str]:
        tasks = asyncio.all_tasks() - {_task()}
        return {t.get_name() for t in tasks if not t.get_name().startswith("Task-")}

    async def fixture() -> AsyncGenerator[None]:
        seen["fixture"] = names()
        yield

    async def test(when: str) -> None:
        seen[when] = names()

    with TaskRunner() as runner:
        _run(runner, test("before"))
        started = _start(runner, fixture(), name="fixture")
        _run(runner, test("during"))
        runner.finish_fixture(started)
        _run(runner, test("after"))
        root = runner.get_loop()  # the loop is open until the runner closes
    assert seen["before"] == seen["fixture"] == seen["after"] == {"pytest-asyncio"}
    assert seen["during"] == {"fixture", "pytest-asyncio"}
    assert root.is_closed()


@_REQUIRES_311
def test_a_cancellation_of_the_root_ends_the_normal_work_of_the_loop():
    """A test cancelling every task, say: it is cancelled, the loop is teardown-only."""
    log = []

    async def fixture() -> AsyncGenerator[None]:
        yield
        log.append("torn down")

    async def cancel_the_root() -> None:
        _root_of(asyncio.get_running_loop()).cancel()
        await asyncio.sleep(0)
        log.append("the test went on")

    with TaskRunner() as runner:
        started = _start(runner, fixture())
        with pytest.raises(asyncio.CancelledError):
            _run(runner, cancel_the_root())
        _assert_teardown_only(runner)
        runner.finish_fixture(started)
    assert log == ["torn down"]


def _loop_factory_doing(
    first: Callable[[asyncio.AbstractEventLoop, asyncio.Task[Any]], object],
):
    """A loop factory whose task factory does something to the first task."""
    loops = []

    def loop_factory() -> asyncio.AbstractEventLoop:
        loop = asyncio.new_event_loop()
        loops.append(loop)
        seen_first = False

        def task_factory(loop, coro, **kwargs):
            nonlocal seen_first
            task = asyncio.Task(coro, loop=loop, **kwargs)
            if not seen_first:
                seen_first = True
                first(loop, task)
            return task

        loop.set_task_factory(task_factory)
        return loop

    return loop_factory, loops


@_REQUIRES_311
def test_a_root_cancelled_before_it_entered_fails_the_opening_of_the_runner():
    """The loop's task factory cancels the first task, which is the root."""
    loop_factory, loops = _loop_factory_doing(lambda loop, task: task.cancel())
    with pytest.raises(asyncio.CancelledError), TaskRunner(loop_factory=loop_factory):
        pytest.fail("the runner opened")
    (loop,) = loops
    assert loop.is_closed()


@_REQUIRES_311
def test_a_root_cancelled_right_after_it_entered_makes_the_loop_teardown_only():
    """The factory queues the cancellation: it lands after the root's first step."""
    loop_factory, loops = _loop_factory_doing(
        lambda loop, task: loop.call_soon(task.cancel)
    )
    with TaskRunner(loop_factory=loop_factory) as runner:
        _assert_teardown_only(runner)
    (loop,) = loops
    assert loop.is_closed()


@_REQUIRES_311
def test_an_interruption_while_the_runner_opens_releases_the_loop():
    """A callback queued on the new loop raises before the root's first step."""
    reported = []

    def interrupt(loop: asyncio.AbstractEventLoop, task: asyncio.Task[Any]) -> None:
        loop.set_exception_handler(lambda loop, ctx: reported.append(ctx))
        _interrupt(loop)

    loop_factory, loops = _loop_factory_doing(interrupt)
    with pytest.raises(KeyboardInterrupt), TaskRunner(loop_factory=loop_factory):
        pytest.fail("the runner opened")
    (loop,) = loops
    assert loop.is_closed()
    gc.collect()
    assert reported == []


def test_an_abandoned_coroutine_is_kept_until_the_loop_closes():
    """
    Nobody waits for an abandoned task, but the runner keeps it: a task
    nobody references is garbage collected while pending, and its cleanup
    with it. The loop's close cancels it once more and joins it.
    """
    events = []
    reported = []
    references = []

    async def job() -> None:
        loop = asyncio.get_running_loop()
        references.append(weakref.ref(_task()))
        _interrupt(loop)
        try:
            await asyncio.Event().wait()
        finally:
            events.append("first cleanup")
            _interrupt(loop)
            try:
                await asyncio.Event().wait()
            finally:
                events.append("nested cleanup")
                try:
                    await asyncio.Event().wait()
                finally:
                    events.append("nested cleanup done")

    with TaskRunner() as runner:
        runner.get_loop().set_exception_handler(lambda loop, ctx: reported.append(ctx))
        with pytest.raises(KeyboardInterrupt):
            _run(runner, job())
        _run(runner, _nothing())
        gc.collect()
        assert references[0]() is not None
        assert events == ["first cleanup", "nested cleanup"]
    assert events == ["first cleanup", "nested cleanup", "nested cleanup done"]
    assert reported == []


def test_an_abandoned_coroutine_that_ends_is_released():
    """The runner keeps an abandoned task only until it ends."""
    references = []

    async def job() -> None:
        loop = asyncio.get_running_loop()
        references.append(weakref.ref(_task()))
        _interrupt(loop)
        try:
            await asyncio.Event().wait()
        finally:
            _interrupt(loop)
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0)

    async def turns() -> None:
        for _ in range(3):
            await asyncio.sleep(0)

    with TaskRunner() as runner:
        with pytest.raises(KeyboardInterrupt):
            _run(runner, job())
        _run(runner, turns())
        gc.collect()
        assert references[0]() is None


def test_an_error_settled_just_before_the_second_interruption_is_reported_once():
    """
    The teardown ends with an error in the same loop iteration as the
    interruption of the wait for it: the interruption is raised, and the
    error is reported to the loop's exception handler, once.
    """
    reported = []

    async def fixture() -> AsyncGenerator[None]:
        loop = asyncio.get_running_loop()
        yield
        _interrupt(loop)
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            _interrupt(loop)
            raise ValueError("settled before the interruption") from None

    with TaskRunner() as runner:
        runner.get_loop().set_exception_handler(lambda loop, ctx: reported.append(ctx))
        started = _start(runner, fixture())
        with pytest.raises(KeyboardInterrupt):
            runner.finish_fixture(started)
        _run(runner, _nothing())
        del started
        gc.collect()
    gc.collect()
    assert [str(ctx["exception"]) for ctx in reported] == [
        "settled before the interruption"
    ]


@_REQUIRES_311
def test_a_failure_escaping_a_fixture_task_ends_normal_work_and_is_raised_at_close(
    monkeypatch: pytest.MonkeyPatch,
):
    """
    Fault injection: a bug of the runner's own lets an exception out of a
    fixture's task. The task group cancels the root, which ends the normal
    work of the loop, and the group raises the failure when the runner
    closes; the loop closes all the same.
    """

    async def buggy(self: FixtureTask[Any]) -> None:
        raise RuntimeError("a bug in the runner")

    monkeypatch.setattr(FixtureTask, "_live", buggy)
    runner = TaskRunner()
    with pytest.raises(_BaseExceptionGroup) as info, runner:
        with pytest.raises(RuntimeError, match="a bug in the runner"):
            _start(runner, _plain_fixture())
        _assert_teardown_only(runner)
    assert [str(exc) for exc in info.value.exceptions] == ["a bug in the runner"]
    with pytest.raises(RuntimeError, match="Runner is closed"):
        runner.get_loop()


@pytest.mark.parametrize("cancelled_first", [False, True], ids=["plain", "self_cancel"])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_an_error_raised_after_a_self_cancellation_is_the_outcome(
    error_type: type[BaseException], cancelled_first: bool
):
    """A raised error wins over a pending cancellation, as it does in any task."""

    async def job() -> None:
        if cancelled_first:
            _task().cancel()
        raise error_type("raised by the coroutine")

    with TaskRunner() as runner:
        with pytest.raises(error_type, match="raised by the coroutine"):
            _run(runner, job())
        _run(runner, _nothing())


def test_a_self_cancellation_at_return_is_a_cancellation():
    async def job() -> int:
        _task().cancel()
        return 1

    with TaskRunner() as runner:
        with pytest.raises(asyncio.CancelledError):
            _run(runner, job())
        assert not runner.teardown_only


def test_the_task_of_a_test_ends_without_the_error_pytest_receives():
    """
    A test's failure goes to pytest, not to the task: a done callback on the
    test's own task sees the task end without an exception. Only a
    cancellation ends the task as such.
    """
    seen = {}

    async def failing() -> None:
        _task().add_done_callback(lambda t: seen.update(task_exception=t.exception()))
        raise ValueError("for pytest")

    with TaskRunner() as runner, pytest.raises(ValueError, match="for pytest"):
        _run(runner, failing())
    assert seen == {"task_exception": None}


def test_a_fixture_closed_at_close_reports_its_error_without_claiming_an_interruption():
    reported = []

    async def fixture() -> AsyncGenerator[None]:
        try:
            yield
        finally:
            raise ValueError("cleanup failed at close")

    with TaskRunner() as runner:
        runner.get_loop().set_exception_handler(lambda loop, ctx: reported.append(ctx))
        _start(runner, fixture())
    (context,) = reported
    assert str(context["exception"]) == "cleanup failed at close"
    assert "interruption" not in context["message"]
    assert "stopped waiting" in context["message"]


def test_a_cancellation_at_the_end_of_a_fixture_teardown_is_its_outcome():
    """The teardown is read from the task's end: the task ended cancelled."""

    async def fixture() -> AsyncGenerator[str]:
        yield "ready"
        _task().cancel()

    with TaskRunner() as runner:
        started = _start(runner, fixture())
        assert started.setup_result().value == "ready"
        with pytest.raises(asyncio.CancelledError):
            runner.finish_fixture(started)
        assert started.task.cancelled()


@_REQUIRES_311
def test_a_fixture_task_dying_at_its_yield_fails_its_teardown(
    monkeypatch: pytest.MonkeyPatch,
):
    """
    Fault injection: a signal handler raising in the fixture task's own frame
    while it handles a cancellation at its yield (a timeout plugin's, say)
    ends the task before pytest asks for the teardown. The teardown then
    reads how the task ended instead of waiting for a result nobody makes.
    """

    class TimeoutSignal(BaseException):
        pass

    def raising(self: TaskRunner) -> None:
        raise TimeoutSignal("in the fixture task's frame")

    monkeypatch.setattr(TaskRunner, "_stop_normal_work", raising)
    trigger = asyncio.Event()

    async def fixture() -> AsyncGenerator[None]:
        async def fail() -> None:
            await trigger.wait()
            raise ValueError("child failed")

        async with asyncio.TaskGroup() as group:
            group.create_task(fail())
            yield

    async def test() -> None:
        trigger.set()
        await asyncio.Event().wait()

    runner = TaskRunner()
    with pytest.raises(_BaseExceptionGroup), runner:
        started = _start(runner, fixture())
        with pytest.raises(asyncio.CancelledError):
            _run(runner, test())
        assert started.task.done()
        with pytest.raises(TimeoutSignal):
            runner.finish_fixture(started)
