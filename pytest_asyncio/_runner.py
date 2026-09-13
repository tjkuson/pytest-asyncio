"""Run the async fixtures and tests of an event loop scope, each in a task."""

from __future__ import annotations

import asyncio
import contextvars
import sys
from collections.abc import AsyncGenerator, Callable, Coroutine
from typing import Any, Generic, TypeVar

if sys.version_info >= (3, 11):
    from asyncio import Runner
    from typing import Self
else:
    from backports.asyncio.runner import Runner
    from typing_extensions import Self

_T = TypeVar("_T")

_REFUSED_MESSAGE = (
    "The coroutine was refused: an async generator fixture of this event loop "
    "was cancelled at its yield, by a task group, cancel scope or timeout "
    "spanning the yield. Only fixture teardowns run until the loop is closed."
)
_DID_NOT_STOP_MESSAGE = "Async generator fixture didn't stop. Yield only once."
_RUNNING_LOOP_MESSAGE = (
    "pytest-asyncio fixtures cannot be requested from a running event loop"
)
_ABANDONED_MESSAGE = (
    "Exception from a fixture or test that pytest-asyncio abandoned after a "
    "second interruption"
)


class TaskRunner:
    """
    Run the async fixtures and tests of an event loop scope, each in a task.

    A test or a coroutine fixture runs in a task of its own, in a copy of
    the given context, as with :meth:`asyncio.Runner.run`; the task ends
    before :meth:`run` returns. An async generator fixture runs in a task of
    its own that outlives the synchronous call that started it: it stays
    alive across the fixture's ``yield``, so that the task that entered a
    task group, cancel scope or timeout before the ``yield`` is the task
    that exits it at teardown (see :class:`FixtureTask`). Those tasks are
    the children of the loop scope's task group (see _FixtureTasks), which
    lives as long as the runner and joins them when the runner closes.

    A cancellation of a fixture's task while it waits at the ``yield``, e.g.
    by a task group whose child failed, ends the normal work of the loop.
    The test or fixture setup running at the time is cancelled, and from
    then on only fixture teardowns run, until the loop closes; anything
    else is refused. The fixture's own teardown exits the scope, which
    reports what happened.

    An interruption of a wait (SIGINT, or a BaseException raised by a
    callback of the loop) cancels the task being waited for and waits for
    it to end, so that it cleans up before pytest tears down what it may be
    using. A second interruption abandons the wait: the task is cancelled
    again and what it ends with is reported to the loop's exception handler.
    """

    def __init__(
        self,
        *,
        debug: bool | None = None,
        loop_factory: Callable[[], asyncio.AbstractEventLoop] | None = None,
    ) -> None:
        self._runner = Runner(debug=debug, loop_factory=loop_factory)
        # The task a fixture's cancellation is to cancel (see
        # _fixture_cancelled): the test or fixture setup pytest waits for.
        self._cancellable: asyncio.Task[Any] | None = None
        self._teardown_only = False

    def __enter__(self) -> Self:
        self._runner.__enter__()
        self._fixtures = _FixtureTasks(self._runner)
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def get_loop(self) -> asyncio.AbstractEventLoop:
        return self._runner.get_loop()

    def run(
        self,
        coro: Coroutine[Any, Any, _T],
        *,
        context: contextvars.Context,
        name: str | None = None,
    ) -> _T:
        """Run the coroutine in a task of its own and return its result."""
        if _in_running_loop() or self._teardown_only:
            coro.close()
            if self._teardown_only:
                raise asyncio.CancelledError(_REFUSED_MESSAGE)
            raise RuntimeError(_RUNNING_LOOP_MESSAGE)
        task = _create_task(self.get_loop(), coro, context=context, name=name)
        return self._wait_for(task, _Outcome(task), cancellable=True)

    def start_fixture(
        self,
        gen: AsyncGenerator[_T],
        *,
        context: contextvars.Context,
        name: str | None = None,
    ) -> FixtureTask[_T]:
        """Run the fixture up to its yield, in a task of its own."""
        if _in_running_loop():
            raise RuntimeError(_RUNNING_LOOP_MESSAGE)
        if self._teardown_only:
            raise asyncio.CancelledError(_REFUSED_MESSAGE)
        fixture: FixtureTask[_T] = FixtureTask(self, gen)
        self._fixtures.spawn(fixture, context=context, name=name)
        self._wait_for(fixture.task, fixture.setup, cancellable=True)
        return fixture

    def finish_fixture(self, fixture: FixtureTask[Any]) -> None:
        """Run the fixture from its yield to its end, in its task."""
        if _in_running_loop():
            raise RuntimeError(_RUNNING_LOOP_MESSAGE)
        fixture.resume()
        self._wait_for(fixture.task, fixture.teardown, cancellable=False)

    def close(self) -> None:
        try:
            # Fixtures still alive (their teardown was never requested, or
            # was abandoned) are closed in their tasks, which the task group
            # joins as it exits.
            for fixture in self._fixtures.alive():
                fixture.close()
            self._fixtures.close()
        finally:
            self._runner.close()

    def _wait_for(
        self, task: asyncio.Task[Any], outcome: _Outcome[_T], *, cancellable: bool
    ) -> _T:
        """Run the loop until the outcome is settled, and return it."""
        self._cancellable = task if cancellable else None
        try:
            try:
                self._runner.run(_wait_until(outcome.settled))
            except BaseException as interruption:
                self._interrupt(task, outcome, interruption)
        finally:
            self._cancellable = None
        return outcome.settled.result()

    def _interrupt(
        self, task: asyncio.Task[Any], outcome: _Outcome[Any], exc: BaseException
    ) -> None:
        """
        End the interrupted task, then raise the interruption.

        The task is cancelled and run until it has settled the outcome, so
        that it cleans up before pytest goes on. A failure of that cleanup,
        not the cancellation, is then reported instead of the interruption,
        as asyncio.Runner reports it and as for a synchronous test; an
        outcome that ended, or suppressed, the cancellation does not
        suppress the interruption. A second interruption cancels the task
        again and abandons the outcome (see _Outcome.abandon).
        """
        if not outcome.settled.done():
            task.cancel()
            try:
                self._runner.run(_wait_until(outcome.settled))
            except BaseException:
                outcome.abandon()
                task.cancel()
                raise
        if _failure(outcome.settled) is None:
            raise exc

    def _fixture_cancelled(self) -> None:
        """A fixture's task was cancelled at its yield (see FixtureTask)."""
        self._teardown_only = True
        if self._cancellable is not None:
            self._cancellable.cancel()


class _FixtureTasks:
    """
    The task group of a loop scope, owning the tasks of its generator
    fixtures.

    A root task enters the group and keeps it entered until the runner
    closes; fixture tasks are its children, so their lifetimes are bounded
    by the root's block and the group joins them when it exits. A fixture
    delivers its outcomes to pytest through futures and never lets an
    exception reach the group (see FixtureTask), so one fixture's failure
    cannot cancel another's task; a failure that does reach the group is
    the runner's own bug, and ends the root loudly. The root has nothing of
    its own to clean up, so a cancellation of it (a test cancelling every
    task, say) is ignored until the runner closes.

    Python 3.10 has no task group: there the tasks are joined at close,
    with the same interface.
    """

    def __init__(self, runner: Runner) -> None:
        self._runner = runner
        loop = runner.get_loop()
        # The fixtures whose tasks are alive: the group's children.
        self._alive: set[FixtureTask[Any]] = set()
        self._closing = loop.create_future()
        if sys.version_info >= (3, 11):
            self._group: asyncio.TaskGroup | None = None
            self._root = loop.create_task(self._live(), name="pytest-asyncio")
            # Entered, or ended before entering (cancelled by a task factory).
            self._entered: _Outcome[None] = _Outcome(self._root)
            runner.run(_wait_until(self._entered.settled))
            self._entered.settled.result()

    def alive(self) -> list[FixtureTask[Any]]:
        return list(self._alive)

    def spawn(
        self,
        fixture: FixtureTask[Any],
        *,
        context: contextvars.Context,
        name: str | None,
    ) -> None:
        if sys.version_info >= (3, 11):
            assert self._group is not None
            task = context.run(self._group.create_task, fixture.live(), name=name)
        else:
            loop = self._runner.get_loop()
            task = _create_task(loop, fixture.live(), context=context, name=name)
        fixture.start(task)
        self._alive.add(fixture)
        task.add_done_callback(lambda _: self._alive.discard(fixture))

    def close(self) -> None:
        """Exit the group: joins every fixture task, then the root ends."""
        if sys.version_info >= (3, 11):
            self._closing.set_result(None)
            self._runner.run(_wait_until(self._root))
            self._root.result()
        elif self._alive:
            self._runner.run(_wait_until(*(f.task for f in self._alive)))

    if sys.version_info >= (3, 11):

        async def _live(self) -> None:
            async with asyncio.TaskGroup() as group:
                self._group = group
                self._entered.set_result(None)
                while not self._closing.done():
                    try:
                        await asyncio.shield(self._closing)
                    except asyncio.CancelledError:
                        pass
            # Left the block: every fixture task has ended.


class _Outcome(Generic[_T]):
    """
    What a task's phase ends with: a result, an exception, or a cancellation.

    Settled by the phase when it ends, or by the task when it ends first
    (e.g. cancelled before its first step, it never runs the phase), so a
    wait for the outcome cannot outlive the task. Once the waiter has given
    up (:meth:`abandon`), a late exception is reported to the loop's
    exception handler instead, as that of a task nobody awaits.
    """

    def __init__(self, task: asyncio.Task[Any]) -> None:
        self.settled: asyncio.Future[_T] = task.get_loop().create_future()
        self._abandoned = False
        task.add_done_callback(self._task_done)

    def set_result(self, value: _T) -> None:
        if not self.settled.done():
            self.settled.set_result(value)

    def set_exception(self, exc: BaseException) -> None:
        if not self.settled.done():
            self.settled.set_exception(exc)
        elif self._abandoned and not isinstance(exc, asyncio.CancelledError):
            self.settled.get_loop().call_exception_handler(
                {"message": _ABANDONED_MESSAGE, "exception": exc}
            )

    def abandon(self) -> None:
        """Stop waiting: what the phase ends with is reported, not returned."""
        self._abandoned = True
        self.settled.cancel()

    def _task_done(self, task: asyncio.Task[Any]) -> None:
        if task.cancelled():
            self.set_exception(asyncio.CancelledError())
        elif (exc := task.exception()) is not None:
            self.set_exception(exc)
        else:
            self.set_result(task.result())


class FixtureTask(Generic[_T]):
    """
    An async generator fixture, run in a task of its own.

    The task runs the fixture up to its ``yield``, waits there for pytest's
    decision, and acts on it: run the fixture to its end (``resume``), or
    close it because the runner is closing (``close``). While it waits, a
    cancellation of the task is a scope's (a task group whose child failed,
    a cancel scope, a timeout): the runner is told, once, and the task
    keeps waiting, so that the scope is exited by its own task at teardown.
    Every outcome of the fixture is delivered through :attr:`setup` and
    :attr:`teardown`; nothing escapes the task.
    """

    task: asyncio.Task[None]
    # The outcomes of the two phases, for the runner to wait for.
    setup: _Outcome[_T]
    teardown: _Outcome[None]

    def __init__(self, runner: TaskRunner, gen: AsyncGenerator[_T]) -> None:
        loop = runner.get_loop()
        self._runner = runner
        self._gen = gen
        # pytest's decision for the fixture at its yield: resume or close.
        self._decision: asyncio.Future[Callable[[], Coroutine[Any, Any, object]]]
        self._decision = loop.create_future()
        self._aclose = gen.aclose
        # A copy of the task's context once the fixture was set up.
        self.context_after: contextvars.Context | None = None

    def start(self, task: asyncio.Task[None]) -> None:
        """The task now runs :meth:`live`; its outcomes are settled by it."""
        self.task = task
        self.setup = _Outcome(task)
        self.teardown = _Outcome(task)

    @property
    def value(self) -> _T:
        return self.setup.settled.result()

    def resume(self) -> None:
        """Have the task run the fixture from its yield to its end."""
        self._decision.set_result(self._gen.__anext__)

    def close(self) -> None:
        """Have the task close the fixture (the runner is closing)."""
        self.teardown.abandon()
        if not self._decision.done():
            self._decision.set_result(self._aclose)
        else:
            # The teardown pytest asked for was abandoned; it ends when
            # cancelled again.
            self.task.cancel()

    async def live(self) -> None:
        """The fixture's whole life, as the body of its task."""
        try:
            value = await self._gen.__anext__()
        except BaseException as exc:
            self.setup.set_exception(exc)
            return
        self.context_after = contextvars.copy_context()
        self.setup.set_result(value)
        cancellation_reported = False
        while not self._decision.done():
            try:
                await asyncio.shield(self._decision)
            except asyncio.CancelledError:
                # A scope's, while the fixture is in use.
                if not cancellation_reported:
                    cancellation_reported = True
                    self._runner._fixture_cancelled()
        try:
            await self._decision.result()()
        except StopAsyncIteration:
            self.teardown.set_result(None)
        except BaseException as exc:
            self.teardown.set_exception(exc)
        else:
            # aclose() returns; __anext__() returning means a second yield.
            if self._decision.result() is self._aclose:
                self.teardown.set_result(None)
            else:
                self.teardown.set_exception(ValueError(_DID_NOT_STOP_MESSAGE))


def _create_task(
    loop: asyncio.AbstractEventLoop,
    coro: Coroutine[Any, Any, Any],
    *,
    context: contextvars.Context,
    name: str | None,
) -> asyncio.Task[Any]:
    # The task copies the current context, as any task does; here that is a
    # copy of the given one.
    return context.run(loop.create_task, coro, name=name)


async def _wait_until(*futures: asyncio.Future[Any]) -> None:
    # Unlike awaiting the futures directly, this neither cancels them when
    # the waiter is cancelled nor raises their exceptions.
    await asyncio.wait(futures)


def _failure(future: asyncio.Future[Any]) -> BaseException | None:
    """The exception of the done future, unless it is a cancellation."""
    exc = None if future.cancelled() else future.exception()
    return None if isinstance(exc, asyncio.CancelledError) else exc


def _in_running_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True
