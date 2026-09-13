"""Run the async fixtures and tests of an event loop scope, each in a task."""

from __future__ import annotations

import asyncio
import contextvars
import enum
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
    "was cancelled at its yield (by a task group, cancel scope or timeout "
    "spanning the yield), or a task of pytest-asyncio's own was cancelled. "
    "Only fixture teardowns run until the loop is closed."
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

    Each task runs in a copy of the context given for it, as a task copies
    the context it is created in. A test or a coroutine fixture runs in a
    task of its own, which :meth:`run` creates on the loop and waits for;
    an async generator fixture runs in a task of its own that outlives the
    synchronous call which started it: it stays alive across the fixture's
    ``yield``, so that the task that entered a task group, cancel scope or
    timeout before the ``yield`` is the task that exits it at teardown
    (see :class:`FixtureTask`). Those tasks are the children of a task
    group of the runner's own, which lives as long as the runner and joins
    them as it exits (see _FixtureGroup). A test's task is not a child of
    that group: a task that cancels the tasks of its loop, its owner among
    them, and waits for them to end could never end.

    A cancellation of one of the runner's own tasks, e.g. of a fixture's
    task waiting at the ``yield`` by a task group whose child failed, ends
    the normal work of the loop: the test or fixture setup running at the
    time is cancelled, and from then on only fixture teardowns run, until
    the loop closes; anything else is refused. The fixture's own teardown
    exits the scope, which reports what happened.

    An interruption of a wait (SIGINT, or a BaseException raised by a
    callback of the loop) cancels the task being waited for and waits for
    it to end, so that it cleans up before pytest tears down what it may be
    using. A second interruption abandons the wait: the task is cancelled
    again and left to end on its own; what it ended or ends with is
    reported to the loop's exception handler, and the runner keeps the task
    until it closes the loop, where asyncio cancels the tasks still alive
    once more and waits for them.

    :meth:`open` opens the loop and the group, :meth:`close` closes them;
    ``with`` does both.
    """

    def __init__(
        self,
        *,
        debug: bool | None = None,
        loop_factory: Callable[[], asyncio.AbstractEventLoop] | None = None,
    ) -> None:
        self._runner = Runner(debug=debug, loop_factory=loop_factory)
        # The task a cancellation of one of the runner's own tasks is to
        # cancel too (see _end_normal_work): the test or fixture setup
        # pytest waits for.
        self._cancellable: asyncio.Task[Any] | None = None
        self._teardown_only = False
        # Owned until they end, at the latest when the loop closes.
        self._abandoned: set[asyncio.Task[Any]] = set()

    def open(self) -> None:
        """Open the loop and the fixtures' task group, or nothing."""
        try:
            self._fixtures = _FixtureGroup.open(self)
        except BaseException:
            self._runner.close()
            raise

    def close(self) -> None:
        """Close the group, joining the fixtures' tasks, then the loop."""
        try:
            self._fixtures.close()
        finally:
            self._runner.close()

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def get_loop(self) -> asyncio.AbstractEventLoop:
        # Runner creates the loop on first use.
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
        loop = self.get_loop()
        task = context.run(loop.create_task, coro, name=name)
        outcome: _Outcome[_T] = _Outcome(loop)
        task.add_done_callback(outcome.settle_from)
        return self._wait_for(task, outcome, cancellable=True)

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
        fixture = self._fixtures.spawn(gen, context=context, name=name)
        self._wait_for(fixture.task, fixture.setup, cancellable=True)
        return fixture

    def finish_fixture(self, fixture: FixtureTask[Any]) -> None:
        """Run the fixture from its yield to its end, in its task."""
        if _in_running_loop():
            raise RuntimeError(_RUNNING_LOOP_MESSAGE)
        fixture.resume()
        self._wait_for(fixture.task, fixture.teardown, cancellable=False)

    def _run_until(self, *futures: asyncio.Future[Any]) -> None:
        """Run the loop until the futures are done; an interruption raises."""
        self._runner.run(_wait_until(*futures))

    def _wait_for(
        self, task: asyncio.Task[Any], outcome: _Outcome[_T], *, cancellable: bool
    ) -> _T:
        """Run the loop until the outcome is settled, and return it."""
        self._cancellable = task if cancellable else None
        try:
            try:
                self._run_until(outcome.settled)
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
        suppress the interruption. A second interruption abandons the
        outcome (see _Outcome.abandon) and the task: cancelled once more,
        it is left to end on its own, owned until it does.
        """
        if not outcome.settled.done():
            task.cancel()
            try:
                self._run_until(outcome.settled)
            except BaseException:
                outcome.abandon()
                task.cancel()
                self._abandoned.add(task)
                task.add_done_callback(self._abandoned.discard)
                raise
        if _failure(outcome.settled) is None:
            raise exc

    def _end_normal_work(self) -> None:
        """One of the runner's own tasks was cancelled: only teardowns from now on."""
        self._teardown_only = True
        if self._cancellable is not None:
            self._cancellable.cancel()
            self._cancellable = None


if sys.version_info >= (3, 11):

    class _FixtureGroup:
        """
        The task group of a loop scope's async generator fixtures.

        A root task of the runner's, named ``pytest-asyncio``, enters an
        :class:`asyncio.TaskGroup` when the runner opens and exits it when
        the runner closes; the fixtures' tasks are the group's children,
        so their lifetimes are bounded by the root's block and the group
        joins them as it exits. A fixture delivers what its phases end with
        to pytest and lets no exception reach the group (see FixtureTask),
        so one fixture's failure cannot cancel another's task; an exception
        that does reach the group is the runner's own bug, which the group
        raises when the runner closes.

        A cancellation of the root is a cancellation of the group: the root
        tells the runner to end the normal work of the loop, and the group
        cancels its children and waits for them, as any task group does.
        The fixtures, cancelled at their yield, keep waiting for pytest to
        tear them down, in its order; the root ends once they have ended,
        at once if there are none.
        """

        @classmethod
        def open(cls, runner: TaskRunner) -> Self:
            """Start the root and run the loop until it has entered the group."""
            loop = runner.get_loop()
            closing: asyncio.Future[None] = loop.create_future()
            entered: _Outcome[asyncio.TaskGroup] = _Outcome(loop)
            live = cls._live(runner, entered, closing)
            root = loop.create_task(live, name="pytest-asyncio")
            root.add_done_callback(entered.settle_from)
            runner._run_until(entered.settled)
            # A root cancelled before it entered (by a task factory of the
            # loop, say) raises here, and the runner opens nothing.
            return cls(runner, root, entered.settled.result(), closing)

        def __init__(
            self,
            runner: TaskRunner,
            root: asyncio.Task[None],
            group: asyncio.TaskGroup,
            closing: asyncio.Future[None],
        ) -> None:
            self._runner = runner
            self._root = root
            self._group = group
            self._closing = closing
            # The fixtures whose tasks are alive, for close() to decide.
            self._alive: set[FixtureTask[Any]] = set()

        def spawn(
            self,
            gen: AsyncGenerator[_T],
            *,
            context: contextvars.Context,
            name: str | None,
        ) -> FixtureTask[_T]:
            """Start the fixture in a task of the group."""
            fixture = FixtureTask(
                self._runner,
                gen,
                lambda live: context.run(self._group.create_task, live, name=name),
            )
            self._alive.add(fixture)
            fixture.task.add_done_callback(lambda _: self._alive.discard(fixture))
            return fixture

        def close(self) -> None:
            """Close the fixtures still alive, and exit the group: it joins them."""
            for fixture in list(self._alive):
                fixture.close()
            self._closing.set_result(None)
            self._runner._run_until(self._root)
            failure = _failure(self._root)
            if failure is not None:
                raise failure

        @staticmethod
        async def _live(
            runner: TaskRunner,
            entered: _Outcome[asyncio.TaskGroup],
            closing: asyncio.Future[None],
        ) -> None:
            async with asyncio.TaskGroup() as group:
                entered.set_result(group)
                try:
                    await _wait_until(closing)
                except asyncio.CancelledError:
                    runner._end_normal_work()
                    raise

else:

    class _FixtureGroup:
        """
        The tasks of a loop scope's async generator fixtures, on a Python
        without :class:`asyncio.TaskGroup`: joined by hand when the runner
        closes.
        """

        @classmethod
        def open(cls, runner: TaskRunner) -> Self:
            runner.get_loop()
            return cls(runner)

        def __init__(self, runner: TaskRunner) -> None:
            self._runner = runner
            self._alive: set[FixtureTask[Any]] = set()

        def spawn(
            self,
            gen: AsyncGenerator[_T],
            *,
            context: contextvars.Context,
            name: str | None,
        ) -> FixtureTask[_T]:
            loop = self._runner.get_loop()
            fixture = FixtureTask(
                self._runner,
                gen,
                lambda live: context.run(loop.create_task, live, name=name),
            )
            self._alive.add(fixture)
            fixture.task.add_done_callback(lambda _: self._alive.discard(fixture))
            return fixture

        def close(self) -> None:
            alive = list(self._alive)
            for fixture in alive:
                fixture.close()
            if alive:
                self._runner._run_until(*(fixture.task for fixture in alive))


class _Outcome(Generic[_T]):
    """
    What an operation of a task ends with, for pytest to wait for.

    Settled by the operation, or by the end of the task if that comes
    first (:meth:`settle_from`), so that a wait for the outcome cannot
    outlive the task. Once pytest has stopped waiting (:meth:`abandon`),
    what the operation ended or ends with is reported to the loop's
    exception handler instead, as the error of a task nobody awaits.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.settled: asyncio.Future[_T] = loop.create_future()
        self._abandoned = False

    def set_result(self, value: _T) -> None:
        if not self.settled.done():
            self.settled.set_result(value)

    def set_exception(self, exc: BaseException) -> None:
        if not self.settled.done():
            self.settled.set_exception(exc)
        elif self._abandoned:
            self._report(exc)

    def cancel(self) -> None:
        if not self.settled.done():
            self.settled.cancel()

    def settle_from(self, task: asyncio.Task[Any]) -> None:
        """Settle with what the task ended with (a done callback)."""
        if task.cancelled():
            self.cancel()
        elif (exc := task.exception()) is not None:
            self.set_exception(exc)
        else:
            self.set_result(task.result())

    def abandon(self) -> None:
        """Stop waiting: what the operation ended or ends with is reported."""
        if self._abandoned:
            return
        self._abandoned = True
        if not self.settled.done():
            self.settled.cancel()
        elif (exc := _failure(self.settled)) is not None:
            self._report(exc)

    def _report(self, exc: BaseException) -> None:
        if not isinstance(exc, asyncio.CancelledError):
            self.settled.get_loop().call_exception_handler(
                {"message": _ABANDONED_MESSAGE, "exception": exc}
            )


class _Decision(enum.Enum):
    """pytest's decision for a fixture waiting at its yield."""

    RESUME = "run the fixture to its end"
    CLOSE = "close the fixture: the runner is closing"


class FixtureTask(Generic[_T]):
    """
    An async generator fixture, run in a task of its own.

    The task runs the fixture up to its ``yield``, waits there for pytest's
    decision, and acts on it: it runs the fixture to its end
    (:meth:`resume`), or closes it because the runner is closing
    (:meth:`close`). A cancellation of the task while it waits, whether by
    a scope spanning the ``yield`` or by a direct call, ends the normal
    work of the loop; the task keeps waiting, so that the fixture's
    teardown, in this task, is what exits the scope.

    What each phase ends with goes to :attr:`setup` and :attr:`teardown`,
    for pytest to wait for; nothing escapes the task. A phase still pending
    when the task ends (cancelled before its first step, say) ends with
    the task.
    """

    def __init__(
        self,
        runner: TaskRunner,
        gen: AsyncGenerator[_T],
        create_task: Callable[[Coroutine[Any, Any, None]], asyncio.Task[None]],
    ) -> None:
        loop = runner.get_loop()
        self._runner = runner
        self._gen = gen
        self._decision: asyncio.Future[_Decision] = loop.create_future()
        self.setup: _Outcome[_T] = _Outcome(loop)
        self.teardown: _Outcome[None] = _Outcome(loop)
        # A copy of the task's context once the fixture was set up.
        self.context_after: contextvars.Context | None = None
        self.task = create_task(self.live())
        self.task.add_done_callback(self._task_done)

    @property
    def value(self) -> _T:
        return self.setup.settled.result()

    def resume(self) -> None:
        """Have the task run the fixture from its yield to its end."""
        self._decision.set_result(_Decision.RESUME)

    def close(self) -> None:
        """Have the task close the fixture; nobody waits for the outcome."""
        self.teardown.abandon()
        if not self._decision.done():
            self._decision.set_result(_Decision.CLOSE)
        else:
            # The teardown pytest asked for was abandoned: ask it to end.
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
        while not self._decision.done():
            try:
                await _wait_until(self._decision)
            except asyncio.CancelledError:
                self._runner._end_normal_work()
        if self._decision.result() is _Decision.CLOSE:
            try:
                await self._gen.aclose()
            except BaseException as exc:
                self.teardown.set_exception(exc)
            else:
                self.teardown.set_result(None)
            return
        try:
            await self._gen.__anext__()
        except StopAsyncIteration:
            self.teardown.set_result(None)
        except BaseException as exc:
            self.teardown.set_exception(exc)
        else:
            self.teardown.set_exception(ValueError(_DID_NOT_STOP_MESSAGE))

    def _task_done(self, task: asyncio.Task[None]) -> None:
        for phase in (self.setup, self.teardown):
            if not phase.settled.done():
                phase.settle_from(task)
                return


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
