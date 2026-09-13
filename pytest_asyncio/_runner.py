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

    Every task of the runner's is a child of one task group, entered by a
    root task of the runner's own when the runner opens and exited when it
    closes (see _Lifespan). A test or a coroutine fixture runs in a task
    :meth:`run` waits for; an async generator fixture runs in a task that
    outlives the call which started it, staying alive across the fixture's
    ``yield`` so that the task that entered a task group, cancel scope or
    timeout before the ``yield`` is the task that exits it at teardown (see
    :class:`FixtureTask`). Each task runs in a copy of the context given
    for it, as a task copies the context it is created in. What a task's
    coroutine ends with goes to pytest, through an :class:`_Outcome`; only
    a cancellation ends the task itself, so no test's failure reaches the
    group and cancels the others.

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
    reported to the loop's exception handler, and the runner cancels it
    once more when it closes, before the group joins it.

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
        # The tasks pytest stopped waiting for, to cancel at close.
        self._abandoned: set[asyncio.Task[Any]] = set()

    def open(self) -> None:
        """Open the loop and the task group, or nothing."""
        try:
            self._lifespan = _Lifespan.open(self)
        except BaseException:
            self._runner.close()
            raise

    def close(self) -> None:
        """Exit the group, which joins the runner's tasks, then close the loop."""
        try:
            for task in self._abandoned:
                task.cancel()
            self._lifespan.close()
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

    @property
    def teardown_only(self) -> bool:
        """Whether the normal work of the loop has ended (see the class)."""
        return self._teardown_only

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
        outcome: _Outcome[_T] = _Outcome(self.get_loop())
        task = self._lifespan.spawn(outcome.capture(coro), context=context, name=name)
        task.add_done_callback(outcome.settle_from)
        # A task cancelled before its first step never started the coroutine.
        task.add_done_callback(lambda _: coro.close())
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
        fixture = self._lifespan.spawn_fixture(gen, context=context, name=name)
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
        it is left to end on its own, until the runner closes.
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

    class _Lifespan:
        """
        The task group of a loop scope, owning every task of the runner's.

        A root task of the runner's, named ``pytest-asyncio``, enters an
        :class:`asyncio.TaskGroup` when the runner opens and exits it when
        the runner closes; every task the runner creates is a child of
        that group, so its lifetime is bounded by the root's block and the
        group joins it as it exits. What a child's coroutine ends with
        goes to pytest and never reaches the group (see _Outcome and
        FixtureTask), so one task's failure cannot cancel the others; an
        exception that does reach the group is the runner's own bug,
        which the group raises when the runner closes.

        A cancellation of the root is a cancellation of the group: the root
        tells the runner to end the normal work of the loop, and the group
        cancels its children and waits for them, as any task group does.
        The fixtures, cancelled at their yield, keep waiting for pytest to
        tear them down, in its order; the root ends once every child has
        ended.
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
            coro: Coroutine[Any, Any, _T],
            *,
            context: contextvars.Context,
            name: str | None,
        ) -> asyncio.Task[_T]:
            """Start the coroutine in a task of the group, in a copy of the context."""
            return context.run(self._group.create_task, coro, name=name)

        def spawn_fixture(
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
                lambda live: self.spawn(live, context=context, name=name),
            )
            self._alive.add(fixture)
            fixture.task.add_done_callback(lambda _: self._alive.discard(fixture))
            return fixture

        def close(self) -> None:
            """Close the fixtures still alive; the group joins its tasks as it exits."""
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

    class _Lifespan:
        """
        The tasks of a loop scope, on a Python without
        :class:`asyncio.TaskGroup`: the fixtures' tasks are joined by hand
        when the runner closes, the others by the loop's close.
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
            coro: Coroutine[Any, Any, _T],
            *,
            context: contextvars.Context,
            name: str | None,
        ) -> asyncio.Task[_T]:
            loop = self._runner.get_loop()
            return context.run(loop.create_task, coro, name=name)

        def spawn_fixture(
            self,
            gen: AsyncGenerator[_T],
            *,
            context: contextvars.Context,
            name: str | None,
        ) -> FixtureTask[_T]:
            fixture = FixtureTask(
                self._runner,
                gen,
                lambda live: self.spawn(live, context=context, name=name),
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
        self._raised: BaseException | None = None

    async def capture(self, coro: Coroutine[Any, Any, _T]) -> _T | None:
        """
        The coroutine, as the body of a task the outcome is settled from.

        What the coroutine raises is kept for the outcome instead of ending
        the task with it, so that it reaches pytest and not the task group;
        a cancellation ends the task, as it does any task.
        """
        try:
            return await coro
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            self._raised = exc
            return None

    def set_result(self, value: _T) -> None:
        if not self.settled.done():
            self.settled.set_result(value)

    def set_exception(self, exc: BaseException) -> None:
        if not self.settled.done():
            self.settled.set_exception(exc)
        elif self._abandoned:
            self._report(exc)

    def settle_from(self, task: asyncio.Task[Any]) -> None:
        """Settle with what the task ended with (a done callback)."""
        if self._raised is not None:
            # Kept by capture(); a cancellation pending as the coroutine
            # raised does not replace it, as it does not in any task.
            self.set_exception(self._raised)
            return
        try:
            value = task.result()
        except BaseException as exc:
            self.set_exception(exc)
        else:
            self.set_result(value)

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
