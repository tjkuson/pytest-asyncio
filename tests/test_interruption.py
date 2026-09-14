"""
Interrupting the wait for a test or a fixture.

``Runner.run`` raises on the first SIGINT, when a KeyboardInterrupt or a
pytest-timeout failure escapes the event loop, or when the loop is stopped.
The runner then cancels the task it was waiting for, as asyncio.Runner
cancels its main task, drives the loop until that task has ended and
re-raises: the cleanup of the test or fixture completes before pytest tears
anything down, and finds the request on its task like any other
cancellation. The interruption reaches that task alone, and the loop goes
on serving the other tests of its scope, which shows after a stopped loop,
a pytest-timeout signal or a cleanup failure; Ctrl-C ends the session
anyway.
"""

from __future__ import annotations

import sys
from textwrap import dedent

import pytest
from pytest import Pytester

_REFUSED = "*RuntimeError: This event loop no longer accepts new tests*"
_SIGNALS = pytest.mark.skipif(sys.platform == "win32", reason="SIGINT and SIGALRM")
_NEEDS_TASK_GROUP = pytest.mark.skipif(
    sys.version_info < (3, 11), reason="asyncio.TaskGroup needs Python 3.11"
)


def _run(pytester: Pytester, source: str, *args: str):
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent(source))
    return pytester.runpytest_subprocess(
        "--asyncio-mode=strict", "-s", *args, timeout=30
    )


def _assert_ordered(out: str, *events: str) -> None:
    positions = [out.index(event) for event in events]
    assert positions == sorted(positions), events


# The interruption arrives while the test awaits; what the test does about it
# varies.
_INTERRUPTED_TEST_SOURCE = """
    import asyncio
    import os
    import signal
    import pytest
    import pytest_asyncio

    def sigint():
        asyncio.get_running_loop().call_soon(os.kill, os.getpid(), signal.SIGINT)

    def raise_():
        raise KeyboardInterrupt

    @pytest.fixture(scope="module")
    def resource():
        state = {{"open": True}}
        yield state
        print("SYNC TEARDOWN")
        state["open"] = False

    @pytest_asyncio.fixture(loop_scope="module")
    async def async_resource(resource):
        yield resource
        await asyncio.sleep(0)
        print("ASYNC TEARDOWN")

    async def finish(resource):
        await asyncio.sleep(0.05)
        assert resource["open"]
        print("CLEANUP FINISHED")

    async def hang():
        sigint()
        try:
            await asyncio.Event().wait()
        finally:
            print("HUNG CLEANUP ENDED")
            {then}

    @pytest.mark.asyncio(loop_scope="module")
    async def test_interrupted(async_resource):
        try:
            {interrupt}
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            print("TEST CANCELLED")
            {on_cancel}
        finally:
            print("CLEANUP STARTED")
            {cleanup}

    @pytest.mark.asyncio(loop_scope="module")
    async def test_next():
        print("NEXT RAN")
"""


def _interrupted_test(
    interrupt: str = "sigint()",
    on_cancel: str = "raise",
    cleanup: str = "pass",
    then: str = "pass",
) -> str:
    return _INTERRUPTED_TEST_SOURCE.format(
        interrupt=interrupt, on_cancel=on_cancel, cleanup=cleanup, then=then
    )


@_SIGNALS
@pytest.mark.parametrize(
    "interrupt",
    [
        pytest.param("sigint()", id="sigint"),
        pytest.param(
            "asyncio.get_running_loop().call_soon(raise_)",
            id="keyboard_interrupt_from_callback",
        ),
        pytest.param("raise KeyboardInterrupt", id="keyboard_interrupt_from_test"),
    ],
)
def test_interrupted_test_cleans_up_before_fixture_teardown(
    pytester: Pytester, interrupt: str
):
    """
    The test's cleanup runs to completion, then the async fixture is torn
    down, then the sync fixture; the session ends by interruption.
    """
    source = _interrupted_test(interrupt, cleanup="await finish(async_resource)")
    result = _run(pytester, source)
    assert result.ret == pytest.ExitCode.INTERRUPTED
    out = result.stdout.str()
    _assert_ordered(
        out, "CLEANUP STARTED", "CLEANUP FINISHED", "ASYNC TEARDOWN", "SYNC TEARDOWN"
    )
    assert "NEXT RAN" not in out
    result.stdout.fnmatch_lines(["*KeyboardInterrupt*"])


@_SIGNALS
@pytest.mark.parametrize(
    "then",
    [
        pytest.param("pass", id="ends"),
        pytest.param("raise RuntimeError('abandoned cleanup failed')", id="fails"),
    ],
)
def test_second_interruption_abandons_hung_cleanup(pytester: Pytester, then: str):
    """
    The wait for the cleanup is interrupted too: the interruption reaches
    pytest at once and the abandoned task ends at its next await, the next
    time the loop runs (here, for the async fixture's teardown). Nothing is
    left pending or unawaited; an error the abandoned cleanup ends with goes
    to the loop's exception handler, which logs it (pytest shows such logs
    with live logging, as any asyncio error report during an interruption).
    """
    source = _interrupted_test(cleanup="await hang()", then=then)
    result = _run(pytester, source, "-W", "error", "-o", "log_cli=true")
    assert result.ret == pytest.ExitCode.INTERRUPTED
    out = result.stdout.str()
    _assert_ordered(
        out, "CLEANUP STARTED", "HUNG CLEANUP ENDED", "ASYNC TEARDOWN", "SYNC TEARDOWN"
    )
    assert "NEXT RAN" not in out
    result.stdout.fnmatch_lines(["*KeyboardInterrupt*"])
    err = result.stderr.str()
    for noise in ("Task was destroyed", "never awaited", "never retrieved"):
        assert noise not in out + err
    if then == "pass":
        assert "abandoned" not in out + err
    else:
        result.stdout.fnmatch_lines(
            [
                "*Exception from an async fixture or test after pytest stopped*",
                "*RuntimeError: abandoned cleanup failed",
            ]
        )


@_SIGNALS
def test_cleanup_failure_replaces_the_interruption(pytester: Pytester):
    """
    As for a synchronous test, the error the test ends with is its outcome:
    the test fails with it (the cancellation is its context) and the session
    goes on, the next test on the shared loop included.
    """
    source = _interrupted_test(cleanup="raise AssertionError('cleanup bug')")
    result = _run(pytester, source)
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    result.assert_outcomes(failed=1, passed=1)
    result.stdout.fnmatch_lines(
        [
            "*CLEANUP STARTED*",
            "*ASYNC TEARDOWN*",
            "*NEXT RAN*",
            "*SYNC TEARDOWN*",
            "*_ test_interrupted _*",
            "*asyncio.exceptions.CancelledError*",
            "*During handling of the above exception, another exception occurred:*",
            "*AssertionError: cleanup bug",
        ]
    )


@_SIGNALS
def test_suppressed_cancellation_does_not_suppress_the_interruption(
    pytester: Pytester,
):
    """Ctrl-C is not the test's to suppress: the session is still interrupted."""
    result = _run(pytester, _interrupted_test(on_cancel="pass"))
    assert result.ret == pytest.ExitCode.INTERRUPTED
    out = result.stdout.str()
    assert "TEST CANCELLED" in out
    assert "NEXT RAN" not in out
    result.stdout.fnmatch_lines(["*ASYNC TEARDOWN*", "*KeyboardInterrupt*"])


@_SIGNALS
@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="Task.cancelling() needs Python 3.11"
)
def test_interrupted_test_finds_the_request_on_its_task(pytester: Pytester):
    """
    The runner cancels the test's task, as asyncio.Runner does: a test
    awaiting a child task tells the interruption, a request on its own
    task, from a cancellation of the child alone, and does not carry on
    serving.
    """
    result = _run(
        pytester,
        """
        import asyncio
        import os
        import signal
        import pytest

        async def serve():
            await asyncio.Event().wait()

        @pytest.mark.asyncio
        async def test_cancel_aware_parent():
            loop = asyncio.get_running_loop()
            loop.call_soon(os.kill, os.getpid(), signal.SIGINT)
            child = asyncio.create_task(serve())
            try:
                try:
                    await child
                except asyncio.CancelledError:
                    count = asyncio.current_task().cancelling()
                    print("PARENT CANCELLATION REQUESTS:", count)
                    if count:
                        raise
                    print("CONTINUING AFTER CHILD CANCELLATION")
                await serve()
            finally:
                print("PARENT CLEANUP")
        """,
    )
    assert result.ret == pytest.ExitCode.INTERRUPTED
    out = result.stdout.str()
    assert "CONTINUING AFTER CHILD CANCELLATION" not in out
    result.stdout.fnmatch_lines(
        ["*PARENT CANCELLATION REQUESTS: 1*", "*PARENT CLEANUP*", "*KeyboardInterrupt*"]
    )


def test_interruption_after_the_test_ended_cancels_nothing(pytester: Pytester):
    """
    The loop is stopped by a callback the test scheduled before returning,
    after the test ended and before the runner's wait for it did: there is
    nothing to cancel or join, the test fails with asyncio's error and the
    next test on the shared loop runs.
    """
    result = _run(
        pytester,
        """
        import asyncio
        import pytest
        import pytest_asyncio

        @pytest_asyncio.fixture(scope="module", loop_scope="module")
        async def resource():
            yield
            await asyncio.sleep(0)
            print("RESOURCE TORN DOWN")

        @pytest.mark.asyncio(loop_scope="module")
        async def test_completed(resource):
            loop = asyncio.get_running_loop()
            loop.call_soon(loop.stop)
            print("TEST BODY COMPLETED")

        @pytest.mark.asyncio(loop_scope="module")
        async def test_next(resource):
            print("NEXT TEST RAN")
        """,
    )
    result.assert_outcomes(failed=1, passed=1)
    result.stdout.fnmatch_lines(
        [
            "*TEST BODY COMPLETED*",
            "*NEXT TEST RAN*",
            "*RESOURCE TORN DOWN*",
            "*_ test_completed _*",
            "*RuntimeError: Event loop stopped before Future completed*",
        ]
    )


# A task group spanning a module fixture's yield; while the runner joins the
# interrupted test, the test makes the group's child fail, which cancels the
# fixture's task at its yield and, through the runner, the test's once more.
_SERVICE_FAILS_DURING_JOIN_SOURCE = """
    import asyncio
    import signal
    import pytest
    import pytest_asyncio

    @pytest_asyncio.fixture(scope="module", loop_scope="module")
    async def service():
        trigger = asyncio.Event()

        async def serve():
            await trigger.wait()
            raise ValueError("fixture child failed")

        async with asyncio.TaskGroup() as group:
            group.create_task(serve())
            yield trigger

    def alarm():
        def fail(signum, frame):
            pytest.fail("simulated pytest-timeout signal")

        signal.signal(signal.SIGALRM, fail)
        signal.setitimer(signal.ITIMER_REAL, 0.05)

    async def trigger_failure(service):
        print("TEST TRIGGERS SERVICE FAILURE")
        service.set()
        await asyncio.Event().wait()

    async def suppress_and_trigger_failure(service):
        # The runner's request is the only one on the task.
        assert asyncio.current_task().uncancel() == 0
        await trigger_failure(service)

    @pytest.mark.asyncio(loop_scope="module")
    async def test_interrupted(service):
        {interrupt}
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            {on_cancel}
        finally:
            {cleanup}

    @pytest.mark.asyncio(loop_scope="module")
    async def test_later_consumer(service):
        print("LATER CONSUMER ADMITTED")
        pytest.fail("this consumer should have been refused")
"""

_STOP = "asyncio.get_running_loop().stop()"
_STOPPED = "RuntimeError: Event loop stopped before Future completed"


@_NEEDS_TASK_GROUP
@pytest.mark.parametrize(
    ("interrupt", "on_cancel", "cleanup", "error"),
    [
        pytest.param(
            "service.set()",
            "raise",
            "pass",
            "asyncio.exceptions.CancelledError",
            id="control",
        ),
        pytest.param(
            "alarm()",
            "raise",
            "await trigger_failure(service)",
            "Failed: simulated pytest-timeout signal",
            id="signal",
            marks=_SIGNALS,
        ),
        pytest.param(
            _STOP, "raise", "await trigger_failure(service)", _STOPPED, id="stop"
        ),
        pytest.param(
            _STOP,
            "await suppress_and_trigger_failure(service)",
            "pass",
            _STOPPED,
            id="resolved",
        ),
    ],
)
def test_fixture_cancellation_during_interrupted_cleanup_is_kept(
    pytester: Pytester, interrupt: str, on_cancel: str, cleanup: str, error: str
):
    """
    The group's request, made while the runner joined the interrupted test,
    ends the test's cleanup, even one that suppressed the runner's
    cancellation and resolved its request (resolved), and is reported at the
    fixture's teardown; the test fails with the interruption's error and
    the later consumer is refused, as when the group alone cancels the test
    (control).
    """
    source = _SERVICE_FAILS_DURING_JOIN_SOURCE.format(
        interrupt=interrupt, on_cancel=on_cancel, cleanup=cleanup
    )
    result = _run(pytester, source)
    out = result.stdout.str()
    assert ("TEST TRIGGERS SERVICE FAILURE" in out) is (interrupt != "service.set()")
    assert "LATER CONSUMER ADMITTED" not in out
    result.assert_outcomes(failed=2, errors=1)
    result.stdout.fnmatch_lines(
        [
            "*ERROR at teardown of test_later_consumer*",
            "*ValueError: fixture child failed*",
            "*_ test_interrupted _*",
            f"*{error}*",
            "*_ test_later_consumer _*",
            _REFUSED,
        ]
    )


@_SIGNALS
@pytest.mark.parametrize(
    "interrupt",
    ["loop.call_soon(os.kill, os.getpid(), signal.SIGINT)", "loop.call_soon(raise_)"],
    ids=["sigint", "keyboard_interrupt_from_callback"],
)
def test_interruption_during_async_fixture_teardown(pytester: Pytester, interrupt: str):
    """
    The teardown is cancelled at its await and joined before the
    interruption reaches pytest; the loop closes without leaving the
    generator pending.
    """
    result = _run(
        pytester,
        f"""
        import asyncio
        import os
        import signal
        import pytest
        import pytest_asyncio

        def raise_():
            raise KeyboardInterrupt

        @pytest_asyncio.fixture
        async def resource():
            yield
            loop = asyncio.get_running_loop()
            {interrupt}
            try:
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                print("TEARDOWN CANCELLED")
                raise
            print("TEARDOWN FINISHED")

        @pytest.mark.asyncio
        async def test_it(resource):
            pass
        """,
    )
    assert result.ret == pytest.ExitCode.INTERRUPTED
    out = result.stdout.str()
    assert "TEARDOWN FINISHED" not in out
    result.stdout.fnmatch_lines(["*TEARDOWN CANCELLED*", "*KeyboardInterrupt*"])
    assert "Task was destroyed" not in out + result.stderr.str()


def test_error_of_a_teardown_settled_just_before_its_abandonment_is_reported(
    pytester: Pytester,
):
    """
    The teardown, cancelled by a first interruption, ends with an error in
    the same loop iteration as a second interruption of the wait for it.
    The interruption ends the session; the error is reported through the
    loop's exception handler, once, like an error after the abandonment.
    """
    result = _run(
        pytester,
        """
        import asyncio
        import pytest
        import pytest_asyncio

        def raise_():
            raise KeyboardInterrupt

        @pytest_asyncio.fixture
        async def resource():
            loop = asyncio.get_running_loop()
            yield
            loop.call_soon(raise_)
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                loop.call_soon(raise_)
                print("TEARDOWN RAISES")
                raise ValueError("teardown failed as the second interruption arrived")

        @pytest.mark.asyncio
        async def test_it(resource):
            pass
        """,
        "-W",
        "error",
        "-o",
        "log_cli=true",
    )
    assert result.ret == pytest.ExitCode.INTERRUPTED
    out = result.stdout.str()
    err = result.stderr.str()
    assert out.count("after pytest stopped waiting for it") == 1
    result.stdout.fnmatch_lines(
        [
            "*TEARDOWN RAISES*",
            "*Exception from an async fixture or test after pytest stopped waiting*",
            "*ValueError: teardown failed as the second interruption arrived",
            "*KeyboardInterrupt*",
        ]
    )
    for noise in ("never retrieved", "Task was destroyed"):
        assert noise not in out + err


def test_keyboard_interrupt_from_node_finalizer(pytester: Pytester):
    """
    The item's remaining finalizers are lost to pytest; the runner closes
    the fixture when the loop closes, in the fixture's own task, as
    asyncio's shutdown closes async generators.
    """
    result = _run(
        pytester,
        """
        import asyncio
        import pytest
        import pytest_asyncio

        @pytest_asyncio.fixture
        async def resource():
            try:
                yield
            finally:
                print("RESOURCE TORN DOWN IN", asyncio.current_task().get_name())

        @pytest.mark.asyncio
        async def test_it(resource, request):
            def interrupt():
                raise KeyboardInterrupt

            request.node.addfinalizer(interrupt)
        """,
        "-W",
        "error",
    )
    assert result.ret == pytest.ExitCode.INTERRUPTED
    result.stdout.fnmatch_lines(["*RESOURCE TORN DOWN IN resource"])
    assert "Task was destroyed" not in result.stdout.str() + result.stderr.str()


@_SIGNALS
@pytest.mark.parametrize(
    "hang",
    [
        pytest.param("await asyncio.Event().wait()", id="await"),
        pytest.param(
            "while True: pass",
            id="busy",
            marks=pytest.mark.xfail(
                (3, 11) <= sys.version_info < (3, 13),
                reason="CPython 3.11 and 3.12 raise an exception from a signal "
                "handler past the try/finally of the frame it interrupts, so the "
                "test's cleanup is skipped",
                strict=True,
            ),
        ),
    ],
)
def test_pytest_timeout_fails_the_hung_test(pytester: Pytester, hang: str):
    """
    The test fails once its cleanup has run, and the shared loop goes on. A
    signal that interrupts the wait for the test (await) has the runner
    cancel and join the test's task; one that interrupts the test itself
    (busy) fails it from its own frames, as pytest.fail() would.
    """
    pytest.importorskip("pytest_timeout")
    result = _run(
        pytester,
        f"""
        import asyncio
        import pytest
        import pytest_asyncio

        @pytest_asyncio.fixture(loop_scope="module")
        async def resource():
            yield
            await asyncio.sleep(0)
            print("RESOURCE TORN DOWN")

        @pytest.mark.timeout(0.5)
        @pytest.mark.asyncio(loop_scope="module")
        async def test_hang(resource):
            try:
                {hang}
            finally:
                await asyncio.sleep(0)
                print("HANG CLEANED UP")

        @pytest.mark.asyncio(loop_scope="module")
        async def test_next():
            print("NEXT RAN")
        """,
        "-o",
        "timeout_method=signal",
    )
    result.assert_outcomes(failed=1, passed=1)
    result.stdout.fnmatch_lines(
        [
            "*HANG CLEANED UP*",
            "*RESOURCE TORN DOWN*",
            "*NEXT RAN*",
            "*_ test_hang _*",
            "*Failed: Timeout (>0.5s) from pytest-timeout*",
        ]
    )
