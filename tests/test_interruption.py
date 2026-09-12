"""
Interrupting the task that runs pytest-asyncio's fixtures and tests.

``Runner.run`` raises on the first SIGINT, when a KeyboardInterrupt or a
pytest-timeout failure escapes the event loop, or when the loop is stopped.
The runner then cancels the task, as asyncio.Runner cancels its main task,
drives the loop until the interrupted job has ended and re-raises: the
job's cleanup completes before pytest tears anything down, and finds the
request on its task like any other cancellation. The runner cannot tell
its own request from a scope's, so an interruption ends the normal work of
the loop: a shared loop runs nothing but fixture teardowns until it closes,
on every Python version. That shows after a pytest-timeout signal, a
stopped loop or a cleanup failure, which the session survives; Ctrl-C ends
the session anyway.
"""

from __future__ import annotations

import sys
from textwrap import dedent

import pytest
from pytest import Pytester, RunResult

_REFUSED = "*CancelledError: The coroutine was refused: *until the loop is closed."
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


# SIGINT arrives while the test awaits; what the test does about it varies.
_INTERRUPTED_TEST_SOURCE = """
    import asyncio
    import os
    import signal
    import pytest
    import pytest_asyncio

    def sigint():
        asyncio.get_running_loop().call_soon(os.kill, os.getpid(), signal.SIGINT)

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
    interrupt: str = "sigint()", on_cancel: str = "raise", cleanup: str = "pass"
) -> str:
    return _INTERRUPTED_TEST_SOURCE.format(
        interrupt=interrupt, on_cancel=on_cancel, cleanup=cleanup
    )


@_SIGNALS
@pytest.mark.parametrize("interrupt", ["sigint()", "raise KeyboardInterrupt"])
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
def test_second_interruption_abandons_hung_cleanup(pytester: Pytester):
    """
    The wait for the cleanup is interrupted too: the interruption reaches
    pytest at once and the abandoned job ends at its next await, the next
    time the loop runs (here, for the async fixture's teardown). Nothing is
    left pending, unawaited or unretrieved.
    """
    result = _run(pytester, _interrupted_test(cleanup="await hang()"), "-W", "error")
    assert result.ret == pytest.ExitCode.INTERRUPTED
    out = result.stdout.str()
    _assert_ordered(
        out, "CLEANUP STARTED", "HUNG CLEANUP ENDED", "ASYNC TEARDOWN", "SYNC TEARDOWN"
    )
    assert "NEXT RAN" not in out
    result.stdout.fnmatch_lines(["*KeyboardInterrupt*"])
    for noise in ("Task was destroyed", "never awaited", "never retrieved"):
        assert noise not in out + result.stderr.str()


@_SIGNALS
def test_cleanup_failure_replaces_the_interruption(pytester: Pytester):
    """
    As for a synchronous test, the error the test ends with is its outcome:
    the test fails with it (the cancellation is its context) and the session
    goes on. The shared loop is teardown-only from the interruption on,
    though: the next test on it is refused.
    """
    source = _interrupted_test(cleanup="raise AssertionError('cleanup bug')")
    result = _run(pytester, source)
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    result.assert_outcomes(failed=2)
    assert "NEXT RAN" not in result.stdout.str()
    result.stdout.fnmatch_lines(
        [
            "*CLEANUP STARTED*",
            "*ASYNC TEARDOWN*",
            "*SYNC TEARDOWN*",
            "*_ test_interrupted _*",
            "*asyncio.exceptions.CancelledError*",
            "*During handling of the above exception, another exception occurred:*",
            "*AssertionError: cleanup bug",
            "*_ test_next _*",
            _REFUSED,
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
    The runner cancels the task, as asyncio.Runner does: a test awaiting a
    child task tells the interruption, a request on its own task, from a
    cancellation of the child alone, and does not carry on serving.
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
# task it is running in.
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
        {consumer}
"""


def _service_fails_during_join(
    interrupt: str,
    on_cancel: str = "raise",
    cleanup: str = "pass",
    consumer: str = 'pytest.fail("this consumer should have been refused")',
) -> str:
    return _SERVICE_FAILS_DURING_JOIN_SOURCE.format(
        interrupt=interrupt, on_cancel=on_cancel, cleanup=cleanup, consumer=consumer
    )


def _assert_fixture_cancellation_kept(result: RunResult, error: str) -> None:
    """
    The interrupted test fails with the interruption's error, the later
    consumer of the fixture is refused and the fixture's teardown reports the
    child's error.
    """
    result.assert_outcomes(failed=2, errors=1)
    assert "LATER CONSUMER ADMITTED" not in result.stdout.str()
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


@_NEEDS_TASK_GROUP
@pytest.mark.parametrize(
    ("interrupt", "cleanup", "error"),
    [
        pytest.param(
            "service.set()", "pass", "asyncio.exceptions.CancelledError", id="control"
        ),
        pytest.param(
            "alarm()",
            "await trigger_failure(service)",
            "Failed: simulated pytest-timeout signal",
            id="signal",
            marks=_SIGNALS,
        ),
        pytest.param(
            "asyncio.get_running_loop().stop()",
            "await trigger_failure(service)",
            "RuntimeError: Event loop stopped before Future completed",
            id="stop",
        ),
    ],
)
def test_fixture_cancellation_during_interrupted_cleanup_is_kept(
    pytester: Pytester, interrupt: str, cleanup: str, error: str
):
    """
    The group's request, made while the runner joined the interrupted test,
    ends the test's cleanup and is reported at the fixture's teardown; the
    test fails with the interruption's error and the consumer is refused,
    as when the group alone cancels the test (control).
    """
    result = _run(pytester, _service_fails_during_join(interrupt, cleanup=cleanup))
    triggered = "TEST TRIGGERS SERVICE FAILURE" in result.stdout.str()
    assert triggered is (cleanup != "pass")
    _assert_fixture_cancellation_kept(result, error)


@_NEEDS_TASK_GROUP
def test_resolved_interruption_keeps_the_loop_teardown_only(pytester: Pytester):
    """
    A test that suppresses the runner's cancellation and resolves its
    request, the only one on the task, is not admitted later work either;
    the group's request that follows reaches it and the group's error is
    reported at teardown. The later consumer is refused; admitted, it would
    hang on the failed service.
    """
    source = _service_fails_during_join(
        "asyncio.get_running_loop().stop()",
        on_cancel="await suppress_and_trigger_failure(service)",
        consumer="await asyncio.Event().wait()",
    )
    result = _run(pytester, source)
    result.stdout.fnmatch_lines(["*TEST TRIGGERS SERVICE FAILURE*"])
    _assert_fixture_cancellation_kept(
        result, "RuntimeError: Event loop stopped before Future completed"
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


def test_keyboard_interrupt_from_node_finalizer(pytester: Pytester):
    """
    The item's remaining finalizers are lost to pytest, which finishes the
    fixture at the end of the session, in the task that set it up.
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
    result.stdout.fnmatch_lines(["*RESOURCE TORN DOWN IN pytest-asyncio*"])
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
    The test fails once its cleanup has run. A signal that interrupts the
    wait for the test (await) has the runner cancel the task, after which
    the shared loop runs nothing but fixture teardowns; one that interrupts
    the test itself (busy) fails it from its own frames, as pytest.fail()
    would, and the loop carries on.
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
    interrupted = hang.startswith("await")
    result.assert_outcomes(failed=1 + interrupted, passed=1 - interrupted)
    assert ("NEXT RAN" in result.stdout.str()) is not interrupted
    result.stdout.fnmatch_lines(
        [
            "*HANG CLEANED UP*",
            "*RESOURCE TORN DOWN*",
            "*_ test_hang _*",
            "*Failed: Timeout (>0.5s) from pytest-timeout*",
        ]
    )
    if interrupted:
        result.stdout.fnmatch_lines(["*_ test_next _*", _REFUSED])
