"""
Interrupting the wait for a test or a fixture.

Ctrl-C, a callback raising KeyboardInterrupt, a timeout plugin's failure or a
stopped event loop interrupts pytest-asyncio's wait for the running test or
fixture. pytest-asyncio cancels that task and waits for it to end, so that
its cleanup completes before pytest tears down the fixtures it uses; what
the cleanup raises is reported instead of the interruption. A second
interruption stops waiting.
"""

from __future__ import annotations

import sys
from textwrap import dedent

import pytest
from pytest import Pytester

_POSIX = pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX signals")
_REQUIRES_311 = pytest.mark.skipif(sys.version_info < (3, 11), reason="needs 3.11")
_REFUSED = "*RuntimeError: This event loop no longer accepts new tests*"
_REPORTED_LATE = "Exception from an async fixture or test after pytest stopped waiting"

# The fixtures and tests of an example append to ``events``; the conftest
# writes them to a file once every fixture is torn down, so that the outer
# test reads what ran, and in which order, instead of matching output that a
# traceback may repeat.
_EVENTS_CONFTEST = """\
    from pathlib import Path

    import pytest

    events = []

    @pytest.hookimpl(wrapper=True)
    def pytest_sessionfinish(session):
        yield
        Path(session.config.rootpath, "events.txt").write_text("\\n".join(events))
    """


def _events(pytester: Pytester) -> list[str]:
    return (pytester.path / "events.txt").read_text().splitlines()


@pytest.mark.parametrize(
    "interrupt",
    [
        pytest.param("raise KeyboardInterrupt", id="raised_by_the_test"),
        pytest.param("loop.call_soon(interrupt)", id="raised_by_a_callback"),
        pytest.param(
            "loop.call_soon(os.kill, os.getpid(), signal.SIGINT)",
            id="sigint",
            marks=_POSIX,
        ),
    ],
)
def test_an_interrupted_test_finishes_its_cleanup_before_its_fixtures_are_torn_down(
    pytester: Pytester, interrupt: str
):
    """The cleanup of the interrupted test still finds its fixtures open."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makeconftest(dedent(_EVENTS_CONFTEST))
    pytester.makepyfile(dedent(f"""\
        import asyncio
        import os
        import signal

        import pytest
        import pytest_asyncio
        from conftest import events

        def interrupt():
            raise KeyboardInterrupt

        @pytest.fixture
        def connection():
            state = {{"open": True}}
            yield state
            state["open"] = False
            events.append("connection closed")

        @pytest_asyncio.fixture
        async def transaction(connection):
            yield connection
            await asyncio.sleep(0)
            assert connection["open"]
            events.append("transaction closed")

        @pytest.mark.asyncio
        async def test_query(transaction):
            loop = asyncio.get_running_loop()
            try:
                {interrupt}
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0)
                assert transaction["open"]
                events.append("query cleaned up")

        @pytest.mark.asyncio
        async def test_next():
            events.append("next ran")
        """))
    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)
    assert result.ret == pytest.ExitCode.INTERRUPTED
    assert _events(pytester) == [
        "query cleaned up",
        "transaction closed",
        "connection closed",
    ]


def test_a_second_interruption_stops_waiting_for_the_cleanup(pytester: Pytester):
    """The abandoned cleanup ends on its own, before the fixtures are torn down."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makeconftest(dedent(_EVENTS_CONFTEST))
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest
        import pytest_asyncio
        from conftest import events

        def interrupt():
            raise KeyboardInterrupt

        @pytest_asyncio.fixture
        async def connection():
            yield
            await asyncio.sleep(0)
            events.append("connection closed")

        @pytest.mark.asyncio
        async def test_query(connection):
            loop = asyncio.get_running_loop()
            loop.call_soon(interrupt)
            try:
                await asyncio.Event().wait()
            finally:
                loop.call_soon(interrupt)
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    events.append("cleanup ended")
                    raise
        """))
    result = pytester.runpytest_subprocess(
        "--asyncio-mode=strict", "-W", "error", timeout=30
    )
    assert result.ret == pytest.ExitCode.INTERRUPTED
    assert _events(pytester) == ["cleanup ended", "connection closed"]
    for noise in ("Task was destroyed", "never awaited", "never retrieved"):
        assert noise not in result.stdout.str() + result.stderr.str()


def test_an_error_raised_after_the_second_interruption_is_reported_once(
    pytester: Pytester,
):
    """The abandoned cleanup's error goes to the loop's exception handler."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest
        import pytest_asyncio

        def interrupt():
            raise KeyboardInterrupt

        @pytest_asyncio.fixture
        async def connection():
            yield
            await asyncio.sleep(0)

        @pytest.mark.asyncio
        async def test_query(connection):
            loop = asyncio.get_running_loop()
            loop.call_soon(interrupt)
            try:
                await asyncio.Event().wait()
            finally:
                loop.call_soon(interrupt)
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    raise RuntimeError("abandoned cleanup failed") from None
        """))
    result = pytester.runpytest_subprocess(
        "--asyncio-mode=strict", "-W", "error", "-o", "log_cli=true", timeout=30
    )
    assert result.ret == pytest.ExitCode.INTERRUPTED
    assert result.stdout.str().count(_REPORTED_LATE) == 1
    result.stdout.fnmatch_lines(["*RuntimeError: abandoned cleanup failed*"])


def test_a_cleanup_error_is_reported_instead_of_the_interruption(pytester: Pytester):
    """The test fails with its cleanup's error, and the session goes on."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest

        def interrupt():
            raise KeyboardInterrupt

        @pytest.mark.asyncio(loop_scope="module")
        async def test_query():
            asyncio.get_running_loop().call_soon(interrupt)
            try:
                await asyncio.Event().wait()
            finally:
                raise AssertionError("cleanup bug")

        @pytest.mark.asyncio(loop_scope="module")
        async def test_next():
            pass
        """))
    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    result.assert_outcomes(failed=1, passed=1)
    result.stdout.fnmatch_lines(["*_ test_query _*", "*AssertionError: cleanup bug*"])


def test_a_cleanup_error_that_is_false_is_still_reported(pytester: Pytester):
    """An exception is chosen over the interruption whatever its truth value."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest
        import pytest_asyncio

        class FalseError(Exception):
            def __bool__(self):
                return False

        def interrupt():
            raise KeyboardInterrupt

        async def fail_during_cleanup():
            asyncio.get_running_loop().call_soon(interrupt)
            try:
                await asyncio.Event().wait()
            finally:
                raise FalseError("cleanup error")

        @pytest_asyncio.fixture(loop_scope="module")
        async def failing_setup():
            await fail_during_cleanup()
            yield

        @pytest_asyncio.fixture(loop_scope="module")
        async def failing_teardown():
            yield
            await fail_during_cleanup()

        @pytest.mark.asyncio(loop_scope="module")
        async def test_setup(failing_setup):
            pass

        @pytest.mark.asyncio(loop_scope="module")
        async def test_teardown(failing_teardown):
            pass

        @pytest.mark.asyncio(loop_scope="module")
        async def test_body():
            await fail_during_cleanup()
        """))
    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    result.assert_outcomes(passed=1, failed=1, errors=2)
    result.stdout.fnmatch_lines(
        [
            "*ERROR at setup of test_setup*",
            "*FalseError: cleanup error*",
            "*ERROR at teardown of test_teardown*",
            "*FalseError: cleanup error*",
            "*_ test_body _*",
            "*FalseError: cleanup error*",
        ]
    )


def test_a_test_suppressing_the_cancellation_does_not_suppress_the_interruption(
    pytester: Pytester,
):
    """The test runs to its end, and the session is interrupted all the same."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makeconftest(dedent(_EVENTS_CONFTEST))
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest
        from conftest import events

        def interrupt():
            raise KeyboardInterrupt

        @pytest.mark.asyncio
        async def test_query():
            asyncio.get_running_loop().call_soon(interrupt)
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                events.append("cancellation suppressed")
            events.append("test returned")

        @pytest.mark.asyncio
        async def test_next():
            events.append("next ran")
        """))
    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)
    assert result.ret == pytest.ExitCode.INTERRUPTED
    assert _events(pytester) == ["cancellation suppressed", "test returned"]


@_REQUIRES_311
def test_an_interruption_is_a_cancellation_request_on_the_test_task(
    pytester: Pytester,
):
    """A test awaiting a child task can tell the interruption from the child's."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makeconftest(dedent(_EVENTS_CONFTEST))
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest
        from conftest import events

        def interrupt():
            raise KeyboardInterrupt

        @pytest.mark.asyncio
        async def test_parent():
            asyncio.get_running_loop().call_soon(interrupt)
            child = asyncio.create_task(asyncio.Event().wait())
            try:
                await child
            except asyncio.CancelledError:
                if asyncio.current_task().cancelling():
                    events.append("the test task was asked to cancel")
                    raise
                events.append("only the child was cancelled")
            finally:
                events.append("parent cleaned up")
        """))
    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)
    assert result.ret == pytest.ExitCode.INTERRUPTED
    assert _events(pytester) == [
        "the test task was asked to cancel",
        "parent cleaned up",
    ]


def test_a_loop_stopped_after_the_test_returned_does_not_refuse_later_tests(
    pytester: Pytester,
):
    """There is nothing to cancel: the test fails, and the shared loop goes on."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makeconftest(dedent(_EVENTS_CONFTEST))
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest
        import pytest_asyncio
        from conftest import events

        @pytest_asyncio.fixture(scope="module", loop_scope="module")
        async def resource():
            yield
            await asyncio.sleep(0)
            events.append("resource closed")

        @pytest.mark.asyncio(loop_scope="module")
        async def test_stop(resource):
            loop = asyncio.get_running_loop()
            loop.call_soon(loop.stop)
            events.append("test returned")

        @pytest.mark.asyncio(loop_scope="module")
        async def test_next(resource):
            events.append("next ran")
        """))
    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)
    result.assert_outcomes(failed=1, passed=1)
    result.stdout.fnmatch_lines(
        [
            "*_ test_stop _*",
            "*RuntimeError: Event loop stopped before Future completed*",
        ]
    )
    assert _events(pytester) == ["test returned", "next ran", "resource closed"]


@_REQUIRES_311
def test_a_fixture_failing_while_the_interrupted_test_cleans_up_ends_the_cleanup(
    pytester: Pytester,
):
    """
    The cleanup is cancelled once more, even after it resolved the
    interruption's request, and the loop refuses the later tests.
    """
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makeconftest(dedent(_EVENTS_CONFTEST))
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest
        import pytest_asyncio
        from conftest import events

        @pytest_asyncio.fixture(scope="module", loop_scope="module")
        async def service():
            trigger = asyncio.Event()

            async def serve():
                await trigger.wait()
                raise ValueError("service failed")

            async with asyncio.TaskGroup() as group:
                group.create_task(serve())
                yield trigger

        @pytest.mark.asyncio(loop_scope="module")
        async def test_interrupted(service):
            asyncio.get_running_loop().stop()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                asyncio.current_task().uncancel()
                service.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    events.append("cleanup ended by the service failure")
                    raise

        @pytest.mark.asyncio(loop_scope="module")
        async def test_later(service):
            events.append("later test ran")
        """))
    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)
    result.assert_outcomes(failed=2, errors=1)
    result.stdout.fnmatch_lines(
        [
            "*ERROR at teardown of test_later*",
            "*ValueError: service failed*",
            "*_ test_interrupted _*",
            "*RuntimeError: Event loop stopped before Future completed*",
            "*_ test_later _*",
            _REFUSED,
        ]
    )
    assert _events(pytester) == ["cleanup ended by the service failure"]


@pytest.mark.parametrize(
    "interrupt",
    [
        pytest.param("loop.call_soon(interrupt)", id="raised_by_a_callback"),
        pytest.param(
            "loop.call_soon(os.kill, os.getpid(), signal.SIGINT)",
            id="sigint",
            marks=_POSIX,
        ),
    ],
)
def test_an_interrupted_fixture_teardown_is_cancelled_and_finished(
    pytester: Pytester, interrupt: str
):
    """The teardown ends at its await, and the loop closes with nothing pending."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makeconftest(dedent(_EVENTS_CONFTEST))
    pytester.makepyfile(dedent(f"""\
        import asyncio
        import os
        import signal

        import pytest
        import pytest_asyncio
        from conftest import events

        def interrupt():
            raise KeyboardInterrupt

        @pytest_asyncio.fixture
        async def resource():
            yield
            loop = asyncio.get_running_loop()
            {interrupt}
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                events.append("teardown cancelled")
                raise
            events.append("teardown finished")

        @pytest.mark.asyncio
        async def test_it(resource):
            pass
        """))
    result = pytester.runpytest_subprocess(
        "--asyncio-mode=strict", "-W", "error", timeout=30
    )
    assert result.ret == pytest.ExitCode.INTERRUPTED
    assert _events(pytester) == ["teardown cancelled"]
    assert "Task was destroyed" not in result.stdout.str() + result.stderr.str()


def test_a_teardown_error_settled_as_the_second_interruption_arrives_is_reported_once(
    pytester: Pytester,
):
    """The interruption ends the session; the error is reported, not lost."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest
        import pytest_asyncio

        def interrupt():
            raise KeyboardInterrupt

        @pytest_asyncio.fixture
        async def resource():
            loop = asyncio.get_running_loop()
            yield
            loop.call_soon(interrupt)
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                loop.call_soon(interrupt)
                raise ValueError("teardown failed as the second interruption arrived")

        @pytest.mark.asyncio
        async def test_it(resource):
            pass
        """))
    result = pytester.runpytest_subprocess(
        "--asyncio-mode=strict", "-W", "error", "-o", "log_cli=true", timeout=30
    )
    assert result.ret == pytest.ExitCode.INTERRUPTED
    assert result.stdout.str().count(_REPORTED_LATE) == 1
    result.stdout.fnmatch_lines(
        ["*ValueError: teardown failed as the second interruption arrived*"]
    )
    for noise in ("never retrieved", "Task was destroyed"):
        assert noise not in result.stdout.str() + result.stderr.str()


def test_a_fixture_pytest_could_not_finalize_is_closed_in_its_own_task(
    pytester: Pytester,
):
    """An interrupted finalizer leaves the fixture to the closing of the loop."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makeconftest(dedent(_EVENTS_CONFTEST))
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest
        import pytest_asyncio
        from conftest import events

        @pytest_asyncio.fixture
        async def resource():
            task = asyncio.current_task()
            try:
                yield
            finally:
                await asyncio.sleep(0)
                if asyncio.current_task() is task:
                    events.append("resource closed in its own task")

        @pytest.mark.asyncio
        async def test_it(resource, request):
            def interrupt():
                raise KeyboardInterrupt

            request.node.addfinalizer(interrupt)
        """))
    result = pytester.runpytest_subprocess(
        "--asyncio-mode=strict", "-W", "error", timeout=30
    )
    assert result.ret == pytest.ExitCode.INTERRUPTED
    assert _events(pytester) == ["resource closed in its own task"]
    assert "Task was destroyed" not in result.stdout.str() + result.stderr.str()


@_POSIX
def test_pytest_timeout_fails_a_test_hung_at_an_await(pytester: Pytester):
    """The test's cleanup runs, its fixture is torn down and the loop goes on."""
    pytest.importorskip("pytest_timeout")
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makeconftest(dedent(_EVENTS_CONFTEST))
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest
        import pytest_asyncio
        from conftest import events

        @pytest_asyncio.fixture(loop_scope="module")
        async def resource():
            yield
            await asyncio.sleep(0)
            events.append("resource closed")

        @pytest.mark.timeout(0.5)
        @pytest.mark.asyncio(loop_scope="module")
        async def test_hang(resource):
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0)
                events.append("hang cleaned up")

        @pytest.mark.asyncio(loop_scope="module")
        async def test_next():
            events.append("next ran")
        """))
    result = pytester.runpytest_subprocess(
        "--asyncio-mode=strict", "-o", "timeout_method=signal", timeout=30
    )
    result.assert_outcomes(failed=1, passed=1)
    result.stdout.fnmatch_lines(
        ["*_ test_hang _*", "*Failed: Timeout (>0.5s) from pytest-timeout*"]
    )
    assert _events(pytester) == ["hang cleaned up", "resource closed", "next ran"]


@_POSIX
def test_pytest_timeout_fails_a_test_hung_in_a_busy_loop(pytester: Pytester):
    """The signal fails the test from its own frame; the loop serves the next test."""
    pytest.importorskip("pytest_timeout")
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makeconftest(dedent(_EVENTS_CONFTEST))
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest
        import pytest_asyncio
        from conftest import events

        @pytest_asyncio.fixture(loop_scope="module")
        async def resource():
            yield
            await asyncio.sleep(0)
            events.append("resource closed")

        @pytest.mark.timeout(0.5)
        @pytest.mark.asyncio(loop_scope="module")
        async def test_hang(resource):
            while True:
                pass

        @pytest.mark.asyncio(loop_scope="module")
        async def test_next():
            events.append("next ran")
        """))
    result = pytester.runpytest_subprocess(
        "--asyncio-mode=strict", "-o", "timeout_method=signal", timeout=30
    )
    result.assert_outcomes(failed=1, passed=1)
    result.stdout.fnmatch_lines(
        ["*_ test_hang _*", "*Failed: Timeout (>0.5s) from pytest-timeout*"]
    )
    assert _events(pytester) == ["resource closed", "next ran"]


def test_a_fixture_interrupted_after_it_yielded_is_torn_down_before_its_parent(
    pytester: Pytester,
):
    """Not received by pytest, the fixture is torn down by pytest-asyncio."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makeconftest(dedent(_EVENTS_CONFTEST))
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest
        import pytest_asyncio
        from conftest import events

        def interrupt():
            raise KeyboardInterrupt

        @pytest.fixture
        def parent():
            state = {"open": True}
            yield state
            state["open"] = False
            events.append("parent closed")

        @pytest_asyncio.fixture
        async def child(parent):
            asyncio.get_running_loop().call_soon(interrupt)
            yield parent
            await asyncio.sleep(0)
            assert parent["open"]
            events.append("child closed")

        @pytest.mark.asyncio
        async def test_query(child):
            events.append("test ran")
        """))
    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)
    assert result.ret == pytest.ExitCode.INTERRUPTED
    assert _events(pytester) == ["child closed", "parent closed"]


@_REQUIRES_311
def test_a_fixture_setup_that_recovers_from_the_interruption_and_yields_is_torn_down(
    pytester: Pytester,
):
    """The fixture is torn down at once, its parent still open."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makeconftest(dedent(_EVENTS_CONFTEST))
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest
        import pytest_asyncio
        from conftest import events

        def interrupt():
            raise KeyboardInterrupt

        @pytest.fixture
        def parent():
            state = {"open": True}
            yield state
            state["open"] = False
            events.append("parent closed")

        @pytest_asyncio.fixture
        async def child(parent):
            asyncio.get_running_loop().call_soon(interrupt)
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                asyncio.current_task().uncancel()
            yield parent
            await asyncio.sleep(0)
            assert parent["open"]
            events.append("child closed")

        @pytest.mark.asyncio
        async def test_query(child):
            events.append("test ran")
        """))
    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)
    assert result.ret == pytest.ExitCode.INTERRUPTED
    assert _events(pytester) == ["child closed", "parent closed"]


def test_a_teardown_error_of_a_fixture_pytest_never_received_is_reported(
    pytester: Pytester,
):
    """The error is the fixture's setup error, and the session goes on."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makeconftest(dedent(_EVENTS_CONFTEST))
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest
        import pytest_asyncio
        from conftest import events

        def interrupt():
            raise KeyboardInterrupt

        @pytest.fixture
        def parent():
            yield
            events.append("parent closed")

        @pytest_asyncio.fixture
        async def child(parent):
            asyncio.get_running_loop().call_soon(interrupt)
            yield parent
            await asyncio.sleep(0)
            events.append("child closed")
            raise ValueError("child cleanup failed")

        @pytest.mark.asyncio
        async def test_query(child):
            events.append("test ran")

        @pytest.mark.asyncio
        async def test_next():
            events.append("next ran")
        """))
    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    result.assert_outcomes(errors=1, passed=1)
    result.stdout.fnmatch_lines(
        ["*ERROR at setup of test_query*", "*ValueError: child cleanup failed*"]
    )
    assert _events(pytester) == ["child closed", "parent closed", "next ran"]


def test_a_fixture_setup_abandoned_by_a_second_interruption_reports_no_error(
    pytester: Pytester,
):
    """A setup that ends cancelled has no teardown to report."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest
        import pytest_asyncio

        def interrupt():
            raise KeyboardInterrupt

        @pytest_asyncio.fixture
        async def resource():
            loop = asyncio.get_running_loop()
            loop.call_soon(interrupt)
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                loop.call_soon(interrupt)
                await asyncio.Event().wait()
            yield

        @pytest.mark.asyncio
        async def test_it(resource):
            pass
        """))
    result = pytester.runpytest_subprocess(
        "--asyncio-mode=strict", "-W", "error", "-o", "log_cli=true", timeout=30
    )
    assert result.ret == pytest.ExitCode.INTERRUPTED
    output = result.stdout.str() + result.stderr.str()
    for noise in (_REPORTED_LATE, "Task was destroyed", "never retrieved"):
        assert noise not in output


def test_a_setup_returning_without_a_yield_after_the_interruption_keeps_it(
    pytester: Pytester,
):
    """A cancelled setup that cleans up and returns is not a missing yield."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makeconftest(dedent(_EVENTS_CONFTEST))
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest
        import pytest_asyncio
        from conftest import events

        def interrupt():
            raise KeyboardInterrupt

        @pytest_asyncio.fixture
        async def resource():
            asyncio.get_running_loop().call_soon(interrupt)
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                events.append("setup cleaned up")
                return
            yield

        @pytest.mark.asyncio
        async def test_it(resource):
            events.append("test ran")
        """))
    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)
    assert result.ret == pytest.ExitCode.INTERRUPTED
    assert "StopAsyncIteration" not in result.stdout.str()
    assert _events(pytester) == ["setup cleaned up"]


def test_a_setup_returning_without_a_yield_as_the_interruption_arrives_keeps_it(
    pytester: Pytester,
):
    """The interruption is reported, not the missing yield."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest
        import pytest_asyncio

        def interrupt():
            raise KeyboardInterrupt

        @pytest_asyncio.fixture
        async def resource():
            asyncio.get_running_loop().call_soon(interrupt)
            return
            yield

        @pytest.mark.asyncio
        async def test_it(resource):
            pass
        """))
    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)
    assert result.ret == pytest.ExitCode.INTERRUPTED
    assert "StopAsyncIteration" not in result.stdout.str()
