"""Cleanup and error reporting when synchronous pytest calls are interrupted."""

from __future__ import annotations

import sys
from textwrap import dedent

import pytest
from pytest import Pytester

_POSIX = pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX signals")
_REQUIRES_311 = pytest.mark.skipif(sys.version_info < (3, 11), reason="needs 3.11")
_REFUSED = "*RuntimeError: This event loop no longer accepts new tests*"
_REPORTED_LATE = "Exception from an async fixture or test after pytest stopped waiting"


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
def test_interrupted_test_finishes_cleanup_before_fixture_teardown(
    pytester: Pytester, interrupt: str
):
    """The test fills its buffer before async cleanup saves it to the open file."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent(f"""\
        import asyncio
        from io import StringIO
        import os
        import signal

        import pytest
        import pytest_asyncio

        def interrupt():
            raise KeyboardInterrupt

        @pytest.fixture
        def output_file():
            with open("output.txt", "w") as stream:
                yield stream

        @pytest_asyncio.fixture
        async def buffer(output_file):
            with StringIO() as stream:
                yield stream
                await asyncio.sleep(0)
                output_file.write(stream.getvalue())

        @pytest.mark.asyncio
        async def test_cleanup(buffer):
            loop = asyncio.get_running_loop()
            try:
                {interrupt}
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0)
                buffer.write("test output")

        @pytest.mark.asyncio
        async def test_next():
            pytest.fail("the session should have stopped")
        """))
    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)
    assert result.ret == pytest.ExitCode.INTERRUPTED
    result.assert_outcomes()
    assert (pytester.path / "output.txt").read_text() == "test output"


def test_a_second_interruption_stops_waiting_for_the_cleanup(pytester: Pytester):
    """Cancellation of the abandoned cleanup still lets it save its partial result."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest
        import pytest_asyncio

        def interrupt():
            raise KeyboardInterrupt

        @pytest_asyncio.fixture
        async def output():
            with open("report.txt", "w") as stream:
                yield stream
                await asyncio.sleep(0)

        @pytest.mark.asyncio
        async def test_query(output):
            loop = asyncio.get_running_loop()
            loop.call_soon(interrupt)
            try:
                await asyncio.Event().wait()
            finally:
                loop.call_soon(interrupt)
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    output.write("partial result")
                    raise
        """))
    result = pytester.runpytest_subprocess(
        "--asyncio-mode=strict", "-W", "error", timeout=30
    )
    assert result.ret == pytest.ExitCode.INTERRUPTED
    assert (pytester.path / "report.txt").read_text() == "partial result"
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


def test_cleanup_errors_are_reported_even_when_they_evaluate_to_false(
    pytester: Pytester,
):
    """A cleanup error takes precedence even when bool(exception) is False."""
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
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest

        def interrupt():
            raise KeyboardInterrupt

        @pytest.mark.asyncio
        async def test_query(request):
            task = asyncio.current_task()

            def check_finished_normally():
                assert task.done()
                assert not task.cancelled()

            request.addfinalizer(check_finished_normally)
            asyncio.get_running_loop().call_soon(interrupt)
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                pass

        @pytest.mark.asyncio
        async def test_next():
            pytest.fail("the session should have stopped")
        """))
    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)
    assert result.ret == pytest.ExitCode.INTERRUPTED
    result.assert_outcomes()


@_REQUIRES_311
def test_an_interruption_is_a_cancellation_request_on_the_test_task(
    pytester: Pytester,
):
    """A test awaiting a child task can tell the interruption from the child's."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest

        def interrupt():
            raise KeyboardInterrupt

        @pytest.mark.asyncio
        async def test_parent(request):
            task = asyncio.current_task()

            def check_cancelled_task():
                assert task.cancelled()
                assert task.cancelling() == 1

            request.addfinalizer(check_cancelled_task)
            asyncio.get_running_loop().call_soon(interrupt)
            child = asyncio.create_task(asyncio.Event().wait())
            await child
        """))
    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)
    assert result.ret == pytest.ExitCode.INTERRUPTED
    result.assert_outcomes()


def test_a_loop_stopped_after_the_test_returned_does_not_refuse_later_tests(
    pytester: Pytester,
):
    """There is nothing to cancel: the test fails, and the shared loop goes on."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest

        @pytest.mark.asyncio(loop_scope="module")
        async def test_stop():
            loop = asyncio.get_running_loop()
            loop.call_soon(loop.stop)

        @pytest.mark.asyncio(loop_scope="module")
        async def test_next():
            pass
        """))
    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)
    result.assert_outcomes(failed=1, passed=1)
    result.stdout.fnmatch_lines(
        [
            "*_ test_stop _*",
            "*RuntimeError: Event loop stopped before Future completed*",
        ]
    )


@_REQUIRES_311
def test_a_fixture_failing_while_the_interrupted_test_cleans_up_ends_the_cleanup(
    pytester: Pytester,
):
    """
    The cleanup is cancelled once more, even after it resolved the
    interruption's request, and the loop refuses the later tests.
    """
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest
        import pytest_asyncio

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
                await asyncio.Event().wait()

        @pytest.mark.asyncio(loop_scope="module")
        async def test_later(service):
            pytest.fail("the failed service's loop should refuse this test")
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
    pytester.makepyfile(dedent(f"""\
        import asyncio
        import os
        import signal

        import pytest
        import pytest_asyncio

        def interrupt():
            raise KeyboardInterrupt

        @pytest_asyncio.fixture
        async def resource(request):
            task = asyncio.current_task()

            def check_cancelled_task():
                assert task.cancelled()

            request.addfinalizer(check_cancelled_task)
            yield
            loop = asyncio.get_running_loop()
            {interrupt}
            await asyncio.Event().wait()

        @pytest.mark.asyncio
        async def test_it(resource):
            pass
        """))
    result = pytester.runpytest_subprocess(
        "--asyncio-mode=strict", "-W", "error", timeout=30
    )
    assert result.ret == pytest.ExitCode.INTERRUPTED
    result.assert_outcomes(passed=1)
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
    pytester.makepyfile(dedent("""\
        import asyncio
        from io import StringIO
        from pathlib import Path

        import pytest
        import pytest_asyncio

        @pytest_asyncio.fixture
        async def resource():
            task = asyncio.current_task()
            with StringIO("report contents") as report:
                try:
                    yield report
                finally:
                    await asyncio.sleep(0)
                    assert asyncio.current_task() is task
                    Path("report.txt").write_text(report.getvalue())

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
    assert (pytester.path / "report.txt").read_text() == "report contents"
    assert "Task was destroyed" not in result.stdout.str() + result.stderr.str()


@_POSIX
def test_pytest_timeout_fails_a_test_hung_at_an_await(pytester: Pytester):
    """The timed-out test can save its result, and the shared loop serves the next."""
    pytest.importorskip("pytest_timeout")
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest
        import pytest_asyncio

        @pytest_asyncio.fixture(loop_scope="module")
        async def output():
            with open("report.txt", "w") as stream:
                yield stream
                await asyncio.sleep(0)

        @pytest.mark.timeout(0.5, func_only=True)
        @pytest.mark.asyncio(loop_scope="module")
        async def test_hang(output):
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0)
                output.write("partial result")

        @pytest.mark.asyncio(loop_scope="module")
        async def test_next():
            pass
        """))
    result = pytester.runpytest_subprocess(
        "--asyncio-mode=strict", "-o", "timeout_method=signal", timeout=30
    )
    result.assert_outcomes(failed=1, passed=1)
    result.stdout.fnmatch_lines(["*_ test_hang _*", "*Failed: Timeout*0.5s*"])
    assert (pytester.path / "report.txt").read_text() == "partial result"


@_POSIX
def test_pytest_timeout_fails_a_test_hung_in_a_busy_loop(pytester: Pytester):
    """The signal fails the test from its own frame; the loop serves the next test."""
    pytest.importorskip("pytest_timeout")
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
        import pytest

        @pytest.mark.timeout(0.5, func_only=True)
        @pytest.mark.asyncio(loop_scope="module")
        async def test_hang():
            while True:
                pass

        @pytest.mark.asyncio(loop_scope="module")
        async def test_next():
            pass
        """))
    result = pytester.runpytest_subprocess(
        "--asyncio-mode=strict", "-o", "timeout_method=signal", timeout=30
    )
    result.assert_outcomes(failed=1, passed=1)
    result.stdout.fnmatch_lines(["*_ test_hang _*", "*Failed: Timeout*0.5s*"])


def test_a_fixture_interrupted_after_it_yielded_is_torn_down_before_its_parent(
    pytester: Pytester,
):
    """A fixture pytest never received can still save to its parent's open file."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest
        import pytest_asyncio

        def interrupt():
            raise KeyboardInterrupt

        @pytest.fixture
        def output():
            with open("report.txt", "w") as stream:
                yield stream

        @pytest_asyncio.fixture
        async def report(output):
            asyncio.get_running_loop().call_soon(interrupt)
            yield output
            await asyncio.sleep(0)
            output.write("report contents")

        @pytest.mark.asyncio
        async def test_query(report):
            pytest.fail("fixture setup should have been interrupted")
        """))
    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)
    assert result.ret == pytest.ExitCode.INTERRUPTED
    result.assert_outcomes()
    assert (pytester.path / "report.txt").read_text() == "report contents"


@_REQUIRES_311
def test_a_fixture_setup_that_recovers_from_the_interruption_and_yields_is_torn_down(
    pytester: Pytester,
):
    """Recovering setup is finalized while it can still write to its parent's file."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest
        import pytest_asyncio

        def interrupt():
            raise KeyboardInterrupt

        @pytest.fixture
        def output():
            with open("report.txt", "w") as stream:
                yield stream

        @pytest_asyncio.fixture
        async def report(output):
            asyncio.get_running_loop().call_soon(interrupt)
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                asyncio.current_task().uncancel()
            yield output
            await asyncio.sleep(0)
            output.write("report contents")

        @pytest.mark.asyncio
        async def test_query(report):
            pytest.fail("fixture setup should have been interrupted")
        """))
    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)
    assert result.ret == pytest.ExitCode.INTERRUPTED
    result.assert_outcomes()
    assert (pytester.path / "report.txt").read_text() == "report contents"


def test_a_teardown_error_of_a_fixture_pytest_never_received_is_reported(
    pytester: Pytester,
):
    """The error is the fixture's setup error, and the session goes on."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest
        import pytest_asyncio

        def interrupt():
            raise KeyboardInterrupt

        @pytest_asyncio.fixture
        async def child():
            asyncio.get_running_loop().call_soon(interrupt)
            yield
            await asyncio.sleep(0)
            raise ValueError("child cleanup failed")

        @pytest.mark.asyncio
        async def test_query(child):
            pytest.fail("fixture setup should have failed")

        @pytest.mark.asyncio
        async def test_next():
            pass
        """))
    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    result.assert_outcomes(errors=1, passed=1)
    result.stdout.fnmatch_lines(
        ["*ERROR at setup of test_query*", "*ValueError: child cleanup failed*"]
    )


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


def test_cancelled_fixture_returning_without_yield_preserves_keyboard_interrupt(
    pytester: Pytester,
):
    """A missing yield must not replace KeyboardInterrupt after cancelled setup."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
        import asyncio
        from pathlib import Path

        import pytest
        import pytest_asyncio

        def interrupt():
            raise KeyboardInterrupt

        @pytest_asyncio.fixture
        async def resource():
            asyncio.get_running_loop().call_soon(interrupt)
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                Path("report.txt").write_text("partial result")
                return
            yield

        @pytest.mark.asyncio
        async def test_it(resource):
            pytest.fail("fixture setup should have been interrupted")
        """))
    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)
    assert result.ret == pytest.ExitCode.INTERRUPTED
    assert "StopAsyncIteration" not in result.stdout.str()
    result.assert_outcomes()
    assert (pytester.path / "report.txt").read_text() == "partial result"


def test_fixture_setup_returning_without_yield_reports_a_pending_keyboard_interrupt(
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
