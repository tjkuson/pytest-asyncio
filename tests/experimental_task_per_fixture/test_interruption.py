"""
Ctrl-C and pytest-timeout with asyncio_experimental_task_per_fixture.

The interrupted test or fixture is cancelled, and its cleanup runs while its
fixtures are still set up. A second Ctrl-C stops waiting for that cleanup.
A loop callback raising KeyboardInterrupt stands in for Ctrl-C.
"""

from __future__ import annotations

import sys
from textwrap import dedent

import pytest
from pytest import Pytester

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="asyncio_experimental_task_per_fixture requires Python 3.11",
)


@pytest.mark.skipif(sys.platform == "win32", reason="Uses pytest-timeout's signals")
def test_pytest_timeout_lets_the_test_cleanup_use_its_fixtures(pytester: Pytester):
    """After a timeout, the test's cleanup runs before its fixtures are torn down."""
    pytester.makeini(
        "[pytest]\n"
        "asyncio_experimental_task_per_fixture = true\n"
        "asyncio_default_fixture_loop_scope = function"
    )
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest
        import pytest_asyncio

        @pytest_asyncio.fixture(loop_scope="module")
        async def report():
            with open("report.txt", "w") as stream:
                yield stream

        @pytest.mark.timeout(0.5, func_only=True)
        @pytest.mark.asyncio(loop_scope="module")
        async def test_hang(report):
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0)
                report.write("saved during cleanup")

        @pytest.mark.asyncio(loop_scope="module")
        async def test_next():
            pass
        """))

    result = pytester.runpytest_subprocess(
        "--asyncio-mode=strict",
        "-p",
        "timeout",
        "-o",
        "timeout_method=signal",
        timeout=30,
    )

    result.assert_outcomes(failed=1, passed=1)
    result.stdout.fnmatch_lines(["*_ test_hang _*", "*Failed: Timeout*0.5s*"])
    assert (pytester.path / "report.txt").read_text() == "saved during cleanup"


def test_a_cleanup_error_after_ctrl_c_fails_the_test_and_the_session_continues(
    pytester: Pytester,
):
    """The error raised by the test's cleanup is reported instead of the Ctrl-C."""
    pytester.makeini(
        "[pytest]\n"
        "asyncio_experimental_task_per_fixture = true\n"
        "asyncio_default_fixture_loop_scope = function"
    )
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest

        def ctrl_c():
            raise KeyboardInterrupt

        @pytest.mark.asyncio(loop_scope="module")
        async def test_interrupted():
            asyncio.get_running_loop().call_soon(ctrl_c)
            try:
                await asyncio.Event().wait()
            finally:
                raise ValueError("cleanup failed")

        @pytest.mark.asyncio(loop_scope="module")
        async def test_next():
            pass
        """))

    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)

    assert result.ret == pytest.ExitCode.TESTS_FAILED
    result.assert_outcomes(failed=1, passed=1)
    result.stdout.fnmatch_lines(
        ["*_ test_interrupted _*", "*ValueError: cleanup failed", "*KeyboardInterrupt*"]
    )


@pytest.mark.skipif(sys.platform == "win32", reason="Uses pytest-timeout's signals")
def test_a_cleanup_error_after_pytest_timeout_is_reported_with_the_timeout(
    pytester: Pytester,
):
    """The error raised by the test's cleanup is reported, noting the timeout."""
    pytester.makeini(
        "[pytest]\n"
        "asyncio_experimental_task_per_fixture = true\n"
        "asyncio_default_fixture_loop_scope = function"
    )
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest

        @pytest.mark.timeout(0.1, func_only=True)
        @pytest.mark.asyncio
        async def test_hang():
            try:
                await asyncio.Event().wait()
            finally:
                raise ConnectionError("cleanup failed")

        @pytest.mark.asyncio
        async def test_next():
            pass
        """))

    result = pytester.runpytest_subprocess(
        "--asyncio-mode=strict",
        "-p",
        "timeout",
        "-o",
        "timeout_method=signal",
        timeout=30,
    )

    result.assert_outcomes(failed=1, passed=1)
    result.stdout.fnmatch_lines(
        ["*_ test_hang _*", "*ConnectionError: cleanup failed", "*Timeout*"]
    )


def test_a_cleanup_error_in_a_fixture_setup_after_ctrl_c_is_a_setup_error(
    pytester: Pytester,
):
    """The error raised by the setup's cleanup is its test's setup error."""
    pytester.makeini(
        "[pytest]\n"
        "asyncio_experimental_task_per_fixture = true\n"
        "asyncio_default_fixture_loop_scope = function"
    )
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest
        import pytest_asyncio

        def ctrl_c():
            raise KeyboardInterrupt

        @pytest_asyncio.fixture
        async def resource():
            asyncio.get_running_loop().call_soon(ctrl_c)
            try:
                await asyncio.Event().wait()
                yield
            finally:
                raise ValueError("setup cleanup failed")

        @pytest.mark.asyncio
        async def test_uses_resource(resource):
            pytest.fail("the fixture setup should have failed")

        @pytest.mark.asyncio
        async def test_next():
            pass
        """))

    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)

    assert result.ret == pytest.ExitCode.TESTS_FAILED
    result.assert_outcomes(errors=1, passed=1)
    result.stdout.fnmatch_lines(
        [
            "*ERROR at setup of test_uses_resource*",
            "*ValueError: setup cleanup failed",
            "*KeyboardInterrupt*",
        ]
    )


def test_a_fixture_failing_during_the_test_cleanup_after_ctrl_c_cancels_that_cleanup(
    pytester: Pytester,
):
    """The cleanup does not hang waiting for the failed fixture; the session stops."""
    pytester.makeini(
        "[pytest]\n"
        "asyncio_experimental_task_per_fixture = true\n"
        "asyncio_default_fixture_loop_scope = function"
    )
    pytester.makepyfile(dedent("""\
        import asyncio
        from pathlib import Path

        import pytest
        import pytest_asyncio

        def ctrl_c():
            raise KeyboardInterrupt

        @pytest_asyncio.fixture
        async def service():
            failure = asyncio.Event()

            async def fail():
                await failure.wait()
                raise RuntimeError("service failed")

            async with asyncio.TaskGroup() as group:
                group.create_task(fail())
                yield failure

        @pytest.mark.asyncio
        async def test_interrupted(service):
            asyncio.get_running_loop().call_soon(ctrl_c)
            try:
                await asyncio.Event().wait()
            finally:
                service.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    Path("cleanup.txt").write_text("cancelled")
                    raise
        """))

    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)

    result.stdout.fnmatch_lines(["*! KeyboardInterrupt !*"])
    assert (pytester.path / "cleanup.txt").read_text() == "cancelled"


def test_a_fixture_that_yields_as_ctrl_c_arrives_is_torn_down_before_its_parent(
    pytester: Pytester,
):
    """Pytest never receives the fixture; its teardown runs while its parent is open."""
    pytester.makeini(
        "[pytest]\n"
        "asyncio_experimental_task_per_fixture = true\n"
        "asyncio_default_fixture_loop_scope = function"
    )
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest
        import pytest_asyncio

        def ctrl_c():
            raise KeyboardInterrupt

        @pytest.fixture
        def output():
            with open("report.txt", "w") as stream:
                yield stream

        @pytest_asyncio.fixture
        async def report(output):
            asyncio.get_running_loop().call_soon(ctrl_c)
            yield output
            await asyncio.sleep(0)
            output.write("saved during teardown")

        @pytest.mark.asyncio
        async def test_uses_report(report):
            pytest.fail("the fixture setup should have been interrupted")
        """))

    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)

    assert result.ret == pytest.ExitCode.INTERRUPTED
    result.assert_outcomes()
    assert (pytester.path / "report.txt").read_text() == "saved during teardown"


@pytest.mark.parametrize("error", ["RuntimeError", "SystemExit"])
def test_an_error_from_test_cleanup_abandoned_by_a_second_ctrl_c_is_logged_once(
    pytester: Pytester, error: str
):
    """
    A second Ctrl-C stops waiting for the test's cleanup, whose error is logged.

    After an interrupted session, pytest shows asyncio's log records only with
    live logging, so the session enables it.
    """
    pytester.makeini(
        "[pytest]\n"
        "asyncio_experimental_task_per_fixture = true\n"
        "asyncio_default_fixture_loop_scope = function"
    )
    pytester.makepyfile(dedent(f"""\
        import asyncio

        import pytest

        def ctrl_c():
            raise KeyboardInterrupt

        @pytest.mark.asyncio
        async def test_interrupted():
            loop = asyncio.get_running_loop()
            loop.call_soon(ctrl_c)
            try:
                await asyncio.Event().wait()
            finally:
                loop.call_soon(ctrl_c)
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    raise {error}("abandoned cleanup failed") from None
        """))

    result = pytester.runpytest_subprocess(
        "--asyncio-mode=strict", "-W", "error", "-o", "log_cli=true", timeout=30
    )

    assert result.ret == pytest.ExitCode.INTERRUPTED
    output = result.stdout.str() + result.stderr.str()
    assert output.count(f"{error}: abandoned cleanup failed") == 1
    for noise in ("Task was destroyed", "never awaited", "never retrieved"):
        assert noise not in output


def test_an_error_from_fixture_setup_abandoned_by_a_second_ctrl_c_is_logged_once(
    pytester: Pytester,
):
    """
    A fixture setup that survives both Ctrl-Cs and yields has its error logged.

    After an interrupted session, pytest shows asyncio's log records only with
    live logging, so the session enables it.
    """
    pytester.makeini(
        "[pytest]\n"
        "asyncio_experimental_task_per_fixture = true\n"
        "asyncio_default_fixture_loop_scope = function"
    )
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest
        import pytest_asyncio

        def ctrl_c():
            raise KeyboardInterrupt

        @pytest_asyncio.fixture
        async def resource():
            loop = asyncio.get_running_loop()
            loop.call_soon(ctrl_c)
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                loop.call_soon(ctrl_c)
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    pass
            try:
                yield
            finally:
                raise ValueError("abandoned fixture cleanup failed")

        @pytest.mark.asyncio
        async def test_uses_resource(resource):
            pytest.fail("the fixture setup should have been interrupted")
        """))

    result = pytester.runpytest_subprocess(
        "--asyncio-mode=strict", "-W", "error", "-o", "log_cli=true", timeout=30
    )

    assert result.ret == pytest.ExitCode.INTERRUPTED
    output = result.stdout.str() + result.stderr.str()
    assert output.count("ValueError: abandoned fixture cleanup failed") == 1
    for noise in ("Task was destroyed", "never awaited", "never retrieved"):
        assert noise not in output


def test_an_error_from_fixture_teardown_abandoned_by_a_second_ctrl_c_is_logged_once(
    pytester: Pytester,
):
    """
    A second Ctrl-C stops waiting for the fixture's teardown, whose error is logged.

    After an interrupted session, pytest shows asyncio's log records only with
    live logging, so the session enables it.
    """
    pytester.makeini(
        "[pytest]\n"
        "asyncio_experimental_task_per_fixture = true\n"
        "asyncio_default_fixture_loop_scope = function"
    )
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest
        import pytest_asyncio

        def ctrl_c():
            raise KeyboardInterrupt

        @pytest_asyncio.fixture
        async def resource():
            yield
            loop = asyncio.get_running_loop()
            loop.call_soon(ctrl_c)
            try:
                await asyncio.Event().wait()
            finally:
                loop.call_soon(ctrl_c)
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    raise ValueError("abandoned teardown failed") from None

        @pytest.mark.asyncio
        async def test_uses_resource(resource):
            pass
        """))

    result = pytester.runpytest_subprocess(
        "--asyncio-mode=strict", "-W", "error", "-o", "log_cli=true", timeout=30
    )

    assert result.ret == pytest.ExitCode.INTERRUPTED
    output = result.stdout.str() + result.stderr.str()
    assert output.count("ValueError: abandoned teardown failed") == 1
    for noise in ("Task was destroyed", "never awaited", "never retrieved"):
        assert noise not in output


def test_ctrl_c_while_the_loop_waits_for_abandoned_cleanup_still_closes_it(
    pytester: Pytester,
):
    """
    A third Ctrl-C, while the loop closes, stops waiting and still closes the loop.

    Pytest does not handle a KeyboardInterrupt raised as its session finishes,
    so the process ends as for any uncaught KeyboardInterrupt.
    """
    pytester.makeini(
        "[pytest]\n"
        "asyncio_experimental_task_per_fixture = true\n"
        "asyncio_default_fixture_loop_scope = function"
    )
    pytester.makeconftest(dedent("""\
        import asyncio
        from pathlib import Path

        loops = []

        def loop_factory():
            loop = asyncio.new_event_loop()
            loops.append(loop)
            return loop

        def pytest_asyncio_loop_factories(config, item):
            return {"tracked": loop_factory}

        def pytest_unconfigure(config):
            closed = [loop.is_closed() for loop in loops]
            Path("loops-closed.txt").write_text(str(closed))
        """))
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest

        def ctrl_c():
            raise KeyboardInterrupt

        @pytest.mark.asyncio
        async def test_interrupted():
            loop = asyncio.get_running_loop()
            loop.call_soon(ctrl_c)
            try:
                await asyncio.Event().wait()
            finally:
                loop.call_soon(ctrl_c)
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    loop.call_soon(ctrl_c)
                    await asyncio.Event().wait()
        """))

    pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)

    assert (pytester.path / "loops-closed.txt").read_text() == "[True]"


def test_a_fixture_setup_that_survives_a_second_ctrl_c_is_torn_down_once_it_yields(
    pytester: Pytester,
):
    """Pytest never receives the fixture, so its yield returns into its teardown."""
    pytester.makeini(
        "[pytest]\n"
        "asyncio_experimental_task_per_fixture = true\n"
        "asyncio_default_fixture_loop_scope = function"
    )
    pytester.makepyfile(dedent("""\
        import asyncio
        from pathlib import Path

        import pytest
        import pytest_asyncio

        resource_yielded = asyncio.Event()

        def ctrl_c():
            raise KeyboardInterrupt

        @pytest_asyncio.fixture
        async def teardown_waiting_for_resource():
            yield
            await resource_yielded.wait()

        @pytest_asyncio.fixture
        async def resource():
            loop = asyncio.get_running_loop()
            loop.call_soon(ctrl_c)
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                loop.call_soon(ctrl_c)
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    pass
            resource_yielded.set()
            try:
                yield
            except asyncio.CancelledError:
                Path("teardown.txt").write_text("cancelled at the yield")
            else:
                Path("teardown.txt").write_text("returned from the yield")

        @pytest.mark.asyncio
        async def test_uses_resource(teardown_waiting_for_resource, resource):
            pytest.fail("the fixture setup should have been interrupted")
        """))

    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)

    assert result.ret == pytest.ExitCode.INTERRUPTED
    assert (pytester.path / "teardown.txt").read_text() == "returned from the yield"
