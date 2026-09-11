"""Async tests that cancel their own task."""

from __future__ import annotations

from textwrap import dedent

import pytest
from pytest import Pytester


def test_a_test_that_cancels_itself_fails_and_its_fixture_is_torn_down_normally(
    pytester: Pytester,
):
    """Only that test fails; its fixture's teardown runs to its end."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
        import asyncio
        from pathlib import Path

        import pytest
        import pytest_asyncio

        @pytest_asyncio.fixture(loop_scope="module")
        async def resource():
            yield
            await asyncio.sleep(0)
            Path("teardown.txt").write_text("finished")

        @pytest.mark.asyncio(loop_scope="module")
        async def test_cancels_itself(resource):
            asyncio.current_task().cancel()
            await asyncio.sleep(0)

        @pytest.mark.asyncio(loop_scope="module")
        async def test_next():
            pass
        """))

    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)

    result.assert_outcomes(failed=1, passed=1)
    result.stdout.fnmatch_lines(["*_ test_cancels_itself _*", "*CancelledError*"])
    assert (pytester.path / "teardown.txt").read_text() == "finished"


def test_a_cancellation_pending_as_the_test_returns_fails_it(pytester: Pytester):
    """The test fails as cancelled, as its task would under asyncio.run()."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest

        @pytest.mark.asyncio(loop_scope="module")
        async def test_cancels_itself():
            asyncio.current_task().cancel()

        @pytest.mark.asyncio(loop_scope="module")
        async def test_next():
            pass
        """))

    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)

    result.assert_outcomes(failed=1, passed=1)
    result.stdout.fnmatch_lines(["*_ test_cancels_itself _*", "*CancelledError*"])


def test_a_test_that_suppresses_its_own_cancellation_passes(pytester: Pytester):
    """Nothing else is cancelled, so later tests on the loop run."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
        import asyncio
        import contextlib

        import pytest

        @pytest.mark.asyncio(loop_scope="module")
        async def test_cancels_itself():
            asyncio.current_task().cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.sleep(0)

        @pytest.mark.asyncio(loop_scope="module")
        async def test_next():
            pass
        """))

    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)

    result.assert_outcomes(passed=2)


def test_an_error_raised_after_the_test_cancels_itself_is_its_failure(
    pytester: Pytester,
):
    """The error, not the pending cancellation, is reported, as in any task."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest

        @pytest.mark.asyncio(loop_scope="module")
        async def test_cancels_itself():
            asyncio.current_task().cancel()
            raise RuntimeError("raised after the cancellation")

        @pytest.mark.asyncio(loop_scope="module")
        async def test_next():
            pass
        """))

    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)

    result.assert_outcomes(failed=1, passed=1)
    result.stdout.fnmatch_lines(
        [
            "*_ test_cancels_itself _*",
            "*RuntimeError: raised after the cancellation*",
        ]
    )


def test_a_keyboard_interrupt_raised_after_the_test_cancels_itself_stops_the_session(
    pytester: Pytester,
):
    """The KeyboardInterrupt, not the pending cancellation, is reported."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest

        @pytest.mark.asyncio(loop_scope="module")
        async def test_cancels_itself():
            asyncio.current_task().cancel()
            raise KeyboardInterrupt

        @pytest.mark.asyncio(loop_scope="module")
        async def test_next():
            pytest.fail("the session should have stopped")
        """))

    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)

    assert result.ret == pytest.ExitCode.INTERRUPTED
    result.assert_outcomes()
