"""Fixture requests made from running async tests."""

from __future__ import annotations

import sys
from textwrap import dedent

import pytest
from pytest import Pytester


@pytest.mark.parametrize(
    "statement", ["return", "yield"], ids=["coroutine", "generator"]
)
def test_session_loop_remains_usable_after_rejected_async_fixture_request(
    pytester: Pytester, statement: str
):
    """Rejection leaves later tests using the requested loop scope unaffected."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent(f"""\
        from pathlib import Path

        import pytest
        import pytest_asyncio

        @pytest_asyncio.fixture(scope="session", loop_scope="session")
        async def value():
            Path("fixture-started").touch()
            {statement} 42

        @pytest.mark.asyncio
        async def test_rejects_new_async_setup(request):
            with pytest.raises(RuntimeError, match="event loop is running"):
                request.getfixturevalue("value")

        @pytest.mark.asyncio(loop_scope="session")
        async def test_session_loop_is_usable():
            pass
        """))
    result = pytester.runpytest_subprocess(
        "--asyncio-mode=strict", "-W", "error", timeout=30
    )
    result.assert_outcomes(passed=2)
    assert not (pytester.path / "fixture-started").exists()


def test_repeated_async_fixture_requests_report_the_original_setup_error(
    pytester: Pytester,
):
    """Pytest caches a rejected setup like any other fixture setup error."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
        import pytest
        import pytest_asyncio

        @pytest_asyncio.fixture(scope="session", loop_scope="session")
        async def value():
            return 42

        @pytest.mark.asyncio
        async def test_repeated_request(request):
            with pytest.raises(RuntimeError, match="event loop is running") as first:
                request.getfixturevalue("value")
            with pytest.raises(RuntimeError, match="event loop is running") as second:
                request.getfixturevalue("value")
            assert second.value is first.value
        """))
    result = pytester.runpytest_subprocess(
        "--asyncio-mode=strict", "-W", "error", timeout=30
    )
    result.assert_outcomes(passed=1)


@pytest.mark.parametrize(
    "statement", ["return", "yield"], ids=["function", "generator"]
)
def test_async_test_can_set_up_sync_fixture_with_a_new_session_loop(
    pytester: Pytester, statement: str
):
    """Synchronous setup does not need to run the fixture's configured loop."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent(f"""\
        import pytest
        import pytest_asyncio

        @pytest_asyncio.fixture(scope="session", loop_scope="session")
        def value():
            {statement} 42

        @pytest.mark.asyncio
        async def test_requests_sync_fixture(request):
            assert request.getfixturevalue("value") == 42

        @pytest.mark.asyncio(loop_scope="session")
        async def test_uses_the_fixture_on_its_session_loop(value):
            assert value == 42
        """))
    result = pytester.runpytest_subprocess(
        "--asyncio-mode=strict", "-W", "error", timeout=30
    )
    result.assert_outcomes(passed=2)


def test_async_test_can_retrieve_an_already_initialized_async_fixture(
    pytester: Pytester,
):
    """Retrieving a cached fixture value does not require nested async setup."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
        import pytest
        import pytest_asyncio

        @pytest_asyncio.fixture(scope="session", loop_scope="session")
        async def value():
            yield object()

        @pytest.mark.asyncio
        async def test_retrieves_existing_value(value, request):
            assert request.getfixturevalue("value") is value
        """))
    result = pytester.runpytest_subprocess(
        "--asyncio-mode=strict", "-W", "error", timeout=30
    )
    result.assert_outcomes(passed=1)


def test_async_tests_in_a_running_loop_report_setup_errors_without_coroutine_warnings(
    pytester: Pytester,
):
    """Nested pytest reports the unsupported execution before creating async work."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(test_nested=dedent("""\
        import pytest

        @pytest.mark.asyncio
        async def test_async():
            pass
        """))
    launcher = pytester.makepyfile(run_pytest=dedent("""\
        import asyncio

        import pytest

        async def run():
            return pytest.main(
                ["test_nested.py", "--asyncio-mode=strict", "-W", "error"]
            )

        raise SystemExit(asyncio.run(run()))
        """))
    result = pytester.run(sys.executable, str(launcher), timeout=30)
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(
        ["*RuntimeError: pytest-asyncio cannot run async tests*running event loop*"]
    )
    output = result.stdout.str() + result.stderr.str()
    assert "never awaited" not in output
    assert "never retrieved" not in output
