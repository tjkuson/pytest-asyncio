"""
Tests for the runtime execution model.

pytest-asyncio classifies test items at runtime from their ``asyncio`` marker and
their callable, rather than converting them to a dedicated item subclass during
collection. These tests cover behavior specific to that model: the deprecation of the
removed ``PytestAsyncioFunction`` item type, markers applied via a parametrize
parameter set, and the classification of Hypothesis and staticmethod edge cases.
"""

from __future__ import annotations

from textwrap import dedent

import pytest
from pytest import Pytester


def test_pytestasyncio_function_is_removed():
    """The removed PytestAsyncioFunction item type is no longer importable."""
    import pytest_asyncio.plugin as plugin

    assert not hasattr(plugin, "PytestAsyncioFunction")
    with pytest.raises(ImportError):
        from pytest_asyncio.plugin import PytestAsyncioFunction  # noqa: F401


def test_marker_on_parameter_set_is_honored_at_runtime(pytester: Pytester):
    """A marker on a parametrize parameter set is recognized at runtime (see #1463)."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
        import asyncio
        import pytest

        @pytest.mark.parametrize(
            "backend",
            [pytest.param("asyncio", marks=pytest.mark.asyncio), "plain"],
        )
        async def test_async(backend):
            assert asyncio.get_running_loop()
        """))
    result = pytester.runpytest("--asyncio-mode=strict")
    # The asyncio-marked parameter runs in an event loop; the unmarked parameter is an
    # unhandled coroutine and fails, as expected under strict mode.
    result.assert_outcomes(passed=1, failed=1)


def test_marker_on_parameter_set_with_loop_factories_errors(pytester: Pytester):
    """Loop factories cannot be driven by a marker that is only on a parameter set."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makeconftest(dedent("""\
        import asyncio

        def pytest_asyncio_loop_factories(config, item):
            return {"custom": asyncio.new_event_loop}
        """))
    pytester.makepyfile(dedent("""\
        import pytest

        @pytest.mark.parametrize(
            "backend",
            [pytest.param("asyncio", marks=pytest.mark.asyncio)],
        )
        async def test_async(backend):
            pass
        """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(
        ["*was not visible when loop-factory parametrization ran*"]
    )


def test_sync_hypothesis_test_with_asyncio_marker_is_not_adopted(pytester: Pytester):
    """A synchronous Hypothesis test carrying the asyncio marker is not run async."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
        import pytest
        from hypothesis import given, strategies as st

        @pytest.mark.asyncio
        @given(st.integers())
        def test_sync_hypothesis(value):
            assert isinstance(value, int)
        """))
    result = pytester.runpytest("--asyncio-mode=strict", "-W", "default")
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(
        ["*is marked with '@pytest.mark.asyncio' but it is not an async function*"]
    )


def test_staticmethod_async_generator_is_xfailed(pytester: Pytester):
    """A staticmethod async-generator test is recognized and xfailed, not run."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
        class TestStaticAsyncGen:
            @staticmethod
            async def test_static_async_gen():
                yield
        """))
    result = pytester.runpytest("--asyncio-mode=auto", "-W", "default")
    result.assert_outcomes(xfailed=1)
    result.stdout.fnmatch_lines(
        ["*Tests based on asynchronous generators are not supported*"]
    )
