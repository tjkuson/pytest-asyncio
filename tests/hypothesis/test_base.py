"""
Tests for the Hypothesis integration, which wraps async functions in a
sync shim for Hypothesis.
"""

from __future__ import annotations

import sys
from textwrap import dedent

import pytest
from pytest import Pytester


def test_hypothesis_given_decorator_before_asyncio_mark(pytester: Pytester):
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
            import pytest
            from hypothesis import given, strategies as st

            @given(st.integers())
            @pytest.mark.asyncio
            async def test_mark_inner(n):
                assert isinstance(n, int)
            """))
    result = pytester.runpytest("--asyncio-mode=strict", "-W default")
    result.assert_outcomes(passed=1)


def test_hypothesis_given_decorator_after_asyncio_mark(pytester: Pytester):
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
            import pytest
            from hypothesis import given, strategies as st

            @pytest.mark.asyncio
            @given(st.integers())
            async def test_mark_outer(n):
                assert isinstance(n, int)
            """))
    result = pytester.runpytest("--asyncio-mode=strict", "-W default")
    result.assert_outcomes(passed=1)


def test_parametrization(pytester: Pytester):
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
            import pytest
            from hypothesis import given, strategies as st

            @pytest.mark.parametrize("y", [1, 2])
            @given(x=st.none())
            @pytest.mark.asyncio
            async def test_mark_and_parametrize(x, y):
                assert x is None
                assert y in (1, 2)
            """))
    result = pytester.runpytest("--asyncio-mode=strict", "-W default")
    result.assert_outcomes(passed=2)


def test_async_auto_marked(pytester: Pytester):
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
        import asyncio
        import pytest
        from hypothesis import given
        import hypothesis.strategies as st

        pytest_plugins = 'pytest_asyncio'

        @given(n=st.integers())
        async def test_hypothesis(n: int):
            assert isinstance(n, int)
        """))
    result = pytester.runpytest("--asyncio-mode=auto")
    result.assert_outcomes(passed=1)


def test_sync_not_auto_marked(pytester: Pytester):
    """Assert that synchronous Hypothesis functions are not marked with asyncio"""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
        import asyncio
        import pytest
        from hypothesis import given
        import hypothesis.strategies as st

        pytest_plugins = 'pytest_asyncio'

        @given(n=st.integers())
        def test_hypothesis(request, n: int):
            markers = [marker.name for marker in request.node.own_markers]
            assert "asyncio" not in markers
            assert isinstance(n, int)
        """))
    result = pytester.runpytest("--asyncio-mode=auto")
    result.assert_outcomes(passed=1)


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="The experimental runner requires Python 3.11"
)
def test_experimental_runner_isolates_context_variables_between_examples(
    pytester: Pytester,
):
    """Each example starts without context assignments made by earlier examples."""
    pytester.makeini(dedent("""
        [pytest]
        asyncio_default_fixture_loop_scope = function
        experimental_asyncio_task_group_runner = true
        """))
    pytester.makepyfile(dedent("""
        from contextvars import ContextVar

        import pytest
        from hypothesis import example, given, settings, strategies as st

        request_id = ContextVar("request_id", default="initial")

        @pytest.mark.asyncio(loop_scope="module")
        @settings(database=None, deadline=None)
        @example(value=False)
        @example(value=True)
        @given(value=st.booleans())
        async def test_example(value):
            assert request_id.get() == "initial"
            request_id.set("changed")
        """))

    result = pytester.runpytest("--asyncio-mode=strict")

    result.assert_outcomes(passed=1)
