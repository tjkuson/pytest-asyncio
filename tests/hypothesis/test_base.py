"""
Tests for the Hypothesis integration, which wraps async functions in a
sync shim for Hypothesis.
"""

from __future__ import annotations

from textwrap import dedent

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


def test_each_example_runs_in_a_task_of_its_own(pytester: Pytester):
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest
        from hypothesis import given, settings, strategies as st

        tasks = []

        @pytest.mark.asyncio(loop_scope="module")
        @settings(max_examples=3, deadline=None, database=None)
        @given(st.integers())
        async def test_examples(n):
            tasks.append(asyncio.current_task())

        def test_tasks():
            assert len(tasks) > 1
            assert len(set(tasks)) == len(tasks)
        """))
    result = pytester.runpytest("--asyncio-mode=strict", "-W default")
    result.assert_outcomes(passed=2)
