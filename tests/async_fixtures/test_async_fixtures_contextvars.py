"""
Regression test for https://github.com/pytest-dev/pytest-asyncio/issues/127:
contextvars were not properly maintained among fixtures and tests.
"""

from __future__ import annotations

import sys
from textwrap import dedent
from typing import Literal

import pytest
from pytest import Pytester

_prelude = dedent("""
    import pytest
    import pytest_asyncio
    from contextlib import contextmanager
    from contextvars import ContextVar

    _context_var = ContextVar("context_var")

    @contextmanager
    def context_var_manager(value):
        token = _context_var.set(value)
        try:
            yield
        finally:
            _context_var.reset(token)
""")


def test_var_from_sync_generator_propagates_to_async(pytester: Pytester):
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(_prelude + dedent("""
        @pytest.fixture
        def var_fixture():
            with context_var_manager("value"):
                yield

        @pytest_asyncio.fixture
        async def check_var_fixture(var_fixture):
            assert _context_var.get() == "value"

        @pytest.mark.asyncio
        async def test(check_var_fixture):
            assert _context_var.get() == "value"
        """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(passed=1)


def test_var_from_async_generator_propagates_to_sync(pytester: Pytester):
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(_prelude + dedent("""
        @pytest_asyncio.fixture
        async def var_fixture():
            with context_var_manager("value"):
                yield

        @pytest.fixture
        def check_var_fixture(var_fixture):
            assert _context_var.get() == "value"

        @pytest.mark.asyncio
        async def test(check_var_fixture):
            assert _context_var.get() == "value"
        """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(passed=1)


def test_var_from_async_fixture_propagates_to_sync(pytester: Pytester):
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(_prelude + dedent("""
        @pytest_asyncio.fixture
        async def var_fixture():
            _context_var.set("value")
            # Rely on async fixture teardown to reset the context var.

        @pytest.fixture
        def check_var_fixture(var_fixture):
            assert _context_var.get() == "value"

        def test(check_var_fixture):
            assert _context_var.get() == "value"
        """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(passed=1)


def test_var_from_generator_reset_before_previous_fixture_cleanup(pytester: Pytester):
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(_prelude + dedent("""
        @pytest_asyncio.fixture
        async def no_var_fixture():
            with pytest.raises(LookupError):
                _context_var.get()
            yield
            with pytest.raises(LookupError):
                _context_var.get()

        @pytest_asyncio.fixture
        async def var_fixture(no_var_fixture):
            with context_var_manager("value"):
                yield

        @pytest.mark.asyncio
        async def test(var_fixture):
            assert _context_var.get() == "value"
        """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(passed=1)


def test_var_from_fixture_reset_before_previous_fixture_cleanup(pytester: Pytester):
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(_prelude + dedent("""
        @pytest_asyncio.fixture
        async def no_var_fixture():
            with pytest.raises(LookupError):
                _context_var.get()
            yield
            with pytest.raises(LookupError):
                _context_var.get()

        @pytest_asyncio.fixture
        async def var_fixture(no_var_fixture):
            _context_var.set("value")
            # Rely on async fixture teardown to reset the context var.

        @pytest.mark.asyncio
        async def test(var_fixture):
            assert _context_var.get() == "value"
        """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(passed=1)


def test_var_previous_value_restored_after_fixture(pytester: Pytester):
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(_prelude + dedent("""
        @pytest_asyncio.fixture
        async def var_fixture_1():
            with context_var_manager("value1"):
                yield
                assert _context_var.get() == "value1"

        @pytest_asyncio.fixture
        async def var_fixture_2(var_fixture_1):
            with context_var_manager("value2"):
                yield
                assert _context_var.get() == "value2"

        @pytest.mark.asyncio
        async def test(var_fixture_2):
            assert _context_var.get() == "value2"
        """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(passed=1)


def test_var_set_to_existing_value_ok(pytester: Pytester):
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(_prelude + dedent("""
        @pytest_asyncio.fixture
        async def var_fixture():
            with context_var_manager("value"):
                yield

        @pytest_asyncio.fixture
        async def same_var_fixture(var_fixture):
            with context_var_manager(_context_var.get()):
                yield

        @pytest.mark.asyncio
        async def test(same_var_fixture):
            assert _context_var.get() == "value"
        """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(passed=1)


def test_no_isolation_against_context_changes_in_sync_tests(pytester: Pytester):
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""
            import pytest
            import pytest_asyncio
            from contextvars import ContextVar

            _context_var = ContextVar("my_var")

            def test_sync():
                _context_var.set("new_value")

            @pytest.mark.asyncio
            async def test_async():
                assert _context_var.get() == "new_value"
            """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(passed=2)


@pytest.mark.parametrize("loop_scope", ("function", "module"))
def test_isolation_against_context_changes_in_async_tests(
    pytester: Pytester, loop_scope: Literal["function", "module"]
):
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent(f"""
            import pytest
            import pytest_asyncio
            from contextvars import ContextVar

            _context_var = ContextVar("my_var")

            @pytest.mark.asyncio(loop_scope="{loop_scope}")
            async def test_async_first():
                _context_var.set("new_value")

            @pytest.mark.asyncio(loop_scope="{loop_scope}")
            async def test_async_second():
                with pytest.raises(LookupError):
                    _context_var.get()
            """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(passed=2)


def test_var_from_sync_fixture_seen_by_async_whatever_the_creation_order(
    pytester: Pytester,
):
    """
    Async code sees a var a sync fixture set for as long as it is set, whether
    the fixture runs before the loop is created (the first test's fixtures are
    set up before its loop) or after (the third test's).
    """
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(_prelude + dedent("""
        @pytest.fixture
        def var_fixture():
            with context_var_manager("value"):
                yield

        @pytest.mark.asyncio(loop_scope="module")
        async def test_fixture_before_loop(var_fixture):
            assert _context_var.get() == "value"

        def test_sync():
            with pytest.raises(LookupError):
                _context_var.get()

        @pytest.mark.asyncio(loop_scope="module")
        async def test_fixture_after_loop(var_fixture):
            assert _context_var.get() == "value"

        @pytest.mark.asyncio(loop_scope="module")
        async def test_var_reset():
            with pytest.raises(LookupError):
                _context_var.get()
        """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(passed=4)


def test_var_from_sync_test_propagates_to_async_on_shared_loop(pytester: Pytester):
    """
    Async tests on a module loop see the vars a sync test set, and those a sync
    fixture sets only for as long as the fixture lives.
    """
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(_prelude + dedent("""
        @pytest.mark.asyncio(loop_scope="module")
        async def test_async_before():
            with pytest.raises(LookupError):
                _context_var.get()

        def test_sync():
            _context_var.set("value")

        @pytest.mark.asyncio(loop_scope="module")
        async def test_async_after():
            assert _context_var.get() == "value"

        @pytest.fixture
        def var_fixture():
            with context_var_manager("other"):
                yield

        @pytest.mark.asyncio(loop_scope="module")
        async def test_async_with_fixture(var_fixture):
            assert _context_var.get() == "other"

        @pytest.mark.asyncio(loop_scope="module")
        async def test_async_after_fixture():
            assert _context_var.get() == "value"
        """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(passed=5)


def test_var_reset_by_fixture_after_test_set_it(pytester: Pytester):
    """
    A test's assignment is not seen by the teardown of its fixtures: a fixture
    resetting the token of a var the test then set to another value still sees
    its own value and resets it in its own context, and an earlier fixture's
    teardown sees the var unset.
    """
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(_prelude + dedent("""
        @pytest_asyncio.fixture
        async def no_var_fixture():
            yield
            with pytest.raises(LookupError):
                _context_var.get()

        @pytest_asyncio.fixture
        async def var_fixture(no_var_fixture):
            with context_var_manager("value"):
                yield
                assert _context_var.get() == "value"

        @pytest.mark.asyncio
        async def test(var_fixture):
            assert _context_var.get() == "value"
            _context_var.set("other")
        """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(passed=1)


def test_var_from_fixture_on_shared_loop_unaffected_by_earlier_async_test(
    pytester: Pytester,
):
    """
    An async test's assignment on a module loop is not seen by a fixture set
    up later on that loop: the fixture's assignment of the same object still
    propagates to a sync fixture depending on it, at each setup.
    """
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(_prelude + dedent("""
        VALUE = object()

        @pytest.mark.asyncio(loop_scope="module")
        async def test_set_var():
            _context_var.set(VALUE)

        @pytest_asyncio.fixture(loop_scope="module")
        async def var_fixture():
            with context_var_manager(VALUE):
                yield

        @pytest.fixture
        def check_var_fixture(var_fixture):
            assert _context_var.get() is VALUE

        @pytest.mark.asyncio(loop_scope="module")
        async def test_first(check_var_fixture):
            assert _context_var.get() is VALUE

        @pytest.mark.asyncio(loop_scope="module")
        async def test_second(check_var_fixture):
            assert _context_var.get() is VALUE
        """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(passed=3)


def test_var_set_again_by_fixture_on_shared_loop_propagates_again(
    pytester: Pytester,
):
    """
    A fixture on a module loop setting a var to the same object at each setup
    propagates it at each setup: to a sync fixture depending on it, and to a
    test on a fresh function loop.
    """
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(_prelude + dedent("""
        VALUE = object()

        @pytest_asyncio.fixture(loop_scope="module")
        async def var_fixture():
            _context_var.set(VALUE)

        @pytest.fixture
        def check_var_fixture(var_fixture):
            assert _context_var.get() is VALUE

        @pytest.mark.asyncio(loop_scope="module")
        async def test_first(check_var_fixture):
            assert _context_var.get() is VALUE

        @pytest.mark.asyncio(loop_scope="module")
        async def test_second(check_var_fixture):
            assert _context_var.get() is VALUE

        @pytest.mark.asyncio
        async def test_third(var_fixture):
            assert _context_var.get() is VALUE
        """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(passed=3)


def test_var_from_async_generator_reset_in_sync_when_teardown_fails(
    pytester: Pytester,
):
    """
    The copy of a var in the synchronous context is reset even when the
    fixture's teardown raises.
    """
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(_prelude + dedent("""
        @pytest.fixture
        def check_var_reset_fixture():
            yield
            with pytest.raises(LookupError):
                _context_var.get()
            print("var reset in the synchronous context")

        @pytest_asyncio.fixture
        async def var_fixture(check_var_reset_fixture):
            _context_var.set("value")
            yield
            raise ValueError("teardown error")

        @pytest.mark.asyncio
        async def test(var_fixture):
            assert _context_var.get() == "value"
        """))
    result = pytester.runpytest("--asyncio-mode=strict", "-s")
    result.assert_outcomes(passed=1, errors=1)
    result.stdout.fnmatch_lines(
        ["*var reset in the synchronous context", "*ValueError: teardown error*"]
    )


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="Task.get_context() requires Python 3.12"
)
def test_current_task_context_is_the_context_of_the_coroutine(pytester: Pytester):
    """
    Each fixture and test runs in a task of its own, so
    ``asyncio.current_task().get_context()`` is the coroutine's own context:
    it holds the value the coroutine set, and not one another coroutine set.
    """
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(_prelude + dedent("""
        import asyncio

        def check_context(value):
            assert asyncio.current_task().get_context()[_context_var] == value

        @pytest_asyncio.fixture
        async def var_fixture():
            with context_var_manager("fixture value"):
                check_context("fixture value")
                yield
                check_context("fixture value")

        @pytest.mark.asyncio
        async def test(var_fixture):
            _context_var.set("test value")
            check_context("test value")
        """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(passed=1)


def test_var_from_task_factory_seen_by_async_fixture_sync_dependent_and_test(
    pytester: Pytester,
):
    """
    Each fixture and test runs in a task the loop's task factory created, so a
    var the factory supplies to the task's context is seen by the async
    fixture, by a sync fixture depending on it and by the test.
    """
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makeconftest(dedent("""\
        import asyncio
        import contextvars

        trace = contextvars.ContextVar("trace")

        def traced_task_factory(loop, coro, **kwargs):
            context = contextvars.copy_context()
            context.run(trace.set, "instrumented")
            return context.run(asyncio.Task, coro, loop=loop, **kwargs)

        def traced_loop_factory():
            loop = asyncio.new_event_loop()
            loop.set_task_factory(traced_task_factory)
            return loop

        def pytest_asyncio_loop_factories(config, item):
            return {"traced": traced_loop_factory}
        """))
    pytester.makepyfile(dedent("""\
        import pytest
        import pytest_asyncio

        from conftest import trace

        @pytest_asyncio.fixture
        async def traced_fixture():
            assert trace.get() == "instrumented"

        @pytest.fixture
        def sync_dependent(traced_fixture):
            assert trace.get() == "instrumented"

        @pytest.mark.asyncio
        async def test(sync_dependent):
            assert trace.get() == "instrumented"
        """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(passed=1)
