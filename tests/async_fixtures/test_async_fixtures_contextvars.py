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


def test_a_sync_fixture_assignment_is_seen_and_restored_on_a_reused_loop(
    pytester: Pytester,
):
    """The fixture may run before the shared loop is created, or after."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""
        from contextvars import ContextVar

        import pytest

        _context_var = ContextVar("context_var")

        @pytest.fixture
        def var_fixture():
            token = _context_var.set("value")
            try:
                yield
            finally:
                _context_var.reset(token)

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


def test_a_sync_test_assignment_is_seen_by_later_async_tests_on_a_shared_loop(
    pytester: Pytester,
):
    """Each test's task starts from the current context, however old its loop."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""
        from contextvars import ContextVar

        import pytest

        _context_var = ContextVar("context_var")

        @pytest.mark.asyncio(loop_scope="module")
        async def test_async_before():
            with pytest.raises(LookupError):
                _context_var.get()

        def test_sync():
            _context_var.set("value")

        @pytest.mark.asyncio(loop_scope="module")
        async def test_async_after():
            assert _context_var.get() == "value"
        """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(passed=3)


def test_a_test_assignment_does_not_change_its_fixtures_teardown_context(
    pytester: Pytester,
):
    """The fixture still sees its own value, and resets it in its own context."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""
        from contextvars import ContextVar

        import pytest
        import pytest_asyncio

        _context_var = ContextVar("context_var")

        @pytest_asyncio.fixture
        async def var_fixture():
            token = _context_var.set("value")
            try:
                yield
                assert _context_var.get() == "value"
            finally:
                _context_var.reset(token)

        @pytest.mark.asyncio
        async def test(var_fixture):
            assert _context_var.get() == "value"
            _context_var.set("other")
        """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(passed=1)


def test_an_async_test_assignment_is_not_seen_by_a_fixture_set_up_later(
    pytester: Pytester,
):
    """A test's assignment stays out of the next fixture's starting context."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""
        from contextvars import ContextVar

        import pytest
        import pytest_asyncio

        _context_var = ContextVar("context_var")

        @pytest.mark.asyncio(loop_scope="module")
        async def test_set_var():
            _context_var.set("test value")

        @pytest_asyncio.fixture(loop_scope="module")
        async def var_fixture():
            with pytest.raises(LookupError):
                _context_var.get()
            yield

        @pytest.mark.asyncio(loop_scope="module")
        async def test_uses_var(var_fixture):
            pass
        """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(passed=2)


def test_a_fixture_assigning_the_same_object_again_propagates_it_again(
    pytester: Pytester,
):
    """Each setup of the fixture on a shared loop reaches its sync dependent."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""
        from contextvars import ContextVar

        import pytest
        import pytest_asyncio

        _context_var = ContextVar("context_var")
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
        """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(passed=2)


def test_an_async_fixture_context_is_restored_in_sync_code_when_its_teardown_fails(
    pytester: Pytester,
):
    """A teardown error does not prevent restoring the synchronous context."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""
        from contextvars import ContextVar

        import pytest
        import pytest_asyncio

        _context_var = ContextVar("context_var")

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
def test_the_current_task_context_is_the_context_of_the_fixture_or_test(
    pytester: Pytester,
):
    """Task.get_context() exposes the context the running fixture or test uses."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""
        import asyncio
        from contextvars import ContextVar

        import pytest
        import pytest_asyncio

        _context_var = ContextVar("context_var")

        @pytest_asyncio.fixture
        async def var_fixture():
            _context_var.set("fixture value")
            context = asyncio.current_task().get_context()
            assert context[_context_var] == "fixture value"
            yield
            context = asyncio.current_task().get_context()
            assert context[_context_var] == "fixture value"

        @pytest.mark.asyncio
        async def test(var_fixture):
            _context_var.set("test value")
            context = asyncio.current_task().get_context()
            assert context[_context_var] == "test value"
        """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(passed=1)


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="The experimental runner requires Python 3.11"
)
def test_a_task_factory_context_is_seen_by_the_fixture_its_sync_dependent_and_the_test(
    pytester: Pytester,
):
    """The task factory's context value propagates through fixtures to the test."""
    pytester.makeini(
        "[pytest]\n"
        "experimental_asyncio_task_group_runner = true\n"
        "asyncio_default_fixture_loop_scope = function"
    )
    pytester.makeconftest(dedent("""\
        import asyncio
        import contextvars

        import pytest

        _trace = contextvars.ContextVar("trace")

        @pytest.fixture
        def trace():
            return _trace

        def traced_task_factory(loop, coro, *, context=None):
            calling_context = contextvars.copy_context()
            calling_context.run(_trace.set, "instrumented")
            return calling_context.run(asyncio.Task, coro, loop=loop, context=context)

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

        @pytest_asyncio.fixture
        async def traced_fixture(trace):
            assert trace.get() == "instrumented"

        @pytest.fixture
        def sync_dependent(traced_fixture, trace):
            assert trace.get() == "instrumented"

        @pytest.mark.asyncio
        async def test(sync_dependent, trace):
            assert trace.get() == "instrumented"
        """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(passed=1)


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="The experimental runner requires Python 3.11",
)
def test_a_task_factory_assignment_propagates_from_the_async_fixture_to_sync_code(
    pytester: Pytester,
):
    """The experimental runner propagates the factory's inherited context."""
    pytester.makeini(
        "[pytest]\n"
        "experimental_asyncio_task_group_runner = true\n"
        "asyncio_default_fixture_loop_scope = function"
    )
    pytester.makepyfile(dedent("""
        import asyncio
        from contextvars import ContextVar

        import pytest_asyncio

        value = ContextVar("value")

        @pytest_asyncio.fixture
        async def instrument():
            token = value.set("fixture")
            loop = asyncio.get_running_loop()
            original_factory = loop.get_task_factory()

            def task_factory(loop, coro, context=None, **kwargs):
                # Change inherited contexts, leaving explicitly supplied ones alone.
                if context is None:
                    value.set("factory")
                return asyncio.Task(coro, loop=loop, context=context, **kwargs)

            loop.set_task_factory(task_factory)
            try:
                yield
            finally:
                loop.set_task_factory(original_factory)
                value.reset(token)

        @pytest_asyncio.fixture
        async def resource(instrument):
            assert value.get() == "factory"
            return value.get()

        def test_sync_consumer(resource):
            assert resource == "factory"
            assert value.get() == "factory"
    """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(passed=1)


@pytest.mark.parametrize(
    "fixture_exit", ["return", "yield"], ids=["coroutine", "generator"]
)
def test_sync_dependents_see_fixture_values_after_task_factory_context_changes(
    pytester: Pytester, fixture_exit: Literal["return", "yield"]
):
    """Factory changes outside the fixture's task do not replace its setup values."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent(f"""
        import asyncio
        from contextvars import ContextVar

        import pytest_asyncio

        value = ContextVar("value")

        @pytest_asyncio.fixture
        async def configured_task_factory():
            value.set("fixture")
            loop = asyncio.get_running_loop()
            original_factory = loop.get_task_factory()

            def task_factory(loop, coro, **kwargs):
                task = asyncio.Task(coro, loop=loop, **kwargs)
                value.set("factory")
                return task

            loop.set_task_factory(task_factory)
            try:
                yield
            finally:
                loop.set_task_factory(original_factory)

        @pytest_asyncio.fixture
        async def resource(configured_task_factory):
            assert value.get() == "fixture"
            {fixture_exit} value.get()

        def test_sync_consumer(resource):
            assert resource == "fixture"
            assert value.get() == "fixture"
        """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(passed=1)
