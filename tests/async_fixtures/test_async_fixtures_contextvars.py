"""
Regression test for https://github.com/pytest-dev/pytest-asyncio/issues/127:
contextvars were not properly maintained among fixtures and tests.

Async fixtures and tests see the context variables of the synchronous code that
requested them; values set by an async fixture are copied back until its
teardown and stay in the context of the loop's task while the loop lives.
"""

from __future__ import annotations

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


def test_var_from_sync_fixture_set_after_loop_exists_propagates_to_async(
    pytester: Pytester,
):
    """An async test sees vars set after the task of its loop was created."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(_prelude + dedent("""
        @pytest_asyncio.fixture
        async def earlier_async_fixture():
            # Creates the loop and its task before var_fixture sets the var.
            with pytest.raises(LookupError):
                _context_var.get()

        @pytest.fixture
        def var_fixture(earlier_async_fixture):
            with context_var_manager("value"):
                yield

        @pytest.mark.asyncio
        async def test(var_fixture):
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


def test_var_from_fixture_persists_for_previous_fixture_cleanup(pytester: Pytester):
    """
    A coroutine fixture cannot reset a var it sets: the value stays in the
    context of the loop's task, where an earlier fixture's teardown still sees
    it, while its copy in the synchronous context is reset at teardown.
    """
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(_prelude + dedent("""
        @pytest_asyncio.fixture
        async def no_var_fixture():
            with pytest.raises(LookupError):
                _context_var.get()
            yield
            assert _context_var.get() == "value"

        @pytest.fixture
        def sync_fixture(no_var_fixture):
            yield
            with pytest.raises(LookupError):
                _context_var.get()

        @pytest_asyncio.fixture
        async def var_fixture(sync_fixture):
            _context_var.set("value")

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


def test_later_fixture_on_shared_loop_propagates_only_its_own_changes(
    pytester: Pytester,
):
    """
    A fixture copies into the synchronous context only the vars it set itself,
    not what earlier fixtures left in the context of a shared loop's task.
    """
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(_prelude + dedent("""
        _other_var = ContextVar("other_var")

        @pytest_asyncio.fixture(loop_scope="module")
        async def var_fixture():
            _context_var.set("value")

        @pytest.mark.asyncio(loop_scope="module")
        async def test_first(var_fixture):
            assert _context_var.get() == "value"

        @pytest_asyncio.fixture(loop_scope="module")
        async def other_var_fixture():
            assert _context_var.get() == "value"
            _other_var.set("other")

        @pytest.fixture
        def check_sync_context(other_var_fixture):
            assert _other_var.get() == "other"
            with pytest.raises(LookupError):
                _context_var.get()

        @pytest.mark.asyncio(loop_scope="module")
        async def test_second(check_sync_context):
            assert _context_var.get() == "value"
            assert _other_var.get() == "other"
        """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(passed=2)


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


def test_var_from_sync_test_propagates_to_async_on_shared_loop(pytester: Pytester):
    """
    A module loop's task outlives a sync test: later async tests see the vars
    the test set, and those a sync fixture sets only for as long as it lives.
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


@pytest.mark.parametrize(
    ("loop_scope", "value_seen_by_next_test"),
    [("function", None), ("module", "new_value")],
)
def test_context_changes_in_async_tests_last_as_long_as_the_loop(
    pytester: Pytester,
    loop_scope: Literal["function", "module"],
    value_seen_by_next_test: str | None,
):
    """
    A var set by an async test stays in the context of the loop's task: the next
    test on a module loop sees it, a function loop has a fresh context per test.
    """
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent(f"""
            import pytest
            import pytest_asyncio
            from contextvars import ContextVar

            _context_var = ContextVar("my_var")

            @pytest.mark.asyncio(loop_scope="{loop_scope}")
            async def test_async_first():
                _context_var.set("new_value")

            def test_sync():
                with pytest.raises(LookupError):
                    _context_var.get()

            @pytest.mark.asyncio(loop_scope="{loop_scope}")
            async def test_async_second():
                assert _context_var.get(None) == {value_seen_by_next_test!r}
            """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(passed=3)
