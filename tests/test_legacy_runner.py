"""Legacy compatibility during the TaskGroup opt-in; remove with legacy in V2."""

from __future__ import annotations

import sys
from textwrap import dedent

import pytest
from pytest import MonkeyPatch, Pytester


@pytest.mark.parametrize(
    ("setting", "setup_finished", "same_task"),
    [
        pytest.param("", True, False, id="unset"),
        pytest.param(
            "experimental_asyncio_task_group_runner = false\n",
            True,
            False,
            id="disabled",
        ),
        pytest.param(
            "experimental_asyncio_task_group_runner = true\n",
            False,
            True,
            id="enabled",
        ),
    ],
)
def test_only_enabled_setting_keeps_fixture_task_alive_through_teardown(
    pytester: Pytester,
    monkeypatch: MonkeyPatch,
    setting: str,
    setup_finished: bool,
    same_task: bool,
):
    """Only the opt-in task stays pending until its fixture's teardown."""
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    pytester.makeini(
        "[pytest]\nasyncio_default_fixture_loop_scope = function\n" + setting
    )
    pytester.makepyfile(dedent(f"""\
        import asyncio

        import pytest_asyncio

        completed = []

        @pytest_asyncio.fixture
        async def resource():
            setup_task = asyncio.current_task()
            setup_task.add_done_callback(lambda _: completed.append(True))
            yield setup_task
            assert (asyncio.current_task() is setup_task) is {same_task}

        def test_fixture_task_lifetime(resource):
            assert resource.done() is {setup_finished}
            assert bool(completed) is {setup_finished}
        """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(passed=1)


def test_default_runner_keeps_test_return_values_and_errors_in_their_tasks(
    pytester: Pytester,
):
    """The default test task still exposes the coroutine's original outcome."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest

        tasks = {}
        error = ValueError("test failed")

        @pytest.mark.asyncio
        async def test_returns_value():
            tasks["returned"] = asyncio.current_task()
            return 42

        @pytest.mark.asyncio
        async def test_raises_error():
            tasks["raised"] = asyncio.current_task()
            raise error

        def test_tasks_keep_their_outcomes():
            assert tasks["returned"].result() == 42
            assert tasks["raised"].exception() is error
        """))
    result = pytester.runpytest(
        "--asyncio-mode=strict", "-o", "experimental_asyncio_task_group_runner=false"
    )
    result.assert_outcomes(passed=2, failed=1)


@pytest.mark.parametrize(
    ("setup", "teardown", "passed"),
    [
        pytest.param(
            'raise asyncio.CancelledError("fixture cancelled")',
            "pass",
            0,
            id="setup",
        ),
        pytest.param(
            "pass",
            'raise asyncio.CancelledError("fixture cancelled")',
            1,
            id="teardown",
        ),
    ],
)
def test_default_runner_preserves_fixture_cancelled_error(
    pytester: Pytester, setup: str, teardown: str, passed: int
):
    """An unhandled fixture cancellation keeps its original exception type."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent(f"""\
        import asyncio

        import pytest_asyncio

        @pytest_asyncio.fixture
        async def resource():
            {setup}
            yield
            {teardown}

        def test_resource(resource):
            pass
        """))
    result = pytester.runpytest(
        "--asyncio-mode=strict", "-o", "experimental_asyncio_task_group_runner=false"
    )
    result.assert_outcomes(errors=1, passed=passed)
    result.stdout.fnmatch_lines(["*CancelledError: fixture cancelled*"])
    assert "PytestAsyncioError" not in result.stdout.str()


@pytest.mark.parametrize(
    "statement", ["return", "yield"], ids=["coroutine", "generator"]
)
@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="Task factories receive an explicit context on Python 3.11+",
)
def test_default_runner_propagates_fixture_context_after_task_factory_changes_caller(
    pytester: Pytester, statement: str
):
    """A factory's synchronous assignment must not replace the fixture's context."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent(f"""\
        import asyncio
        from contextvars import ContextVar

        import pytest_asyncio

        value = ContextVar("value")

        @pytest_asyncio.fixture
        def task_factory():
            token = value.set("fixture")
            loop = asyncio.get_event_loop()

            def factory(loop, coro, **kwargs):
                value.set("factory")
                return asyncio.Task(coro, loop=loop, **kwargs)

            loop.set_task_factory(factory)
            yield
            loop.set_task_factory(None)
            value.reset(token)

        @pytest_asyncio.fixture
        async def async_value(task_factory):
            {statement} value.get()

        def test_sync_dependent_sees_fixture_context(async_value):
            assert async_value == "fixture"
            assert value.get() == async_value
        """))
    result = pytester.runpytest(
        "--asyncio-mode=strict", "-o", "experimental_asyncio_task_group_runner=false"
    )
    result.assert_outcomes(passed=1)
