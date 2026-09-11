"""
Behaviour of the default runner that asyncio_experimental_task_per_fixture
changes. Each test turns the option off, so it runs the default runner in
every test environment. Delete this module together with the default runner.
"""

from __future__ import annotations

from textwrap import dedent

import pytest
from pytest import Pytester

_CANCELLED_IN_SETUP = """\
import asyncio

import pytest_asyncio

@pytest_asyncio.fixture
async def resource():
    raise asyncio.CancelledError("fixture cancelled")
    yield
"""

_CANCELLED_IN_TEARDOWN = """\
import asyncio

import pytest_asyncio

@pytest_asyncio.fixture
async def resource():
    yield
    raise asyncio.CancelledError("fixture cancelled")
"""


@pytest.mark.parametrize(
    ("conftest", "passed"),
    [
        pytest.param(_CANCELLED_IN_SETUP, 0, id="setup"),
        pytest.param(_CANCELLED_IN_TEARDOWN, 1, id="teardown"),
    ],
)
def test_default_runner_reports_the_cancelled_error_of_a_fixture_unchanged(
    pytester: Pytester, conftest: str, passed: int
):
    """Characterisation: a fixture's CancelledError reaches pytest unchanged."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makeconftest(conftest)
    pytester.makepyfile(dedent("""\
        def test_uses_resource(resource):
            pass
        """))

    result = pytester.runpytest(
        "--asyncio-mode=strict", "-o", "asyncio_experimental_task_per_fixture=false"
    )

    result.assert_outcomes(errors=1, passed=passed)
    result.stdout.fnmatch_lines(["*CancelledError: fixture cancelled*"])
    result.stdout.no_fnmatch_line("*PytestAsyncioError*")


def test_default_runner_passes_an_async_fixture_that_cancels_every_other_task(
    pytester: Pytester,
):
    """The default runner has no task of its own for the teardown to cancel."""
    pytester.makeini(
        "[pytest]\n"
        "asyncio_default_fixture_loop_scope = module\n"
        "asyncio_default_test_loop_scope = module"
    )
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest_asyncio

        @pytest_asyncio.fixture(autouse=True)
        async def cancel_leftover_tasks():
            yield
            for task in asyncio.all_tasks():
                if task is not asyncio.current_task():
                    task.cancel()

        async def test_leaves_a_task():
            asyncio.get_running_loop().create_task(asyncio.sleep(100))
        """))

    result = pytester.runpytest_subprocess(
        "--asyncio-mode=auto",
        "-o",
        "asyncio_experimental_task_per_fixture=false",
        timeout=30,
    )

    result.assert_outcomes(passed=1)
