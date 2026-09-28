"""Custom task factories that wrap or refuse the tasks of async fixtures and tests."""

from __future__ import annotations

from textwrap import dedent

from pytest import Pytester


def test_a_task_factory_wrapper_failing_after_the_test_raised_reports_both_errors(
    pytester: Pytester,
):
    """The wrapper's error is reported with the test's SystemExit as its context."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest
        import pytest_asyncio

        def failing_wrapper_factory(loop, coro, **kwargs):
            # Wrap only the next task, the test's.
            loop.set_task_factory(None)

            async def run():
                try:
                    return await coro
                finally:
                    raise RuntimeError("wrapper failed")

            return asyncio.Task(run(), loop=loop, **kwargs)

        @pytest_asyncio.fixture
        async def failing_wrapper():
            asyncio.get_running_loop().set_task_factory(failing_wrapper_factory)
            yield

        @pytest.mark.asyncio
        async def test_exits(failing_wrapper):
            raise SystemExit("original exit")
        """))

    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)

    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(
        [
            "E   *SystemExit: original exit",
            "During handling of the above exception, another exception occurred:",
            "E   *RuntimeError: wrapper failed",
        ]
    )


def test_a_refused_fixture_task_is_a_setup_error_of_every_test_using_the_fixture(
    pytester: Pytester,
):
    """Pytest caches the task factory's error as the fixture's setup error."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = module")
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest_asyncio

        def reject_task(loop, coro, **kwargs):
            coro.close()
            raise RuntimeError("task creation failed")

        @pytest_asyncio.fixture(scope="module", loop_scope="module")
        def loop():
            loop = asyncio.get_event_loop()
            loop.set_task_factory(reject_task)
            yield loop
            loop.set_task_factory(None)

        @pytest_asyncio.fixture(scope="module", loop_scope="module")
        async def returned(loop):
            return 42

        @pytest_asyncio.fixture(scope="module", loop_scope="module")
        async def yielded(loop):
            yield 42

        def test_first_returned(returned):
            pass

        def test_second_returned(returned):
            pass

        def test_first_yielded(yielded):
            pass

        def test_second_yielded(yielded):
            pass
        """))

    result = pytester.runpytest_subprocess(
        "--asyncio-mode=strict", "-W", "error", timeout=30
    )

    result.assert_outcomes(errors=4)
    result.stdout.fnmatch_lines(
        [
            "*ERROR at setup of test_first_returned*",
            "E*RuntimeError: task creation failed",
            "*ERROR at setup of test_second_returned*",
            "E*RuntimeError: task creation failed",
            "*ERROR at setup of test_first_yielded*",
            "E*RuntimeError: task creation failed",
            "*ERROR at setup of test_second_yielded*",
            "E*RuntimeError: task creation failed",
        ]
    )


def test_a_sync_fixture_can_use_its_loop_after_the_loop_refused_a_task(
    pytester: Pytester,
):
    """A refused test task does not close the loop that a sync fixture still uses."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = module")
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest
        import pytest_asyncio

        def reject_task(loop, coro, **kwargs):
            coro.close()
            raise RuntimeError("task creation failed")

        @pytest_asyncio.fixture(scope="module", loop_scope="module")
        def loop():
            loop = asyncio.get_event_loop()
            loop.set_task_factory(reject_task)
            yield loop
            loop.set_task_factory(None)
            loop.run_until_complete(asyncio.sleep(0))

        @pytest.mark.asyncio(loop_scope="module")
        async def test_refused(loop):
            pytest.fail("the loop should have refused this test's task")
        """))

    result = pytester.runpytest_subprocess(
        "--asyncio-mode=strict", "-W", "error", timeout=30
    )

    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*RuntimeError: task creation failed*"])
