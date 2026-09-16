"""
The event loop of a scope: opened before its first fixture or test, current
until it has closed, and closed however the scope ends.
"""

from __future__ import annotations

import sys
from textwrap import dedent

import pytest
from pytest import Pytester

_REQUIRES_311 = pytest.mark.skipif(
    sys.version_info < (3, 11), reason="The experimental runner requires Python 3.11"
)


def test_a_custom_loop_remains_current_while_it_closes(pytester: Pytester):
    """Code run by the loop's close still finds it with asyncio.get_event_loop()."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makeconftest(dedent("""\
        import asyncio

        import pytest

        _closed_while_current = []

        @pytest.fixture
        def closed_while_current():
            return _closed_while_current

        class Loop(asyncio.SelectorEventLoop):
            def close(self):
                if not self.is_closed():
                    try:
                        current = asyncio.get_event_loop()
                    except RuntimeError:
                        current = None
                    _closed_while_current.append(current is self)
                super().close()

        def pytest_asyncio_loop_factories(config, item):
            return {"custom": Loop}
        """))
    pytester.makepyfile(dedent("""\
        import pytest

        @pytest.mark.asyncio
        async def test_uses_the_loop():
            pass

        def test_the_loop_was_current_while_it_closed(closed_while_current):
            assert closed_while_current == [True]
        """))
    result = pytester.runpytest("--asyncio-mode=strict", "-W", "error")
    result.assert_outcomes(passed=2)


@_REQUIRES_311
def test_a_loop_that_cannot_create_tasks_is_closed_and_the_error_reported(
    pytester: Pytester,
):
    """Every test of the scope errors with the loop's error; nothing else runs."""
    pytester.makeini(
        "[pytest]\n"
        "experimental_asyncio_task_group_runner = true\n"
        "asyncio_default_fixture_loop_scope = module"
    )
    pytester.makeconftest(dedent("""\
        import asyncio

        import pytest

        loops = []

        def task_factory(loop, coro, **kwargs):
            coro.close()
            raise RuntimeError("no tasks on this loop")

        def loop_factory():
            loop = asyncio.new_event_loop()
            loop.set_task_factory(task_factory)
            loops.append(loop)
            return loop

        def pytest_asyncio_loop_factories(config, item):
            return {"broken": loop_factory}

        @pytest.hookimpl(wrapper=True)
        def pytest_sessionfinish(session):
            yield
            print("\\nLOOPS CLOSED:", [loop.is_closed() for loop in loops])
        """))
    pytester.makepyfile(dedent("""\
        import pytest

        @pytest.mark.asyncio(loop_scope="module")
        async def test_first():
            pass

        @pytest.mark.asyncio(loop_scope="module")
        async def test_second():
            pass
        """))
    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)
    result.assert_outcomes(errors=2)
    result.stdout.fnmatch_lines(["*RuntimeError: no tasks on this loop*"])
    assert "LOOPS CLOSED: [True]" in result.stdout.lines


def test_an_interruption_before_the_first_async_test_still_closes_the_loop(
    pytester: Pytester,
):
    """A callback the loop factory scheduled raises: the loop closes cleanly."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = module")
    pytester.makeconftest(dedent("""\
        import asyncio

        import pytest

        loops = []

        def interrupt():
            raise KeyboardInterrupt

        def loop_factory():
            loop = asyncio.new_event_loop()
            loop.call_soon(interrupt)
            loops.append(loop)
            return loop

        def pytest_asyncio_loop_factories(config, item):
            return {"interrupted": loop_factory}

        @pytest.hookimpl(wrapper=True)
        def pytest_sessionfinish(session):
            yield
            print("\\nLOOPS CLOSED:", [loop.is_closed() for loop in loops])
        """))
    pytester.makepyfile(dedent("""\
        import pytest

        @pytest.mark.asyncio(loop_scope="module")
        async def test_never_runs():
            pass
        """))
    result = pytester.runpytest_subprocess(
        "--asyncio-mode=strict", "-W", "error", timeout=30
    )
    assert result.ret == pytest.ExitCode.INTERRUPTED
    output = result.stdout.str() + result.stderr.str()
    assert "LOOPS CLOSED: [True]" in result.stdout.lines
    for noise in ("Task was destroyed", "never awaited", "never retrieved"):
        assert noise not in output


@_REQUIRES_311
@pytest.mark.parametrize("task_factory", ["reject_task", "create_one_task"])
def test_sync_fixture_can_use_its_loop_during_teardown_after_task_startup_fails(
    pytester: Pytester,
    task_factory: str,
):
    """Failed task startup does not close a loop still used by a fixture."""
    pytester.makeini(
        "[pytest]\n"
        "experimental_asyncio_task_group_runner = true\n"
        "asyncio_default_fixture_loop_scope = module"
    )
    pytester.makepyfile(dedent(f"""\
        import asyncio

        import pytest
        import pytest_asyncio

        def reject_task(loop, coro, **kwargs):
            coro.close()
            raise RuntimeError("task creation failed")

        def create_one_task(loop, coro, **kwargs):
            loop.set_task_factory(reject_task)
            return asyncio.Task(coro, loop=loop, **kwargs)

        @pytest_asyncio.fixture(scope="module", loop_scope="module")
        def event_loop():
            loop = asyncio.get_event_loop()
            loop.set_task_factory({task_factory})
            yield loop
            assert not loop.is_closed()
            loop.set_task_factory(None)
            loop.run_until_complete(asyncio.sleep(0))

        def test_sync_fixture_can_use_its_loop(event_loop):
            assert not event_loop.is_closed()

        @pytest.mark.asyncio(loop_scope="module")
        async def test_async_startup_fails(event_loop):
            pytest.fail("test must not run when task creation fails")
        """))
    result = pytester.runpytest_subprocess(
        "--asyncio-mode=strict", "-W", "error", timeout=30
    )
    result.assert_outcomes(passed=1, errors=1)
    result.stdout.fnmatch_lines(["*RuntimeError: task creation failed*"])


@_REQUIRES_311
@pytest.mark.parametrize(
    "statement", ["return", "yield"], ids=["coroutine", "generator"]
)
def test_task_startup_error_is_reported_for_every_consumer_of_a_shared_fixture(
    pytester: Pytester, statement: str
):
    """Consumers of a shared fixture see its original startup failure."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = module")
    pytester.makepyfile(dedent(f"""\
        import asyncio

        import pytest_asyncio

        def reject_task(loop, coro, **kwargs):
            coro.close()
            raise RuntimeError("task creation failed")

        @pytest_asyncio.fixture(scope="module", loop_scope="module")
        def event_loop():
            loop = asyncio.get_event_loop()
            loop.set_task_factory(reject_task)
            yield loop
            loop.set_task_factory(None)

        @pytest_asyncio.fixture(scope="module", loop_scope="module")
        async def value(event_loop):
            {statement} 42

        def test_first_consumer(value):
            pass

        def test_second_consumer(value):
            pass
        """))
    result = pytester.runpytest_subprocess(
        "--asyncio-mode=strict", "-W", "error", timeout=30
    )
    result.assert_outcomes(errors=2)
    result.stdout.fnmatch_lines(
        [
            "*ERROR at setup of test_first_consumer*",
            "E*RuntimeError: task creation failed",
            "*ERROR at setup of test_second_consumer*",
            "E*RuntimeError: task creation failed",
        ]
    )


def test_a_loop_factory_failure_is_cached_for_all_fixture_consumers(pytester: Pytester):
    """A failed runner acquisition does not run or retry the dependent fixture."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = module")
    pytester.makeconftest(dedent("""\
        def unavailable_loop():
            with open("factory-calls.txt", "a") as calls:
                calls.write("called\\n")
            raise LookupError("the loop is unavailable")

        def pytest_asyncio_loop_factories(config, item):
            return {"unavailable": unavailable_loop}
        """))
    pytester.makepyfile(dedent("""\
        from pathlib import Path

        import pytest
        import pytest_asyncio

        @pytest_asyncio.fixture(scope="module", loop_scope="module")
        async def value():
            Path("fixture-ran.txt").touch()
            return 42

        @pytest.mark.asyncio(loop_scope="module")
        async def test_first(value):
            pass

        @pytest.mark.asyncio(loop_scope="module")
        async def test_second(value):
            pass
        """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(errors=2)
    result.stdout.fnmatch_lines(
        [
            "*ERROR at setup of test_first*",
            "*LookupError: the loop is unavailable*",
            "*ERROR at setup of test_second*",
            "*LookupError: the loop is unavailable*",
        ]
    )
    assert (pytester.path / "factory-calls.txt").read_text() == "called\n"
    assert not (pytester.path / "fixture-ran.txt").exists()


@_REQUIRES_311
def test_a_second_interruption_during_shutdown_still_closes_the_loop(
    pytester: Pytester,
):
    """The loop closes even after a second interruption during shutdown."""
    pytester.makeini(
        "[pytest]\n"
        "experimental_asyncio_task_group_runner = true\n"
        "asyncio_default_fixture_loop_scope = function"
    )
    pytester.makeconftest(dedent("""\
        import asyncio
        from pathlib import Path

        loops = []

        def loop_factory():
            loop = asyncio.new_event_loop()
            loops.append(loop)
            return loop

        def pytest_asyncio_loop_factories(config, item):
            return {"tracked": loop_factory}

        def pytest_unconfigure(config):
            Path("loop-closed.txt").write_text(
                str([loop.is_closed() for loop in loops])
            )
        """))
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest

        def interrupt():
            raise KeyboardInterrupt

        @pytest.mark.asyncio
        async def test_interrupted(request):
            loop = asyncio.get_running_loop()
            request.addfinalizer(lambda: loop.call_soon(interrupt))
            raise KeyboardInterrupt
        """))
    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)
    assert result.ret != pytest.ExitCode.OK
    assert "KeyboardInterrupt" in result.stdout.str() + result.stderr.str()
    assert (pytester.path / "loop-closed.txt").read_text() == "[True]"
