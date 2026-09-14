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
    sys.version_info < (3, 11), reason="pytest-asyncio's own task needs 3.11"
)


def test_a_custom_loop_remains_current_while_it_closes(pytester: Pytester):
    """Code run by the loop's close still finds it with asyncio.get_event_loop()."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makeconftest(dedent("""\
        import asyncio

        closed_while_current = []

        class Loop(asyncio.SelectorEventLoop):
            def close(self):
                if not self.is_closed():
                    try:
                        current = asyncio.get_event_loop()
                    except RuntimeError:
                        current = None
                    closed_while_current.append(current is self)
                super().close()

        def pytest_asyncio_loop_factories(config, item):
            return {"custom": Loop}
        """))
    pytester.makepyfile(dedent("""\
        import pytest
        from conftest import closed_while_current

        @pytest.mark.asyncio
        async def test_uses_the_loop():
            pass

        def test_the_loop_was_current_while_it_closed():
            assert closed_while_current == [True]
        """))
    result = pytester.runpytest("--asyncio-mode=strict", "-W", "error")
    result.assert_outcomes(passed=2)


@_REQUIRES_311
def test_a_loop_that_cannot_create_tasks_is_closed_and_the_error_reported(
    pytester: Pytester,
):
    """Every test of the scope errors with the loop's error; nothing else runs."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = module")
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
    assert "LOOPS CLOSED: [True]" in result.stdout.str()


def test_an_interruption_before_the_loop_served_anything_closes_it(
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
    assert "LOOPS CLOSED: [True]" in output
    for noise in ("Task was destroyed", "never awaited", "never retrieved"):
        assert noise not in output
