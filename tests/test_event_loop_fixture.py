from __future__ import annotations

from textwrap import dedent

from pytest import Pytester


def test_event_loop_fixture_handles_unclosed_async_gen(
    pytester: Pytester,
):
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
            import asyncio
            import pytest

            pytest_plugins = 'pytest_asyncio'

            @pytest.mark.asyncio
            async def test_something():
                async def generator_fn():
                    yield
                    yield

                gen = generator_fn()
                await gen.__anext__()
            """))
    result = pytester.runpytest("--asyncio-mode=strict", "-W", "default")
    result.assert_outcomes(passed=1, warnings=0)


def test_closing_plugin_owned_event_loop_is_an_error(
    pytester: Pytester,
):
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
            import asyncio
            import pytest
            import pytest_asyncio
            pytest_plugins = 'pytest_asyncio'

            @pytest_asyncio.fixture
            async def _event_loop():
                return asyncio.get_running_loop()

            @pytest.fixture
            def close_event_loop(_event_loop):
                yield
                # fixture has its own cleanup code
                _event_loop.close()

            @pytest.mark.asyncio
            async def test_something(close_event_loop):
                await asyncio.sleep(0.01)
            """))
    result = pytester.runpytest_subprocess("--asyncio-mode=strict", "--assert=plain")
    result.assert_outcomes(passed=1, errors=1)
    result.stdout.fnmatch_lines(
        [
            "*pytest-asyncio's event loop was closed before pytest-asyncio "
            "could clean it up*"
        ]
    )


def test_event_loop_fixture_asyncgen_error(
    pytester: Pytester,
):
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
            import asyncio
            import pytest

            pytest_plugins = 'pytest_asyncio'

            @pytest.mark.asyncio
            async def test_something():
                # mock shutdown_asyncgen failure
                loop = asyncio.get_running_loop()
                async def fail():
                    raise RuntimeError("mock error cleaning up...")
                loop.shutdown_asyncgens = fail
            """))
    result = pytester.runpytest("--asyncio-mode=strict", "-W", "default")
    result.assert_outcomes(passed=1, errors=1)
