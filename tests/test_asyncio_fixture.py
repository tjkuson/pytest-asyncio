from __future__ import annotations

import asyncio
from textwrap import dedent

import pytest
import pytest_asyncio
from pytest import Pytester


@pytest_asyncio.fixture
async def fixture_bare():
    await asyncio.sleep(0)
    return 1


@pytest.mark.asyncio
async def test_bare_fixture(fixture_bare):
    await asyncio.sleep(0)
    assert fixture_bare == 1


@pytest_asyncio.fixture(name="new_fixture_name")
async def fixture_with_name(request):
    await asyncio.sleep(0)
    return request.fixturename


@pytest.mark.asyncio
async def test_fixture_with_name(new_fixture_name):
    await asyncio.sleep(0)
    assert new_fixture_name == "new_fixture_name"


@pytest_asyncio.fixture(params=[2, 4])
async def fixture_with_params(request):
    await asyncio.sleep(0)
    return request.param


@pytest.mark.asyncio
async def test_fixture_with_params(fixture_with_params):
    await asyncio.sleep(0)
    assert fixture_with_params % 2 == 0


@pytest.mark.parametrize("mode", ("auto", "strict"))
def test_sync_function_uses_async_fixture(pytester: Pytester, mode):
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
        import pytest_asyncio

        pytest_plugins = 'pytest_asyncio'

        @pytest_asyncio.fixture
        async def always_true():
            return True

        def test_sync_function_uses_async_fixture(always_true):
           assert always_true is True
        """))
    result = pytester.runpytest(f"--asyncio-mode={mode}")
    result.assert_outcomes(passed=1)


def test_sync_fixture_preserves_stop_iteration_for_xfail(pytester: Pytester):
    """Wrapping a synchronous fixture preserves the exception matched by xfail."""
    pytest.importorskip(
        "pluggy",
        minversion="1.6",
        reason="Older pluggy replaces StopIteration from hook wrappers (pluggy#544).",
    )
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
        import pytest
        import pytest_asyncio

        @pytest_asyncio.fixture
        def value():
            raise StopIteration("no value available")

        @pytest.mark.xfail(raises=StopIteration, strict=True)
        def test_missing_value(value):
            pass
        """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(xfailed=1)


@pytest.mark.parametrize(
    "statement", ["return", "yield"], ids=["coroutine", "generator"]
)
def test_rejected_fixture_task_preserves_its_creation_error(
    pytester: Pytester, statement: str
):
    """Task creation errors reach pytest without replacement errors or warnings."""
    pytest.importorskip(
        "pluggy",
        minversion="1.6",
        reason="Older pluggy replaces StopIteration from hook wrappers (pluggy#544).",
    )
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent(f"""\
        import asyncio

        import pytest
        import pytest_asyncio

        def reject_task(loop, coro, **kwargs):
            coro.close()
            raise StopIteration("task creation failed")

        @pytest_asyncio.fixture
        async def reject_new_tasks(request):
            loop = asyncio.get_running_loop()
            original_factory = loop.get_task_factory()
            request.addfinalizer(lambda: loop.set_task_factory(original_factory))
            loop.set_task_factory(reject_task)

        @pytest_asyncio.fixture
        async def value(reject_new_tasks):
            {statement} 42

        @pytest.mark.xfail(raises=StopIteration, strict=True)
        def test_value(value):
            pass
        """))
    result = pytester.runpytest_subprocess(
        "--asyncio-mode=strict", "-W", "error", timeout=30
    )
    result.assert_outcomes(xfailed=1)
