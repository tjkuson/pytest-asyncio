from __future__ import annotations

from textwrap import dedent

from pytest import Pytester

MARK_COROUTINES = dedent("""\
    import inspect

    def pytest_collection_modifyitems(items):
        for item in items:
            if inspect.iscoroutinefunction(getattr(item, "obj", None)):
                item.add_marker("asyncio")
    """)


def test_marker_applied_after_collection_runs_test(pytester: Pytester) -> None:
    pytester.makeconftest(MARK_COROUTINES)
    pytester.makepyfile(dedent("""\
        import asyncio

        async def test_late_marker():
            assert asyncio.get_running_loop()
        """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(passed=1)


def test_late_marker_loop_scope_is_honored(pytester: Pytester) -> None:
    pytester.makeconftest(dedent("""\
        import inspect
        import pytest

        def pytest_collection_modifyitems(items):
            for item in items:
                if inspect.iscoroutinefunction(getattr(item, "obj", None)):
                    item.add_marker(pytest.mark.asyncio(loop_scope="module"))
        """))
    pytester.makepyfile(dedent("""\
        import asyncio

        loop = None

        async def test_first():
            global loop
            loop = asyncio.get_running_loop()

        async def test_second():
            assert asyncio.get_running_loop() is loop
        """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(passed=2)


def test_late_marker_with_configured_factories_has_guided_error(
    pytester: Pytester,
) -> None:
    pytester.makeconftest(MARK_COROUTINES + dedent("""\

        import asyncio

        def pytest_asyncio_loop_factories(config, item):
            return {"default": asyncio.new_event_loop}
        """))
    pytester.makepyfile("async def test_late_marker():\n    pass\n")
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(
        ["*was not visible when loop-factory parametrization ran*"]
    )


def test_late_marker_is_reflected_by_is_async_test(pytester: Pytester) -> None:
    pytester.makeconftest(MARK_COROUTINES + dedent("""\

        import pytest_asyncio

        def pytest_collection_finish(session):
            assert pytest_asyncio.is_async_test(session.items[0]) is True
        """))
    pytester.makepyfile("async def test_late_marker():\n    pass\n")
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(passed=1)
