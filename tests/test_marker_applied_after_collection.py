"""
Tests for asyncio markers that are applied after collection.

The asyncio marker is usually applied to tests during collection, e.g. as a
decorator or via auto mode. Markers can also be applied to existing test items
after collection, e.g. in a pytest_collection_modifyitems hook. pytest-asyncio
classifies items by their marker and callable at run time, so these later markers
are honored. These tests cover that case (see #810).
"""

from __future__ import annotations

from textwrap import dedent

from pytest import Pytester

MARK_COROUTINES_CONFTEST = dedent("""\
    import inspect

    def pytest_collection_modifyitems(items):
        for item in items:
            obj = getattr(item, "obj", None)
            func = getattr(obj, "__func__", obj)
            if inspect.iscoroutinefunction(func):
                item.add_marker("asyncio")
    """)


def test_marker_applied_after_collection_runs_the_test(pytester: Pytester):
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makeconftest(MARK_COROUTINES_CONFTEST)
    pytester.makepyfile(dedent("""\
            import asyncio

            async def test_runs_in_an_event_loop():
                assert asyncio.get_running_loop()
            """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(passed=1)


def test_marker_applied_after_collection_supports_parametrized_tests(
    pytester: Pytester,
):
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makeconftest(MARK_COROUTINES_CONFTEST)
    pytester.makepyfile(dedent("""\
            import asyncio
            import pytest

            @pytest.mark.parametrize("value", [1, 2])
            async def test_parametrized(value):
                assert asyncio.get_running_loop()
                assert value in (1, 2)
            """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(passed=2)


def test_marker_applied_after_collection_provides_function_scoped_loops(
    pytester: Pytester,
):
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makeconftest(MARK_COROUTINES_CONFTEST)
    pytester.makepyfile(dedent("""\
            import asyncio

            loop: asyncio.AbstractEventLoop

            async def test_remember_loop():
                global loop
                loop = asyncio.get_running_loop()

            async def test_runs_in_a_different_loop():
                global loop
                assert asyncio.get_running_loop() is not loop
            """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(passed=2)


def test_marker_applied_after_collection_respects_loop_scope_kwarg(
    pytester: Pytester,
):
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
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

            loop: asyncio.AbstractEventLoop

            async def test_remember_loop():
                global loop
                loop = asyncio.get_running_loop()

            async def test_runs_in_same_loop():
                global loop
                assert asyncio.get_running_loop() is loop
            """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(passed=2)


def test_marker_applied_after_collection_respects_default_test_loop_scope_ini(
    pytester: Pytester,
):
    pytester.makeini(dedent("""\
        [pytest]
        asyncio_default_fixture_loop_scope = function
        asyncio_default_test_loop_scope = module
        """))
    pytester.makeconftest(MARK_COROUTINES_CONFTEST)
    pytester.makepyfile(dedent("""\
            import asyncio

            loop: asyncio.AbstractEventLoop

            async def test_remember_loop():
                global loop
                loop = asyncio.get_running_loop()

            async def test_runs_in_same_loop():
                global loop
                assert asyncio.get_running_loop() is loop
            """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(passed=2)


def test_marker_applied_after_collection_emits_warning_for_legacy_scope_kwarg(
    pytester: Pytester,
):
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makeconftest(dedent("""\
        import inspect
        import pytest

        def pytest_collection_modifyitems(items):
            for item in items:
                if inspect.iscoroutinefunction(getattr(item, "obj", None)):
                    item.add_marker(pytest.mark.asyncio(scope="module"))
        """))
    pytester.makepyfile(dedent("""\
            import asyncio

            async def test_runs_in_an_event_loop():
                assert asyncio.get_running_loop()
            """))
    result = pytester.runpytest("--asyncio-mode=strict", "-W default", "--assert=plain")
    result.assert_outcomes(passed=1, warnings=1)
    result.stdout.fnmatch_lines(
        ['*The "scope" keyword argument to the asyncio marker*']
    )


def test_marker_applied_after_collection_shares_loop_with_async_fixture(
    pytester: Pytester,
):
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makeconftest(MARK_COROUTINES_CONFTEST)
    pytester.makepyfile(dedent("""\
            import asyncio
            import pytest_asyncio

            @pytest_asyncio.fixture
            async def fixture_loop():
                return asyncio.get_running_loop()

            async def test_runs_in_fixture_loop(fixture_loop):
                assert asyncio.get_running_loop() is fixture_loop
            """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(passed=1)


def test_marker_applied_after_collection_supports_static_methods(
    pytester: Pytester,
):
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makeconftest(MARK_COROUTINES_CONFTEST)
    pytester.makepyfile(dedent("""\
            import asyncio

            class TestStaticMethod:
                @staticmethod
                async def test_static_method():
                    assert asyncio.get_running_loop()
            """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(passed=1)


def test_marker_applied_after_collection_supports_hypothesis_tests(
    pytester: Pytester,
):
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makeconftest(dedent("""\
        import inspect

        def pytest_collection_modifyitems(items):
            for item in items:
                obj = getattr(item, "obj", None)
                inner = getattr(getattr(obj, "hypothesis", None), "inner_test", obj)
                if inspect.iscoroutinefunction(inner):
                    item.add_marker("asyncio")
        """))
    pytester.makepyfile(dedent("""\
            import asyncio
            from hypothesis import given, strategies as st

            @given(st.integers())
            async def test_hypothesis(value):
                assert asyncio.get_running_loop()
            """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(passed=1)


def test_marker_applied_after_collection_to_async_generator_emits_warning(
    pytester: Pytester,
):
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makeconftest(dedent("""\
        import inspect

        def pytest_collection_modifyitems(items):
            for item in items:
                if inspect.isasyncgenfunction(getattr(item, "obj", None)):
                    item.add_marker("asyncio")
        """))
    pytester.makepyfile(dedent("""\
            async def test_async_generator():
                yield
            """))
    result = pytester.runpytest("--asyncio-mode=strict", "-W default", "--assert=plain")
    result.assert_outcomes(xfailed=1, warnings=1)
    result.stdout.fnmatch_lines(
        ["*Tests based on asynchronous generators are not supported*"]
    )


def test_marker_applied_after_collection_to_sync_function_emits_warning(
    pytester: Pytester,
):
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makeconftest(dedent("""\
        def pytest_collection_modifyitems(items):
            for item in items:
                item.add_marker("asyncio")
        """))
    pytester.makepyfile(dedent("""\
            def test_sync():
                pass
            """))
    result = pytester.runpytest("--asyncio-mode=strict", "-W default", "--assert=plain")
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(
        ["*is marked with '@pytest.mark.asyncio' but it is not an async function.*"]
    )


def test_marker_applied_after_collection_errors_when_loop_factories_are_configured(
    pytester: Pytester,
):
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makeconftest(MARK_COROUTINES_CONFTEST + dedent("""\

        import asyncio

        def pytest_asyncio_loop_factories(config, item):
            return {"custom": asyncio.new_event_loop}
        """))
    pytester.makepyfile(dedent("""\
            async def test_foo():
                pass
            """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(
        ["*was not visible when loop-factory parametrization ran*"]
    )


def test_marker_applied_after_collection_errors_when_marker_selects_loop_factories(
    pytester: Pytester,
):
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makeconftest(dedent("""\
        import inspect
        import pytest

        def pytest_collection_modifyitems(items):
            for item in items:
                if inspect.iscoroutinefunction(getattr(item, "obj", None)):
                    item.add_marker(pytest.mark.asyncio(loop_factories=["custom"]))
        """))
    pytester.makepyfile(dedent("""\
            async def test_foo():
                pass
            """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(
        ["*was not visible when loop-factory parametrization ran*"]
    )


def test_marker_applied_after_collection_is_recognized_by_is_async_test(
    pytester: Pytester,
):
    # is_async_test reports the item as managed once the marker has been applied,
    # even though the marker was added after collection.
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makeconftest(dedent("""\
        import inspect
        import pytest
        import pytest_asyncio

        def pytest_collection_modifyitems(items):
            for item in items:
                if inspect.iscoroutinefunction(getattr(item, "obj", None)):
                    item.add_marker("asyncio")

        @pytest.hookimpl(trylast=True)
        def pytest_collection_finish(session):
            managed = [i for i in session.items if pytest_asyncio.is_async_test(i)]
            assert len(managed) == 1
        """))
    pytester.makepyfile(dedent("""\
        async def test_runs_in_an_event_loop():
            pass
        """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(passed=1)


def test_marker_applied_after_collection_does_not_replace_the_test_item(
    pytester: Pytester,
):
    # Markers applied after collection must be honored without replacing the
    # test item because other plugins may hold references to the item or have
    # attached state to it, such as stashed values.
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makeconftest(dedent("""\
        import inspect
        import pytest

        stash_key = pytest.StashKey[str]()

        def pytest_collection_modifyitems(items):
            for item in items:
                if inspect.iscoroutinefunction(getattr(item, "obj", None)):
                    item.add_marker("asyncio")
                item.stash[stash_key] = "annotated"

        @pytest.hookimpl(wrapper=True)
        def pytest_runtest_call(item):
            assert item.stash[stash_key] == "annotated"
            return (yield)
        """))
    pytester.makepyfile(dedent("""\
            import asyncio

            async def test_runs_in_an_event_loop():
                assert asyncio.get_running_loop()
            """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(passed=1)
