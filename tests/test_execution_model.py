from __future__ import annotations

from textwrap import dedent

from pytest import Pytester


def test_pytestasyncio_function_is_removed() -> None:
    import pytest_asyncio.plugin as plugin

    assert not hasattr(plugin, "PytestAsyncioFunction")


def test_event_loop_policy_fixture_is_removed(pytester: Pytester) -> None:
    pytester.makepyfile(dedent("""\
        import pytest

        @pytest.mark.asyncio
        async def test_policy(event_loop_policy):
            pass
        """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(["*fixture 'event_loop_policy' not found*"])


def test_marker_on_parameter_set_is_honored_at_runtime(pytester: Pytester) -> None:
    pytester.makepyfile(dedent("""\
        import asyncio
        import pytest

        @pytest.mark.parametrize(
            "backend",
            [pytest.param("asyncio", marks=pytest.mark.asyncio), "plain"],
        )
        async def test_async(backend):
            assert asyncio.get_running_loop()
        """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(passed=1, failed=1)


def test_auto_mode_owns_plain_async_tests_and_fixtures(pytester: Pytester) -> None:
    pytester.makepyfile(dedent("""\
        import asyncio
        import pytest

        @pytest.fixture
        async def loop():
            return asyncio.get_running_loop()

        async def test_auto(loop):
            assert asyncio.get_running_loop() is loop
        """))
    result = pytester.runpytest("--asyncio-mode=auto")
    result.assert_outcomes(passed=1)


def test_strict_mode_rejects_plain_async_fixture(pytester: Pytester) -> None:
    pytester.makepyfile(dedent("""\
        import pytest

        @pytest.fixture
        async def resource():
            return object()

        @pytest.mark.asyncio
        async def test_resource(resource):
            pass
        """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(["*Async fixture 'resource'*strict mode*"])


def test_fixture_loop_scope_defaults_to_function(pytester: Pytester) -> None:
    pytester.makepyfile(dedent("""\
        import pytest
        import pytest_asyncio

        @pytest_asyncio.fixture(scope="module")
        async def resource():
            return object()

        @pytest.mark.asyncio
        async def test_resource(resource):
            pass
        """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(
        ["*fixture 'resource'*module*event loop scope is 'function'*"]
    )


def test_asyncio_marker_scope_alias_is_an_error(pytester: Pytester) -> None:
    pytester.makepyfile(dedent("""\
        import pytest

        @pytest.mark.asyncio(scope="module")
        async def test_alias():
            pass
        """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(["*scope*not supported*loop_scope*"])


def test_is_async_test_returns_bool_for_normal_function_item(
    pytester: Pytester,
) -> None:
    pytester.makeconftest(dedent("""\
        import pytest_asyncio

        def pytest_collection_finish(session):
            values = [pytest_asyncio.is_async_test(item) for item in session.items]
            assert values == [True, False]
            assert all(type(value) is bool for value in values)
        """))
    pytester.makepyfile(dedent("""\
        import pytest

        @pytest.mark.asyncio
        async def test_async():
            pass

        def test_sync():
            pass
        """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(passed=2)
