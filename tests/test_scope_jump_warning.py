from __future__ import annotations

from textwrap import dedent

from pytest import Pytester


def test_warns_when_test_and_fixture_loop_scopes_differ(pytester: Pytester) -> None:
    pytester.makepyfile(dedent("""\
        import pytest
        import pytest_asyncio

        @pytest_asyncio.fixture
        async def resource():
            return object()

        @pytest.mark.asyncio(loop_scope="module")
        async def test_resource(resource):
            pass
        """))
    result = pytester.runpytest("--asyncio-mode=strict", "-W", "default")
    result.assert_outcomes(passed=1, warnings=1)
    result.stdout.fnmatch_lines(
        ["*PytestAsyncioWarning: test 'test_resource' uses*module*resource*function*"]
    )


def test_warns_once_for_fixture_edge_through_sync_fixture(pytester: Pytester) -> None:
    pytester.makepyfile(dedent("""\
        import pytest
        import pytest_asyncio

        @pytest_asyncio.fixture
        async def inner():
            return object()

        @pytest.fixture
        def intermediary(inner):
            return inner

        @pytest_asyncio.fixture(loop_scope="module")
        async def outer(intermediary):
            return intermediary

        @pytest.mark.asyncio(loop_scope="module")
        async def test_one(outer):
            pass

        @pytest.mark.asyncio(loop_scope="module")
        async def test_two(outer):
            pass
        """))
    result = pytester.runpytest("--asyncio-mode=strict", "-W", "default")
    result.assert_outcomes(passed=2, warnings=1)
    result.stdout.fnmatch_lines(
        ["*PytestAsyncioWarning: fixture 'outer' uses*module*inner*function*"]
    )


def test_no_warning_when_loop_scopes_match(pytester: Pytester) -> None:
    pytester.makepyfile(dedent("""\
        import pytest
        import pytest_asyncio

        @pytest_asyncio.fixture(loop_scope="module")
        async def resource():
            return object()

        @pytest.mark.asyncio(loop_scope="module")
        async def test_resource(resource):
            pass
        """))
    result = pytester.runpytest("--asyncio-mode=strict", "-W", "default")
    result.assert_outcomes(passed=1, warnings=0)


def test_warns_once_for_each_test_edge_even_when_fixture_is_cached(
    pytester: Pytester,
) -> None:
    pytester.makepyfile(dedent("""\
        import pytest
        import pytest_asyncio

        @pytest_asyncio.fixture(scope="module", loop_scope="module")
        async def resource():
            return object()

        @pytest.mark.asyncio
        async def test_one(resource):
            pass

        @pytest.mark.asyncio
        async def test_two(resource):
            pass
        """))
    result = pytester.runpytest("--asyncio-mode=strict", "-W", "default")
    result.assert_outcomes(passed=2, warnings=2)


def test_warning_type_is_public() -> None:
    from pytest_asyncio import PytestAsyncioWarning

    assert issubclass(PytestAsyncioWarning, Warning)


def test_warning_follows_fixture_override_chain(pytester: Pytester) -> None:
    pytester.makeconftest(dedent("""\
        import pytest_asyncio

        @pytest_asyncio.fixture(scope="session", loop_scope="session")
        async def resource():
            return object()
        """))
    pytester.makepyfile(dedent("""\
        import pytest
        import pytest_asyncio

        @pytest_asyncio.fixture
        async def resource(resource):
            return resource

        @pytest.mark.asyncio
        async def test_resource(resource):
            pass
        """))
    result = pytester.runpytest("--asyncio-mode=strict", "-W", "default")
    result.assert_outcomes(passed=1, warnings=1)
    result.stdout.fnmatch_lines(
        ["*PytestAsyncioWarning: fixture 'resource' uses*function*resource*session*"]
    )
