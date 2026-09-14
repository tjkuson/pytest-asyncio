"""
Task groups, cancel scopes and timeouts spanning the yield of an async fixture.

An async generator fixture's setup and teardown run in one task, so a scope
entered before the yield is exited by the task that entered it (issues #1083
and #1191). These tests specify how failures in such a scope are reported.
"""

from __future__ import annotations

import importlib.util
import sys
from textwrap import dedent

import pytest
from pytest import Pytester

_REQUIRES_311 = pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="asyncio.TaskGroup and asyncio.timeout need Python 3.11",
)
_REQUIRES_312 = pytest.mark.skipif(
    sys.version_info < (3, 12), reason="asyncio.eager_task_factory needs Python 3.12"
)
_REQUIRES_UVLOOP = pytest.mark.skipif(
    importlib.util.find_spec("uvloop") is None, reason="uvloop is not installed"
)

# The same fixture with either library's task group: the statement entering
# the group, and the one starting its child ``fail``.
_TASK_GROUPS = [
    pytest.param(
        "async with asyncio.TaskGroup() as group:",
        "group.create_task(fail())",
        id="asyncio",
        marks=_REQUIRES_311,
    ),
    pytest.param(
        "async with anyio.create_task_group() as group:",
        "group.start_soon(fail)",
        id="anyio",
    ),
]
# A timeout without a deadline, and the statement making it expire now.
_TIMEOUTS = [
    pytest.param(
        "async with asyncio.timeout(None) as deadline:",
        "deadline.reschedule(asyncio.get_running_loop().time())",
        id="asyncio",
        marks=_REQUIRES_311,
    ),
    pytest.param(
        "with anyio.fail_after(None) as deadline:",
        "deadline.deadline = anyio.current_time()",
        id="anyio",
    ),
]
_REFUSED = "*RuntimeError: This event loop no longer accepts new tests*"

# The fixtures and tests of an example append to ``events``, and the outer
# test reads the literal sequence of what ran. The conftest writes the file
# after pytest's own session-finish hook, which is where an interrupted run
# still tears its fixtures down.
_EVENTS_CONFTEST = """\
    from pathlib import Path

    import pytest

    events = []

    @pytest.hookimpl(wrapper=True)
    def pytest_sessionfinish(session):
        yield
        Path(session.config.rootpath, "events.txt").write_text("\\n".join(events))
    """


def _read_events(pytester: Pytester) -> list[str]:
    return (pytester.path / "events.txt").read_text().splitlines()


def _fixture_cancelled(phase: str) -> list[str]:
    """The error of a cancelled fixture phase, printed after its cause."""
    return [
        "*asyncio.exceptions.CancelledError*",
        "*direct cause of the following*",
        f"*PytestAsyncioError: The {phase} of the async fixture was cancelled.*",
    ]


def test_a_generator_fixture_is_set_up_and_torn_down_in_one_task(pytester: Pytester):
    """So a scope entered before the yield is exited by the task that entered it."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = module")
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest
        import pytest_asyncio

        tasks = {}

        @pytest_asyncio.fixture
        async def generator():
            tasks["generator setup"] = asyncio.current_task()
            yield
            tasks["generator teardown"] = asyncio.current_task()

        @pytest_asyncio.fixture
        async def coroutine(generator):
            tasks["coroutine"] = asyncio.current_task()

        @pytest.mark.asyncio(loop_scope="module")
        async def test_uses_the_fixtures(coroutine):
            tasks["test"] = asyncio.current_task()

        def test_tasks():
            assert tasks["generator setup"] is tasks["generator teardown"]
            own = {tasks["generator setup"], tasks["coroutine"], tasks["test"]}
            assert len(own) == 3
        """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(passed=2)


def test_a_failing_test_ends_its_task_without_the_error(pytester: Pytester):
    """The failure goes to pytest; a callback on the test's task sees none."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest

        seen = {}

        @pytest.mark.asyncio
        async def test_fails():
            task = asyncio.current_task()
            task.add_done_callback(lambda task: seen.update(exception=task.exception()))
            raise AssertionError("reported to pytest")

        def test_the_task_ended_without_the_error():
            assert seen == {"exception": None}
        """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(failed=1, passed=1)
    result.stdout.fnmatch_lines(["*AssertionError: reported to pytest*"])


def test_a_cancellation_at_the_end_of_a_fixture_teardown_is_a_teardown_error(
    pytester: Pytester,
):
    """A fixture that cancels its own task as it returns is reported as cancelled."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest
        import pytest_asyncio

        @pytest_asyncio.fixture
        async def resource():
            yield "ready"
            asyncio.current_task().cancel("cancelled at the end of the teardown")

        @pytest.mark.asyncio
        async def test_uses_resource(resource):
            assert resource == "ready"
        """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(passed=1, errors=1)
    result.stdout.fnmatch_lines(
        ["*ERROR at teardown of test_uses_resource*", "*CancelledError*"]
    )


@_REQUIRES_311
def test_a_child_failing_during_the_fixture_setup_is_a_setup_error(
    pytester: Pytester,
):
    """The loop goes on: the later test runs."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = module")
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest
        import pytest_asyncio

        @pytest_asyncio.fixture
        async def service():
            async def fail():
                raise RuntimeError("background service failed")

            async with asyncio.TaskGroup() as group:
                group.create_task(fail())
                await asyncio.Event().wait()
                yield

        @pytest.mark.asyncio(loop_scope="module")
        async def test_service(service):
            pass

        @pytest.mark.asyncio(loop_scope="module")
        async def test_after():
            pass
        """))
    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)
    result.assert_outcomes(errors=1, passed=1)
    result.stdout.fnmatch_lines(
        [
            "*ERROR at setup of test_service*",
            "*RuntimeError: background service failed*",
        ]
    )


@_REQUIRES_311
def test_a_child_failing_while_a_dependent_fixture_sets_up_cancels_that_setup(
    pytester: Pytester,
):
    """The group's error is reported at the fixture's teardown; the rest is refused."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = module")
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest
        import pytest_asyncio

        @pytest_asyncio.fixture
        async def service():
            trigger = asyncio.Event()

            async def fail():
                await trigger.wait()
                raise RuntimeError("background service failed")

            async with asyncio.TaskGroup() as group:
                group.create_task(fail())
                yield trigger

        @pytest_asyncio.fixture
        async def dependent(service):
            service.set()
            await asyncio.Event().wait()
            yield

        @pytest.mark.asyncio(loop_scope="module")
        async def test_service(dependent):
            pass

        @pytest.mark.asyncio(loop_scope="module")
        async def test_after():
            pass
        """))
    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)
    result.assert_outcomes(errors=2, failed=1)
    result.stdout.fnmatch_lines(
        [
            "*ERROR at setup of test_service*",
            *_fixture_cancelled("setup"),
            "*ERROR at teardown of test_service*",
            "*RuntimeError: background service failed*",
            "*_ test_after _*",
            _REFUSED,
        ]
    )


@_REQUIRES_311
def test_a_child_failing_during_the_test_cancels_the_test(pytester: Pytester):
    """
    Issue #1083: the group is exited by the fixture's task, at its teardown,
    where its error is reported; the test fails, and the loop refuses the rest.
    """
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = module")
    pytester.makeconftest(dedent(_EVENTS_CONFTEST))
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest
        import pytest_asyncio
        from conftest import events

        @pytest_asyncio.fixture
        async def service():
            trigger = asyncio.Event()

            async def fail():
                await trigger.wait()
                raise RuntimeError("background service failed")

            async def serve():
                try:
                    await asyncio.Event().wait()
                finally:
                    events.append("sibling cancelled by the group")

            async with asyncio.TaskGroup() as group:
                group.create_task(fail())
                group.create_task(serve())
                yield trigger
            events.append("group exited")

        @pytest.mark.asyncio(loop_scope="module")
        async def test_service(service):
            service.set()
            await asyncio.Event().wait()

        @pytest.mark.asyncio(loop_scope="module")
        async def test_after():
            pass
        """))
    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)
    result.assert_outcomes(failed=2, errors=1)
    result.stdout.fnmatch_lines(
        [
            "*ERROR at teardown of test_service*",
            "*RuntimeError: background service failed*",
            "*_ test_service _*",
            "*asyncio.exceptions.CancelledError*",
            "*An async generator fixture was cancelled while it waited at*",
            "*_ test_after _*",
            _REFUSED,
        ]
    )
    assert _read_events(pytester) == ["sibling cancelled by the group"]


@pytest.mark.parametrize(("enter", "start"), _TASK_GROUPS)
def test_a_child_failing_between_tests_refuses_the_next_test(
    pytester: Pytester, enter: str, start: str
):
    """The group's error is reported at the fixture's teardown."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = module")
    pytester.makepyfile(dedent(f"""\
        import asyncio

        import anyio
        import pytest
        import pytest_asyncio

        @pytest_asyncio.fixture(scope="module")
        async def service():
            trigger = asyncio.Event()

            async def fail():
                await trigger.wait()
                raise RuntimeError("background service failed")

            {enter}
                {start}
                yield trigger

        @pytest.mark.asyncio(loop_scope="module")
        async def test_first(service):
            service.set()

        @pytest.mark.asyncio(loop_scope="module")
        async def test_second(service):
            pass
        """))
    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)
    result.assert_outcomes(passed=1, failed=1, errors=1)
    result.stdout.fnmatch_lines(
        [
            "*ERROR at teardown of test_second*",
            "*RuntimeError: background service failed*",
            "*_ test_second _*",
            _REFUSED,
        ]
    )


def test_an_anyio_task_group_spanning_the_yield_is_exited_by_the_task_that_entered_it(
    pytester: Pytester,
):
    """Issue #1191: the fixture's teardown exits the group's cancel scope."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
        import asyncio

        import anyio
        import pytest
        import pytest_asyncio

        @pytest_asyncio.fixture
        async def service():
            async with anyio.create_task_group() as group:
                group.start_soon(asyncio.Event().wait)
                yield
                group.cancel_scope.cancel()

        @pytest.mark.asyncio
        async def test_service(service):
            await asyncio.sleep(0)
        """))
    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)
    result.assert_outcomes(passed=1)


@pytest.mark.parametrize(("enter", "start"), _TASK_GROUPS)
def test_a_dependent_fixture_is_torn_down_before_its_cancelled_parent_closes(
    pytester: Pytester, enter: str, start: str
):
    """The dependent's teardown is in a task of its own, outside the parent's scope."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makeconftest(dedent(_EVENTS_CONFTEST))
    pytester.makepyfile(dedent(f"""\
        import asyncio

        import anyio
        import pytest
        import pytest_asyncio
        from conftest import events

        @pytest_asyncio.fixture
        async def connection():
            state = {{"open": True, "fail": asyncio.Event()}}

            async def fail():
                await state["fail"].wait()
                raise RuntimeError("connection worker failed")

            try:
                {enter}
                    {start}
                    yield state
            finally:
                state["open"] = False
                events.append("connection closed")

        @pytest_asyncio.fixture
        async def transaction(connection):
            yield connection
            await asyncio.sleep(0)
            assert connection["open"]
            events.append("transaction closed")

        @pytest.mark.asyncio
        async def test_transaction(transaction):
            transaction["fail"].set()
            await asyncio.Event().wait()
        """))
    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)
    result.assert_outcomes(failed=1, errors=1)
    result.stdout.fnmatch_lines(
        [
            "*ERROR at teardown of test_transaction*",
            "*RuntimeError: connection worker failed*",
        ]
    )
    assert _read_events(pytester) == ["transaction closed", "connection closed"]


@pytest.mark.parametrize(("enter", "expire"), _TIMEOUTS)
def test_a_timeout_spanning_the_yield_expiring_during_the_test_cancels_it(
    pytester: Pytester, enter: str, expire: str
):
    """The test fails with the raw CancelledError; the fixture's teardown is clean."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makeconftest(dedent(_EVENTS_CONFTEST))
    pytester.makepyfile(dedent(f"""\
        import asyncio

        import anyio
        import pytest
        import pytest_asyncio
        from conftest import events

        @pytest_asyncio.fixture
        async def deadline():
            {enter}
                yield deadline
            events.append("teardown clean")

        @pytest.mark.xfail(raises=asyncio.CancelledError, strict=True)
        @pytest.mark.asyncio
        async def test_deadline(deadline):
            {expire}
            await asyncio.Event().wait()
        """))
    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)
    result.assert_outcomes(xfailed=1)
    assert _read_events(pytester) == ["teardown clean"]


@pytest.mark.parametrize(("enter", "expire"), _TIMEOUTS)
def test_a_timeout_spanning_the_yield_expiring_while_idle_refuses_later_tests(
    pytester: Pytester, enter: str, expire: str
):
    """The loop refuses every later test, even once the timeout has exited."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = module")
    pytester.makeconftest(dedent(_EVENTS_CONFTEST))
    pytester.makepyfile(dedent(f"""\
        import asyncio

        import anyio
        import pytest
        import pytest_asyncio
        from conftest import events

        @pytest_asyncio.fixture
        async def deadline():
            {enter}
                yield deadline
            events.append("deadline exited")

        @pytest_asyncio.fixture
        async def dependent(deadline):
            {expire}
            yield

        @pytest.mark.asyncio(loop_scope="module")
        async def test_deadline(dependent):
            pass

        @pytest.mark.asyncio(loop_scope="module")
        async def test_after():
            pass
        """))
    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)
    result.assert_outcomes(failed=2)
    result.stdout.fnmatch_lines(
        ["*_ test_deadline _*", _REFUSED, "*_ test_after _*", _REFUSED]
    )
    assert _read_events(pytester) == ["deadline exited"]


def test_independent_module_fixtures_exit_their_scopes_out_of_setup_order(
    pytester: Pytester,
):
    """Each scope is in a task of its own, so no LIFO order binds the fixtures."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = module")
    pytester.makeconftest(dedent(_EVENTS_CONFTEST))
    pytester.makepyfile(dedent("""\
        import anyio
        import pytest
        import pytest_asyncio
        from conftest import events

        @pytest_asyncio.fixture(scope="module", params=[1, 2])
        async def first(request):
            async with anyio.create_task_group():
                events.append(f"open first {request.param}")
                yield request.param
                events.append(f"close first {request.param}")

        @pytest_asyncio.fixture(scope="module")
        async def second():
            async with anyio.create_task_group():
                events.append("open second")
                yield
                events.append("close second")

        @pytest.mark.asyncio(loop_scope="module")
        async def test_pair(first, second):
            events.append(f"test {first}")
        """))
    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)
    result.assert_outcomes(passed=2)
    assert _read_events(pytester) == [
        "open first 1",
        "open second",
        "test 1",
        "close first 1",
        "open first 2",
        "test 2",
        "close first 2",
        "close second",
    ]


@_REQUIRES_311
@pytest.mark.parametrize("failing", ["first", "second"])
def test_nested_task_group_fixtures_are_exited_in_dependency_order(
    pytester: Pytester, failing: str
):
    """Whichever group fails reports its error at its own fixture's teardown."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makeconftest(dedent(_EVENTS_CONFTEST))
    pytester.makepyfile(dedent(f"""\
        import asyncio

        import pytest
        import pytest_asyncio
        from conftest import events

        trigger = asyncio.Event()

        async def child(name):
            await trigger.wait()
            if name == {failing!r}:
                raise RuntimeError(name + " child failed")

        @pytest_asyncio.fixture
        async def first():
            async with asyncio.TaskGroup() as group:
                group.create_task(child("first"))
                try:
                    yield
                finally:
                    events.append("first teardown")

        @pytest_asyncio.fixture
        async def second(first):
            async with asyncio.TaskGroup() as group:
                group.create_task(child("second"))
                try:
                    yield
                finally:
                    events.append("second teardown")

        @pytest.mark.asyncio
        async def test_services(second):
            try:
                trigger.set()
                await asyncio.Event().wait()
            finally:
                events.append("test")
        """))
    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)
    result.assert_outcomes(failed=1, errors=1)
    result.stdout.fnmatch_lines([f"*RuntimeError: {failing} child failed*"])
    assert _read_events(pytester) == ["test", "second teardown", "first teardown"]


def test_a_fixture_that_never_yields_is_a_setup_error(pytester: Pytester):
    """The generator's StopAsyncIteration is the error; the next test runs."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
        import pytest
        import pytest_asyncio

        @pytest_asyncio.fixture
        async def generator():
            return
            yield

        @pytest.mark.asyncio
        async def test_generator(generator):
            pass

        @pytest.mark.asyncio
        async def test_next():
            pass
        """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(passed=1, errors=1)
    result.stdout.fnmatch_lines(
        ["*ERROR at setup of test_generator*", "*StopAsyncIteration*"]
    )


def test_a_fixture_that_yields_twice_is_a_teardown_error(pytester: Pytester):
    """The second yield is reported at teardown; the tests themselves pass."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
        import pytest
        import pytest_asyncio

        @pytest_asyncio.fixture
        async def generator():
            yield
            yield

        @pytest.mark.asyncio
        async def test_generator(generator):
            pass

        @pytest.mark.asyncio
        async def test_next():
            pass
        """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(passed=2, errors=1)
    result.stdout.fnmatch_lines(
        [
            "*ERROR at teardown of test_generator*",
            "*ValueError: Async generator fixture yielded more than once*",
        ]
    )


@_REQUIRES_311
@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(
            "def factory():\n"
            "    loop = asyncio.new_event_loop()\n"
            "    loop.set_task_factory(asyncio.eager_task_factory)\n"
            "    return loop",
            id="eager",
            marks=_REQUIRES_312,
        ),
        pytest.param(
            "import uvloop\nfactory = uvloop.new_event_loop",
            id="uvloop",
            marks=_REQUIRES_UVLOOP,
        ),
    ],
)
def test_a_failed_fixture_group_is_reported_the_same_on_other_loops(
    pytester: Pytester, factory: str
):
    """An eager task factory or uvloop: the test is cancelled, the rest refused."""
    pytester.makeini(
        "[pytest]\nasyncio_default_fixture_loop_scope = module\nasyncio_debug = true"
    )
    pytester.makeconftest(
        "import asyncio\n"
        f"{factory}\n"
        "def pytest_asyncio_loop_factories(config, item):\n"
        "    return {'factory': factory}\n"
    )
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest
        import pytest_asyncio

        @pytest_asyncio.fixture
        async def service():
            trigger = asyncio.Event()

            async def fail():
                await trigger.wait()
                raise RuntimeError("background service failed")

            async with asyncio.TaskGroup() as group:
                group.create_task(fail())
                yield trigger

        @pytest.mark.asyncio(loop_scope="module")
        async def test_service(service):
            service.set()
            await asyncio.Event().wait()

        @pytest.mark.asyncio(loop_scope="module")
        async def test_after():
            pass
        """))
    result = pytester.runpytest_subprocess(
        "--asyncio-mode=strict", "-W", "error", timeout=30
    )
    result.assert_outcomes(failed=2, errors=1)
    result.stdout.fnmatch_lines(
        ["*RuntimeError: background service failed*", "*_ test_after _*", _REFUSED]
    )
    output = result.stdout.str() + result.stderr.str()
    for noise in ("Task was destroyed", "never retrieved", "never awaited"):
        assert noise not in output


def test_a_cancelled_dependent_teardown_does_not_prevent_the_parent_teardown(
    pytester: Pytester,
):
    """Both teardown errors are reported: the cancellation, and the parent's."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makeconftest(dedent(_EVENTS_CONFTEST))
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest
        import pytest_asyncio
        from conftest import events

        @pytest_asyncio.fixture
        async def parent():
            yield
            events.append("parent cleanup ran")
            raise RuntimeError("parent cleanup failed")

        @pytest_asyncio.fixture
        async def child(parent):
            yield
            asyncio.current_task().cancel()
            await asyncio.sleep(0)
            events.append("child cleanup ran")

        @pytest.mark.asyncio
        async def test_it(child):
            pass
        """))
    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)
    result.assert_outcomes(passed=1, errors=1)
    result.stdout.fnmatch_lines(_fixture_cancelled("teardown"))
    result.stdout.fnmatch_lines(["*RuntimeError: parent cleanup failed*"])
    assert _read_events(pytester) == ["parent cleanup ran"]


def test_an_async_test_cannot_request_an_async_fixture_dynamically(
    pytester: Pytester,
):
    """The request is refused before the fixture is set up."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
        import pytest
        import pytest_asyncio

        calls = []

        @pytest_asyncio.fixture
        async def generator():
            calls.append("set up")
            yield
            calls.append("torn down")

        @pytest.mark.asyncio
        async def test_dynamic(request):
            request.getfixturevalue("generator")

        def test_the_fixture_never_ran():
            assert calls == []
        """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(passed=1, failed=1)
    result.stdout.fnmatch_lines(
        ["*cannot start an async fixture or test while the event loop is running*"]
    )
