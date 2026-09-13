"""
The other tests on a shared event loop after a cancellation.

Every test and fixture of an event loop scope runs in a task of its own, so
a cancellation ends the test or fixture whose task it reached and nothing
else: the later tests on the loop run. The exception is an async generator
fixture's task cancelled as it waits at its yield, by a task group, cancel
scope or timeout spanning the yield or by a test cancelling every task: the
loop then runs nothing but fixture teardowns until it closes. A refused test
fails with CancelledError; a refused fixture setup errors with
PytestAsyncioError, the refusal as its cause. A function-scoped loop closes
right after its test.
"""

from __future__ import annotations

import sys
from textwrap import dedent

import pytest
from pytest import Pytester

_REFUSED = "*CancelledError: The coroutine was refused: *until the loop is closed."
# pytest prints the cause of a fixture's PytestAsyncioError first.
_REFUSED_SETUP = [
    _REFUSED,
    "*PytestAsyncioError: The setup of the async fixture was cancelled.*",
]
_NEEDS_TASK_GROUP = pytest.mark.skipif(
    sys.version_info < (3, 11), reason="asyncio.TaskGroup needs Python 3.11"
)


def _run(
    pytester: Pytester, source: str, *args: str, subprocess: bool = False, **files: str
):
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent(source), **{k: dedent(v) for k, v in files.items()})
    if subprocess:
        return pytester.runpytest_subprocess(
            "--asyncio-mode=strict", "-s", *args, timeout=30
        )
    return pytester.runpytest("--asyncio-mode=strict", "-s", *args)


# The test cancels its own task, or every task of its loop.
_CANCELLING_TEST_SOURCE = """
    import asyncio
    import contextlib
    import pytest
    import pytest_asyncio

    @pytest_asyncio.fixture(loop_scope="{scope}")
    async def resource():
        yield
        await asyncio.sleep(0)
        print("RESOURCE TORN DOWN")

    @pytest.mark.asyncio(loop_scope="{scope}")
    async def test_cancel(resource):
        {cancel}
        {after}

    @pytest.mark.asyncio(loop_scope="{scope}")
    async def test_next():
        print("NEXT RAN")

    @pytest.mark.asyncio(loop_scope="{scope}")
    async def test_with_fixture(resource):
        print("WITH FIXTURE RAN")
"""


@pytest.mark.parametrize(
    ("after", "failed"),
    [
        pytest.param("await asyncio.sleep(0)", 1, id="propagated"),
        pytest.param("pass", 1, id="at_return"),
        pytest.param(
            "with contextlib.suppress(asyncio.CancelledError): await asyncio.sleep(0)",
            0,
            id="swallowed",
        ),
    ],
)
def test_self_cancellation_affects_only_the_culprit(
    pytester: Pytester, after: str, failed: int
):
    """
    A test that cancels its own task fails with CancelledError, whether it
    lets the cancellation propagate or returns before it is delivered, and
    passes if it swallows it. Its fixture is torn down either way, and the
    later tests on the shared loop run.
    """
    source = _CANCELLING_TEST_SOURCE.format(
        scope="module", cancel="asyncio.current_task().cancel()", after=after
    )
    result = _run(pytester, source)
    result.assert_outcomes(failed=failed, passed=3 - failed)
    if failed:
        result.stdout.fnmatch_lines(["*_ test_cancel _*", "*CancelledError*"])
    assert result.stdout.str().count("RESOURCE TORN DOWN") == 2
    result.stdout.fnmatch_lines(["*NEXT RAN*", "*WITH FIXTURE RAN*"])


@pytest.mark.parametrize("scope", ["function", "module"])
def test_cancelling_every_task_reaches_the_fixture_at_its_yield(
    pytester: Pytester, scope: str
):
    """
    asyncio.all_tasks() holds the tasks of the fixtures waiting at their
    yield, and the runner's wait for the test. The test fails with
    CancelledError; the fixture's task, cancelled at its yield, makes its
    loop teardown-only until it closes: right after the test on a function
    loop, at the end of the module on a module loop.
    """
    source = _CANCELLING_TEST_SOURCE.format(
        scope=scope,
        cancel="for task in asyncio.all_tasks(): task.cancel()",
        after="await asyncio.sleep(0)",
    )
    result = _run(pytester, source)
    result.stdout.fnmatch_lines(["*_ test_cancel _*", "*CancelledError*"])
    out = result.stdout.str()
    if scope == "function":
        result.assert_outcomes(failed=1, passed=2)
        assert out.count("RESOURCE TORN DOWN") == 2
        return
    result.assert_outcomes(failed=2, errors=1)
    assert out.count("RESOURCE TORN DOWN") == 1
    assert "NEXT RAN" not in out
    assert "WITH FIXTURE RAN" not in out
    result.stdout.fnmatch_lines(
        [
            "*ERROR at setup of test_with_fixture*",
            *_REFUSED_SETUP,
            "*_ test_next _*",
            _REFUSED,
        ]
    )


@_NEEDS_TASK_GROUP
@pytest.mark.parametrize(
    "cancel",
    [
        pytest.param("task.cancel()", id="before_its_first_step"),
        pytest.param("loop.call_soon(task.cancel)", id="right_after_it_entered"),
    ],
)
def test_loop_whose_task_factory_cancels_the_first_task(
    pytester: Pytester, cancel: str
):
    """
    The first task of a loop is pytest-asyncio's own, which owns the async
    generator fixtures' task group. Cancelled before it entered the group,
    it fails the opening of the loop scope, which is reported as a fixture
    error for every test of the scope, and the loop is closed. Cancelled
    right after it entered, it ends the normal work of the loop: every
    test and fixture setup is refused, and the loop closes at the end of
    the scope.
    """
    result = _run(
        pytester,
        """
        import asyncio
        import pytest
        import pytest_asyncio

        @pytest_asyncio.fixture(loop_scope="module")
        async def resource():
            print("RESOURCE STARTED")
            yield

        @pytest.mark.asyncio(loop_scope="module")
        async def test_first(resource):
            pass

        @pytest.mark.asyncio(loop_scope="module")
        async def test_without_fixtures():
            print("RAN WITHOUT FIXTURES")
        """,
        "-p",
        "conftest",
        conftest=f"""
        import asyncio

        loops = []

        def cancel_first_task_loop():
            loop = asyncio.new_event_loop()
            loops.append(loop)
            first = True

            def task_factory(loop, coro, **kwargs):
                nonlocal first
                task = asyncio.Task(coro, loop=loop, **kwargs)
                if first:
                    first = False
                    {cancel}
                return task

            loop.set_task_factory(task_factory)
            return loop

        def pytest_asyncio_loop_factories(config, item):
            return {{"cancel-first": cancel_first_task_loop}}

        def pytest_sessionfinish(session, exitstatus):
            print("LOOPS CLOSED:", [loop.is_closed() for loop in loops])
        """,
        subprocess=True,
    )
    out = result.stdout.str()
    assert "RESOURCE STARTED" not in out
    assert "RAN WITHOUT FIXTURES" not in out
    assert "LOOPS CLOSED: [True]" in out
    if cancel == "task.cancel()":
        result.assert_outcomes(errors=2)
        result.stdout.fnmatch_lines(
            ["*PytestAsyncioError: pytest-asyncio could not open this event loop: *"]
        )
    else:
        result.assert_outcomes(errors=1, failed=1)
        result.stdout.fnmatch_lines(
            [*_REFUSED_SETUP, "*_ test_without_fixtures _*", _REFUSED]
        )


def test_fixture_cancelled_during_its_setup_fails_only_its_test(pytester: Pytester):
    """
    A cancellation of a fixture's task before its yield is a failed setup
    of that fixture, reported as PytestAsyncioError, whether the fixture is
    a coroutine or an async generator; the later tests on the shared loop
    run.
    """
    result = _run(
        pytester,
        """
        import asyncio
        import pytest
        import pytest_asyncio

        async def cancel_self():
            asyncio.current_task().cancel()
            await asyncio.sleep(0)

        @pytest_asyncio.fixture(loop_scope="module")
        async def cancelled_coroutine():
            await cancel_self()

        @pytest_asyncio.fixture(loop_scope="module")
        async def cancelled_generator():
            await cancel_self()
            yield

        @pytest.mark.asyncio(loop_scope="module")
        async def test_coroutine(cancelled_coroutine):
            pass

        @pytest.mark.asyncio(loop_scope="module")
        async def test_generator(cancelled_generator):
            pass

        @pytest.mark.asyncio(loop_scope="module")
        async def test_next():
            print("NEXT RAN")
        """,
    )
    result.assert_outcomes(passed=1, errors=2)
    result.stdout.fnmatch_lines(
        [
            "*NEXT RAN*",
            "*ERROR at setup of test_coroutine*",
            "*PytestAsyncioError: The setup of the async fixture was cancelled.*",
            "*ERROR at setup of test_generator*",
            "*PytestAsyncioError: The setup of the async fixture was cancelled.*",
        ]
    )


_FAILED_SERVICE_SOURCE = """
    import asyncio
    import {library}
    import pytest
    import pytest_asyncio

    @pytest_asyncio.fixture(scope="{fixture_scope}", loop_scope="module")
    async def service():
        trigger = asyncio.Event()

        async def fail():
            await trigger.wait()
            raise RuntimeError("background service failed")

        async with {task_group}() as group:
            {start}
            yield trigger
            {stop}

    @pytest_asyncio.fixture(loop_scope="module")
    async def unrelated():
        yield
        await asyncio.sleep(0)
        print("UNRELATED TORN DOWN")

    @pytest.mark.asyncio(loop_scope="module")
    async def test_trigger(service, unrelated):
        service.set()
        await asyncio.Event().wait()

    @pytest.mark.asyncio(loop_scope="module")
    async def test_uses_service(service):
        print("USED SERVICE")

    @pytest.mark.asyncio(loop_scope="function")
    async def test_own_loop():
        print("OWN LOOP RAN")

    @pytest.mark.asyncio(loop_scope="module")
    async def test_unrelated(unrelated):
        print("UNRELATED RAN")
"""

_NEXT_MODULE_SOURCE = """
    import pytest

    @pytest.mark.asyncio(loop_scope="module")
    async def test_fresh_loop():
        print("NEXT MODULE RAN")
"""

_LIBRARIES = {
    "asyncio": {
        "task_group": "asyncio.TaskGroup",
        "start": "child = group.create_task(fail())",
        "stop": "child.cancel()",
    },
    "anyio": {
        "task_group": "anyio.create_task_group",
        "start": "group.start_soon(fail)",
        "stop": "group.cancel_scope.cancel()",
    },
}


def _run_failed_service(pytester: Pytester, library: str, fixture_scope: str):
    source = _FAILED_SERVICE_SOURCE.format(
        library=library, fixture_scope=fixture_scope, **_LIBRARIES[library]
    )
    result = _run(
        pytester, source, subprocess=True, test_next_module=_NEXT_MODULE_SOURCE
    )
    out = result.stdout.str()
    assert "USED SERVICE" not in out
    assert "UNRELATED RAN" not in out
    # The unrelated fixture's task is outside the cancelled scope.
    result.stdout.fnmatch_lines(
        ["*UNRELATED TORN DOWN*", "*OWN LOOP RAN*", "*NEXT MODULE RAN*"]
    )
    return result


@pytest.mark.parametrize(
    "library", [pytest.param("asyncio", marks=_NEEDS_TASK_GROUP), "anyio"]
)
def test_failed_module_fixture_makes_its_loop_teardown_only_until_it_closes(
    pytester: Pytester, library: str
):
    """
    The test running when the child failed is cancelled; every later test
    and fixture setup on the module loop is refused; fixture teardowns run,
    each in its own task; a function-loop test has its own loop; the next
    module starts with a fresh one. The group exits at the module teardown
    and reports its error once, there.
    """
    result = _run_failed_service(pytester, library, "module")
    result.assert_outcomes(failed=2, passed=2, errors=2)
    result.stdout.fnmatch_lines(
        [
            "*ERROR at setup of test_unrelated*",
            *_REFUSED_SETUP,
            "*ERROR at teardown of test_unrelated*",
            "*RuntimeError: background service failed*",
            "*_ test_uses_service _*",
            _REFUSED,
        ]
    )


@pytest.mark.parametrize(
    "library", [pytest.param("asyncio", marks=_NEEDS_TASK_GROUP), "anyio"]
)
def test_failed_function_fixture_makes_its_loop_teardown_only_until_it_closes(
    pytester: Pytester, library: str
):
    """
    The group exits at the fixture's teardown, right after the test it
    cancelled, and reports its error there. That does not admit the later
    tests: the cancellation was the loop's, and the fixture's own later
    setups are refused like everything else on it.
    """
    result = _run_failed_service(pytester, library, "function")
    result.assert_outcomes(failed=1, passed=2, errors=3)
    result.stdout.fnmatch_lines(
        [
            "*ERROR at teardown of test_trigger*",
            "*RuntimeError: background service failed*",
            "*ERROR at setup of test_uses_service*",
            *_REFUSED_SETUP,
            "*ERROR at setup of test_unrelated*",
            *_REFUSED_SETUP,
        ]
    )


# The test does something to the running loop; what depends on it.
_LOOP_CONTROL_SOURCE = """
    import asyncio
    import pytest
    import pytest_asyncio

    @pytest_asyncio.fixture(loop_scope="module")
    async def resource():
        yield
        await asyncio.sleep(0)
        print("RESOURCE TORN DOWN")

    @pytest.mark.asyncio(loop_scope="module")
    async def test_{action}(resource):
        asyncio.get_running_loop().{action}()
        await asyncio.sleep(0)
        print("TEST CONTINUED")

    @pytest.mark.asyncio(loop_scope="module")
    async def test_next(resource):
        print("NEXT RAN")
"""


@pytest.mark.parametrize(
    ("action", "error"),
    [
        ("stop", "Event loop stopped before Future completed"),
        ("close", "Cannot close a running event loop"),
    ],
)
def test_test_stops_or_closes_the_running_loop(
    pytester: Pytester, action: str, error: str
):
    """
    Stopping the loop interrupts the wait for the test: the runner cancels
    the test's task, which ends at its await, and raises asyncio's error.
    Closing a running loop is refused by asyncio in the test itself. Either
    way the test fails and the shared loop goes on.
    """
    result = _run(pytester, _LOOP_CONTROL_SOURCE.format(action=action))
    result.assert_outcomes(failed=1, passed=1)
    out = result.stdout.str()
    assert "TEST CONTINUED" not in out
    assert out.count("RESOURCE TORN DOWN") == 2
    result.stdout.fnmatch_lines(
        ["*NEXT RAN*", f"*_ test_{action} _*", f"*RuntimeError: {error}*"]
    )


def test_sync_test_closes_the_shared_loop(pytester: Pytester):
    """Later async work on the loop fails with asyncio's error; the runner warns."""
    result = _run(
        pytester,
        """
        import asyncio
        import pytest

        @pytest.mark.asyncio(loop_scope="module")
        async def test_first():
            pass

        def test_close_loop():
            asyncio.get_event_loop().close()

        @pytest.mark.asyncio(loop_scope="module")
        async def test_after_close():
            print("AFTER CLOSE RAN")
        """,
        "-W",
        "default",
        subprocess=True,
    )
    result.assert_outcomes(passed=2, failed=1)
    assert "AFTER CLOSE RAN" not in result.stdout.str()
    result.stdout.fnmatch_lines(
        [
            "*RuntimeError: Event loop is closed*",
            "*An exception occurred during teardown of an asyncio.Runner*",
        ]
    )


def test_unstructured_task_outlives_the_test(pytester: Pytester):
    """It runs until its loop closes; the runner cancels it then, not asyncio's GC."""
    result = _run(
        pytester,
        """
        import asyncio
        import pytest
        import pytest_asyncio

        async def forever():
            try:
                await asyncio.Event().wait()
            finally:
                print("ORPHAN CANCELLED")

        @pytest_asyncio.fixture(scope="module", loop_scope="module")
        async def orphan():
            return asyncio.create_task(forever())

        @pytest.mark.asyncio(loop_scope="module")
        async def test_spawned(orphan):
            await asyncio.sleep(0)

        @pytest.mark.asyncio(loop_scope="module")
        async def test_still_running(orphan):
            assert not orphan.done()
        """,
        "-W",
        "error",
        subprocess=True,
    )
    result.assert_outcomes(passed=2)
    result.stdout.fnmatch_lines(["*ORPHAN CANCELLED*"])
    assert "Task was destroyed" not in result.stdout.str() + result.stderr.str()


def test_user_drives_the_shared_loop_after_an_anyio_scope_was_cancelled(
    pytester: Pytester,
):
    """
    AnyIO cancels the fixture's task again on every iteration of the loop
    the sync test drives; the task keeps waiting at the yield, the user's
    coroutine completes, the next test is refused, and the fixture's
    teardown exits the scope and reports the error.
    """
    result = _run(
        pytester,
        """
        import asyncio
        import anyio
        import pytest
        import pytest_asyncio

        async def fail_soon():
            await asyncio.sleep(0.01)
            raise RuntimeError("background service failed")

        @pytest_asyncio.fixture(scope="module", loop_scope="module")
        async def service():
            async with anyio.create_task_group() as group:
                group.start_soon(fail_soon)
                yield

        @pytest.mark.asyncio(loop_scope="module")
        async def test_cancelled(service):
            await asyncio.Event().wait()

        def test_drives_loop(service):
            loop = asyncio.get_event_loop()
            assert loop.run_until_complete(asyncio.sleep(0.05, result=42)) == 42
            print("USER COROUTINE COMPLETED")

        @pytest.mark.asyncio(loop_scope="module")
        async def test_next(service):
            print("NEXT RAN")
        """,
        subprocess=True,
    )
    result.assert_outcomes(failed=2, passed=1, errors=1)
    assert "NEXT RAN" not in result.stdout.str()
    result.stdout.fnmatch_lines(["*USER COROUTINE COMPLETED*"])
    result.stdout.fnmatch_lines(
        [
            "*ERROR at teardown of test_next*",
            "*RuntimeError: background service failed*",
            "*_ test_next _*",
            _REFUSED,
        ]
    )
