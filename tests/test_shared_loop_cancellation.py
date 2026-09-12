"""
The other tests on a shared event loop after a cancellation.

Every coroutine of an event loop scope runs in one long-lived task. Once
that task has been cancelled, by a scope of the user's (a task group,
cancel scope or timeout spanning a fixture's yield, a test cancelling its
own task) or by the runner to interrupt a job, it runs nothing but fixture
teardowns until the loop closes. A refused test fails with CancelledError;
a refused fixture setup errors with PytestAsyncioError, the refusal as its
cause. A function-scoped loop closes right after its test.
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


_SELF_CANCELLATION_SOURCE = """
    import asyncio
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


@pytest.mark.parametrize("scope", ["function", "module"])
@pytest.mark.parametrize(
    ("cancel", "after"),
    [
        ("asyncio.current_task().cancel()", "await asyncio.sleep(0)"),
        ("for task in asyncio.all_tasks(): task.cancel()", "await asyncio.sleep(0)"),
        ("asyncio.current_task().cancel()", "pass"),
    ],
    ids=["own_task", "every_task", "at_return"],
)
def test_unresolved_self_cancellation(
    pytester: Pytester, scope: str, cancel: str, after: str
):
    """
    Nobody resolves a request the test makes of its own task. Propagated, it
    fails the test with CancelledError; made at return, with no later await,
    it lands on the task as it waits for the next job and the test passes.
    Either way the test's fixture is torn down, and on a shared loop nothing
    but fixture teardowns runs until the loop closes.
    """
    source = _SELF_CANCELLATION_SOURCE.format(scope=scope, cancel=cancel, after=after)
    result = _run(pytester, source)
    culprit_failed = int(after != "pass")
    if culprit_failed:
        result.stdout.fnmatch_lines(["*_ test_cancel _*", "*CancelledError*"])
    out = result.stdout.str()
    if scope == "function":
        result.assert_outcomes(failed=culprit_failed, passed=3 - culprit_failed)
        assert out.count("RESOURCE TORN DOWN") == 2
        return
    result.assert_outcomes(
        failed=1 + culprit_failed, passed=1 - culprit_failed, errors=1
    )
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


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="asyncio.timeout() and Task.uncancel() need Python 3.11",
)
@pytest.mark.parametrize("resolve", ["pass", "task.uncancel()"])
def test_swallowed_self_cancellation_admits_later_tests(
    pytester: Pytester, resolve: str
):
    """
    The task never saw the cancellation, so nothing is refused; a request
    left unresolved by a test that does not call uncancel() does not upset a
    later timeout either (asyncio counts requests relatively).
    """
    result = _run(
        pytester,
        f"""
        import asyncio
        import pytest

        @pytest.mark.asyncio(loop_scope="module")
        async def test_swallow():
            task = asyncio.current_task()
            task.cancel()
            try:
                await asyncio.sleep(0)
            except asyncio.CancelledError:
                {resolve}

        @pytest.mark.asyncio(loop_scope="module")
        async def test_next():
            with pytest.raises(TimeoutError):
                async with asyncio.timeout(0.01):
                    await asyncio.sleep(1)
        """,
    )
    result.assert_outcomes(passed=2)


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
    # AnyIO's cancelled scope cancels the awaiting teardown again.
    assert ("UNRELATED TORN DOWN" in out) is (library == "asyncio")
    result.stdout.fnmatch_lines(["*OWN LOOP RAN*", "*NEXT MODULE RAN*"])
    return result


@pytest.mark.parametrize(
    "library", [pytest.param("asyncio", marks=_NEEDS_TASK_GROUP), "anyio"]
)
def test_failed_module_fixture_quarantines_its_loop_until_it_closes(
    pytester: Pytester, library: str
):
    """
    Every later test and fixture setup on the module loop is refused;
    fixture teardowns run; a function-loop test has its own task; the next
    module starts with a fresh loop. The group exits at the module teardown
    and reports its error once, there.
    """
    result = _run_failed_service(pytester, library, "module")
    result.assert_outcomes(failed=2, passed=2, errors=2 + (library == "anyio"))
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
def test_failed_function_fixture_quarantines_its_loop_until_it_closes(
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


def test_test_stops_the_running_loop(pytester: Pytester):
    """
    Stopping the loop interrupts the wait for the test: the runner cancels
    the task, the test ends at its await and fails with asyncio's error, and
    the shared loop runs nothing but fixture teardowns from then on.
    """
    result = _run(pytester, _LOOP_CONTROL_SOURCE.format(action="stop"))
    result.assert_outcomes(failed=1, errors=1)
    out = result.stdout.str()
    assert "TEST CONTINUED" not in out
    assert "NEXT RAN" not in out
    assert out.count("RESOURCE TORN DOWN") == 1
    result.stdout.fnmatch_lines(
        [
            "*ERROR at setup of test_next*",
            *_REFUSED_SETUP,
            "*_ test_stop _*",
            "*RuntimeError: Event loop stopped before Future completed*",
        ]
    )


def test_test_closes_the_running_loop(pytester: Pytester):
    """Closing a running loop is refused by asyncio: the test fails with its error."""
    result = _run(pytester, _LOOP_CONTROL_SOURCE.format(action="close"))
    result.assert_outcomes(failed=1, passed=1)
    out = result.stdout.str()
    assert "TEST CONTINUED" not in out
    assert out.count("RESOURCE TORN DOWN") == 2
    result.stdout.fnmatch_lines(
        ["*NEXT RAN*", "*RuntimeError: Cannot close a running event loop*"]
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
    AnyIO cancels the idle task again on every iteration of the loop the sync
    test drives; the user's coroutine completes and the task survives to
    refuse the next test and to tear the fixture down.
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
