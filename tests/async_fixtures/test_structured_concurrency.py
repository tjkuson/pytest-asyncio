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

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="The experimental runner requires Python 3.11",
)
_REQUIRES_312 = pytest.mark.skipif(
    sys.version_info < (3, 12), reason="asyncio.eager_task_factory needs Python 3.12"
)
_REQUIRES_UVLOOP = pytest.mark.skipif(
    importlib.util.find_spec("uvloop") is None, reason="uvloop is not installed"
)

_TASK_GROUPS = [
    pytest.param(
        "async with asyncio.TaskGroup() as group:",
        "group.create_task(fail())",
        id="asyncio",
    ),
    pytest.param(
        "async with anyio.create_task_group() as group:",
        "group.start_soon(fail)",
        id="anyio",
    ),
]
_TIMEOUTS = [
    pytest.param(
        "async with asyncio.timeout(None) as deadline:",
        "deadline.reschedule(asyncio.get_running_loop().time())",
        id="asyncio",
    ),
    pytest.param(
        "with anyio.fail_after(None) as deadline:",
        "deadline.deadline = anyio.current_time()",
        id="anyio",
    ),
]
_REFUSED = "*This event loop no longer accepts new async tests*"


def test_a_generator_fixture_is_set_up_and_torn_down_in_one_task(pytester: Pytester):
    """A fixture exits a task-bound scope in the task that entered it."""
    pytester.makeini(
        "[pytest]\n"
        "experimental_asyncio_task_group_runner = true\n"
        "asyncio_default_fixture_loop_scope = module"
    )
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest
        import pytest_asyncio

        @pytest_asyncio.fixture
        async def generator():
            setup_task = asyncio.current_task()
            yield
            assert asyncio.current_task() is setup_task

        @pytest.mark.asyncio(loop_scope="module")
        async def test_uses_fixture(generator):
            pass
        """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(passed=1)


def test_a_failed_assertion_does_not_prevent_later_tests_using_the_same_loop(
    pytester: Pytester,
):
    """An ordinary test failure leaves the shared loop available to later tests."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
        import pytest

        @pytest.mark.asyncio(loop_scope="module")
        async def test_fails():
            raise AssertionError("reported to pytest")

        @pytest.mark.asyncio(loop_scope="module")
        async def test_next():
            pass
        """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(failed=1, passed=1)
    result.stdout.fnmatch_lines(["*AssertionError: reported to pytest*"])


def test_a_cancellation_at_the_end_of_a_fixture_teardown_is_a_teardown_error(
    pytester: Pytester,
):
    """A fixture that cancels its own task as it returns is reported as cancelled."""
    pytester.makeini(
        "[pytest]\n"
        "experimental_asyncio_task_group_runner = true\n"
        "asyncio_default_fixture_loop_scope = function"
    )
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
        [
            "*ERROR at teardown of test_uses_resource*",
            "*CancelledError: cancelled at the end of the teardown*",
            "*PytestAsyncioError: The teardown of async fixture 'resource'*cancelled.*",
        ]
    )


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


def test_a_child_failing_while_a_dependent_fixture_sets_up_cancels_that_setup(
    pytester: Pytester,
):
    """The group's error is reported at the fixture's teardown; the rest is refused."""
    pytester.makeini(
        "[pytest]\n"
        "experimental_asyncio_task_group_runner = true\n"
        "asyncio_default_fixture_loop_scope = module"
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
            "*asyncio.exceptions.CancelledError*",
            "*direct cause of the following*",
            "*PytestAsyncioError: The setup of async fixture * was cancelled.*",
            "*ERROR at teardown of test_service*",
            "*RuntimeError: background service failed*",
            "*_ test_after _*",
            _REFUSED,
        ]
    )


def test_a_child_failing_during_the_test_cancels_the_test(pytester: Pytester):
    """Issue #1083: cancel the test and report the child's error at fixture teardown."""
    pytester.makeini(
        "[pytest]\n"
        "experimental_asyncio_task_group_runner = true\n"
        "asyncio_default_fixture_loop_scope = module"
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
    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)
    result.assert_outcomes(failed=2, errors=1)
    result.stdout.fnmatch_lines(
        [
            "*ERROR at teardown of test_service*",
            "*RuntimeError: background service failed*",
            "*_ test_service _*",
            "*asyncio.exceptions.CancelledError*",
            "*Async fixture 'service' was cancelled while waiting for teardown.*",
            "*_ test_after _*",
            _REFUSED,
        ]
    )


@pytest.mark.parametrize(("enter_group", "start_child"), _TASK_GROUPS)
def test_a_child_failing_between_tests_refuses_the_next_test(
    pytester: Pytester, enter_group: str, start_child: str
):
    """The group's error is reported at the fixture's teardown."""
    pytester.makeini(
        "[pytest]\n"
        "experimental_asyncio_task_group_runner = true\n"
        "asyncio_default_fixture_loop_scope = module"
    )
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

            {enter_group}
                {start_child}
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
    pytester.makeini(
        "[pytest]\n"
        "experimental_asyncio_task_group_runner = true\n"
        "asyncio_default_fixture_loop_scope = function"
    )
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


@pytest.mark.parametrize(("enter_group", "start_child"), _TASK_GROUPS)
def test_cancelled_fixture_keeps_resources_available_until_dependent_teardown_finishes(
    pytester: Pytester, enter_group: str, start_child: str
):
    """Dependent teardown can use the parent's files before its directory is removed."""
    pytester.makeini(
        "[pytest]\n"
        "experimental_asyncio_task_group_runner = true\n"
        "asyncio_default_fixture_loop_scope = function"
    )
    pytester.makepyfile(dedent(f"""\
        import asyncio
        from pathlib import Path
        from tempfile import TemporaryDirectory

        import anyio
        import pytest
        import pytest_asyncio

        @pytest_asyncio.fixture
        async def parent():
            failure = asyncio.Event()

            async def fail():
                await failure.wait()
                raise RuntimeError("background task failed")

            with TemporaryDirectory(prefix="fixture-", dir=".") as directory:
                source_file = Path(directory, "data.txt")
                source_file.write_text("fixture data")
                {enter_group}
                    {start_child}
                    yield source_file, failure
                    Path("cancelled-fixture.txt").write_text("continued after yield")

        @pytest_asyncio.fixture
        async def dependent(parent):
            source_file, failure = parent
            yield failure
            await asyncio.sleep(0)
            source_file.replace("saved.txt")

        @pytest.mark.asyncio
        async def test_uses_dependent(dependent):
            dependent.set()
            await asyncio.Event().wait()
        """))
    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)
    result.assert_outcomes(failed=1, errors=1)
    result.stdout.fnmatch_lines(
        [
            "*ERROR at teardown of test_uses_dependent*",
            "*RuntimeError: background task failed*",
        ]
    )
    assert (pytester.path / "saved.txt").read_text() == "fixture data"
    assert list(pytester.path.glob("fixture-*")) == []
    assert not (pytester.path / "cancelled-fixture.txt").exists()


@pytest.mark.parametrize(("enter_timeout", "expire_timeout"), _TIMEOUTS)
def test_a_timeout_spanning_the_yield_expiring_during_the_test_cancels_it(
    pytester: Pytester, enter_timeout: str, expire_timeout: str
):
    """An expired fixture cancels the test and reports TimeoutError at teardown."""
    pytester.makeini(
        "[pytest]\n"
        "experimental_asyncio_task_group_runner = true\n"
        "asyncio_default_fixture_loop_scope = function"
    )
    pytester.makepyfile(dedent(f"""\
        import asyncio

        import anyio
        import pytest
        import pytest_asyncio

        @pytest_asyncio.fixture
        async def deadline():
            {enter_timeout}
                yield deadline

        @pytest.mark.asyncio
        async def test_deadline(deadline):
            {expire_timeout}
            await asyncio.Event().wait()
        """))
    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)
    result.assert_outcomes(failed=1, errors=1)
    result.stdout.fnmatch_lines(
        [
            "*ERROR at teardown of test_deadline*",
            "*TimeoutError*",
            "*_ test_deadline _*",
            "*CancelledError*",
        ]
    )


@pytest.mark.parametrize(("enter_timeout", "expire_timeout"), _TIMEOUTS)
def test_a_timeout_expiring_during_dependent_setup_refuses_later_tests(
    pytester: Pytester, enter_timeout: str, expire_timeout: str
):
    """Later tests on the loop are refused even when they do not use the fixture."""
    pytester.makeini(
        "[pytest]\n"
        "experimental_asyncio_task_group_runner = true\n"
        "asyncio_default_fixture_loop_scope = module"
    )
    pytester.makepyfile(dedent(f"""\
        import asyncio

        import anyio
        import pytest
        import pytest_asyncio

        @pytest_asyncio.fixture
        async def deadline():
            {enter_timeout}
                yield deadline

        @pytest_asyncio.fixture
        async def dependent(deadline):
            {expire_timeout}
            yield

        @pytest.mark.asyncio(loop_scope="module")
        async def test_deadline(dependent):
            pass

        @pytest.mark.asyncio(loop_scope="module")
        async def test_after():
            pass
        """))
    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)
    result.assert_outcomes(failed=2, errors=1)
    result.stdout.fnmatch_lines(
        ["*_ test_deadline _*", _REFUSED, "*_ test_after _*", _REFUSED]
    )


def test_parametrized_module_fixtures_can_each_use_an_anyio_task_group(
    pytester: Pytester,
):
    """Replacing a parameter's group leaves the other fixture's cached group usable."""
    pytester.makeini(
        "[pytest]\n"
        "experimental_asyncio_task_group_runner = true\n"
        "asyncio_default_fixture_loop_scope = module"
    )
    pytester.makepyfile(dedent("""\
        import anyio
        import pytest
        import pytest_asyncio

        @pytest_asyncio.fixture(scope="module", params=[1, 2])
        async def first(request):
            async with anyio.create_task_group():
                yield request.param

        @pytest_asyncio.fixture(scope="module")
        async def second():
            async with anyio.create_task_group() as group:
                yield group

        @pytest.mark.asyncio(loop_scope="module")
        async def test_pair(first, second):
            second.start_soon(anyio.sleep, 0)
        """))
    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)
    result.assert_outcomes(passed=2)


@pytest.mark.parametrize("failing", ["first", "second"])
def test_a_failure_in_either_nested_fixture_group_is_reported_at_teardown(
    pytester: Pytester, failing: str
):
    """Whichever group fails reports its error at its own fixture's teardown."""
    pytester.makeini(
        "[pytest]\n"
        "experimental_asyncio_task_group_runner = true\n"
        "asyncio_default_fixture_loop_scope = function"
    )
    pytester.makepyfile(dedent(f"""\
        import asyncio

        import pytest
        import pytest_asyncio

        trigger = asyncio.Event()

        async def child(name):
            await trigger.wait()
            if name == {failing!r}:
                raise RuntimeError(name + " child failed")

        @pytest_asyncio.fixture
        async def first():
            async with asyncio.TaskGroup() as group:
                group.create_task(child("first"))
                yield

        @pytest_asyncio.fixture
        async def second(first):
            async with asyncio.TaskGroup() as group:
                group.create_task(child("second"))
                yield

        @pytest.mark.asyncio
        async def test_services(second):
            trigger.set()
            await asyncio.Event().wait()
        """))
    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)
    result.assert_outcomes(failed=1, errors=1)
    result.stdout.fnmatch_lines(
        [
            "*ERROR at teardown of test_services*",
            f"*RuntimeError: {failing} child failed*",
            "*_ test_services _*",
            "*CancelledError*",
        ]
    )


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
    pytester.makeini(
        "[pytest]\n"
        "experimental_asyncio_task_group_runner = true\n"
        "asyncio_default_fixture_loop_scope = function"
    )
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
def test_custom_loop_factories_preserve_fixture_failure_reporting(
    pytester: Pytester, factory: str
):
    """Fixture failure cancels the test and prevents further tests on its loop."""
    pytester.makeini(
        "[pytest]\n"
        "experimental_asyncio_task_group_runner = true\n"
        "asyncio_default_fixture_loop_scope = module\n"
        "asyncio_debug = true"
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
    assert "Task was destroyed" not in output
    assert "never retrieved" not in output
    assert "never awaited" not in output


def test_a_cancelled_dependent_teardown_does_not_prevent_the_parent_teardown(
    pytester: Pytester,
):
    """Both teardown errors are reported: the cancellation, and the parent's."""
    pytester.makeini(
        "[pytest]\n"
        "experimental_asyncio_task_group_runner = true\n"
        "asyncio_default_fixture_loop_scope = function"
    )
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest
        import pytest_asyncio

        @pytest_asyncio.fixture
        async def parent():
            yield
            raise RuntimeError("parent cleanup failed")

        @pytest_asyncio.fixture
        async def child(parent):
            yield
            asyncio.current_task().cancel()
            await asyncio.sleep(0)

        @pytest.mark.asyncio
        async def test_it(child):
            pass
        """))
    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)
    result.assert_outcomes(passed=1, errors=1)
    result.stdout.fnmatch_lines(
        [
            "*asyncio.exceptions.CancelledError*",
            "*direct cause of the following*",
            "*PytestAsyncioError: The teardown of async fixture * was cancelled.*",
        ]
    )
    result.stdout.fnmatch_lines(["*RuntimeError: parent cleanup failed*"])


def test_a_fixture_that_has_yielded_is_not_cancelled_by_another_fixture_failure(
    pytester: Pytester,
):
    """A completed setup still gets its ordinary teardown after its parent fails."""
    pytester.makeini(
        "[pytest]\n"
        "experimental_asyncio_task_group_runner = true\n"
        "asyncio_default_fixture_loop_scope = function"
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
                raise RuntimeError("service failed")

            with open("report.txt", "w") as report:
                async with asyncio.TaskGroup() as group:
                    group.create_task(fail())
                    yield trigger, report

        @pytest_asyncio.fixture
        async def dependent(service):
            trigger, report = service
            trigger.set()
            yield
            await asyncio.sleep(0)
            report.write("dependent teardown completed")

        @pytest.mark.asyncio
        async def test_uses_service(dependent):
            pass
        """))
    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)
    result.assert_outcomes(failed=1, errors=1)
    result.stdout.fnmatch_lines(["*RuntimeError: service failed*"])
    assert (pytester.path / "report.txt").read_text() == "dependent teardown completed"


def test_a_fixture_timeout_is_reported_when_the_test_catches_its_cancellation(
    pytester: Pytester,
):
    """The fixture reports its timeout even if the test catches cancellation."""
    pytester.makeini(
        "[pytest]\n"
        "experimental_asyncio_task_group_runner = true\n"
        "asyncio_default_fixture_loop_scope = function"
    )
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest
        import pytest_asyncio

        @pytest_asyncio.fixture
        async def deadline():
            async with asyncio.timeout(None) as timeout:
                yield timeout

        @pytest.mark.asyncio
        async def test_deadline(deadline):
            deadline.reschedule(asyncio.get_running_loop().time())
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                pass
        """))
    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)
    result.assert_outcomes(passed=1, errors=1)
    result.stdout.fnmatch_lines(
        ["*ERROR at teardown of test_deadline*", "*TimeoutError*"]
    )
