"""
Task groups, cancel scopes and timeouts spanning the yield of an async fixture.

Every async fixture and test runs in a task of its own, and an async generator
fixture's task lives on across its yield, so a scope entered before the yield
is exited by the same task at teardown (issues #1083 and #1191); these tests
specify how failures in such a scope are reported.
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
_LIBRARIES = [pytest.param("asyncio", marks=_REQUIRES_311), "anyio"]

# A cancelled test raises the CancelledError itself, explained by a note on
# 3.11+. A cancelled fixture setup or teardown is an Exception caused by it, so
# that pytest goes on: with the fixture's other users, or the node's finalizers.
_CANCELLED = ["*asyncio.exceptions.CancelledError*"]
if sys.version_info >= (3, 11):
    _CANCELLED.append("*A task of pytest-asyncio's on this event loop was cancelled: *")
_REFUSED = "*RuntimeError: This event loop no longer accepts new tests*"


def _fixture_cancelled(phase: str) -> list[str]:
    """The error of a cancelled fixture phase, printed after its cause."""
    return [
        "*asyncio.exceptions.CancelledError*",
        "*direct cause of the following*",
        f"*PytestAsyncioError: The {phase} of the async fixture was cancelled.*",
    ]


def _task_group(library: str, *children: str) -> tuple[str, str]:
    """The task group of the library and the statement starting the children."""
    if library == "asyncio":
        return "asyncio.TaskGroup()", "; ".join(
            f"group.create_task({child}())" for child in children
        )
    return "anyio.create_task_group()", "; ".join(
        f"group.start_soon({child})" for child in children
    )


def _timeout(library: str) -> tuple[str, str]:
    """A timeout without a deadline and the statement making it expire now."""
    if library == "asyncio":
        return (
            "async with asyncio.timeout(None)",
            "deadline.reschedule(asyncio.get_running_loop().time())",
        )
    return "with anyio.fail_after(None)", "deadline.deadline = anyio.current_time()"


def test_one_task_per_fixture_and_test(pytester: Pytester):
    """
    Each fixture, test and hypothesis example of a loop runs in a task of its
    own, named after it; a generator fixture's setup and teardown share a task.
    """
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = module")
    pytester.makepyfile(dedent("""\
        import asyncio
        import sys
        import pytest
        import pytest_asyncio
        from hypothesis import given, settings, strategies as st

        tasks = {}
        examples = []

        def record(phase):
            tasks[phase] = asyncio.current_task()

        @pytest_asyncio.fixture
        async def generator():
            record("generator setup")
            yield
            record("generator teardown")

        @pytest_asyncio.fixture
        async def coroutine(generator):
            record("coroutine")

        @pytest.mark.asyncio(loop_scope="module")
        async def test_plain(coroutine):
            record("test_plain")
            # The test's own frame is in the task's await chain, seen while
            # the test is suspended.
            task = asyncio.current_task()
            loop = asyncio.get_running_loop()
            seen = loop.create_future()

            def frames():
                chain = []
                coro = task.get_coro()
                while coro is not None and hasattr(coro, "cr_frame"):
                    chain.append(coro.cr_frame)
                    coro = coro.cr_await
                seen.set_result(chain)

            loop.call_soon(frames)
            assert sys._getframe() in await seen

        @pytest.mark.asyncio(loop_scope="module")
        @settings(max_examples=3, deadline=None, database=None)
        @given(st.integers())
        async def test_examples(n):
            examples.append(asyncio.current_task())

        def test_tasks():
            assert tasks["generator setup"] is tasks["generator teardown"]
            own = [tasks["generator setup"], tasks["coroutine"], tasks["test_plain"]]
            assert len(examples) > 1
            assert len({*own, *examples}) == len(own) + len(examples)
            names = [task.get_name() for task in own + examples]
            assert names == ["generator", "coroutine", "test_plain"] + (
                ["test_examples"] * len(examples)
            )
        """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(passed=3)


@pytest.mark.parametrize("phase", ["fixture", "setup", "test"])
@pytest.mark.parametrize("library", _LIBRARIES)
def test_child_failing_cancels_the_running_phase(
    pytester: Pytester, library: str, phase: str
):
    """
    Issues #1083 and #1191. A child failing during the fixture's own setup makes
    its group the setup error, and the loop goes on. A child failing at the yield
    cancels the dependent's setup or the test running at the time; the group is
    reported at the fixture's teardown, and later tests on the loop are refused.
    """
    group, start = _task_group(library, "fail", "forever")
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = module")
    pytester.makepyfile(dedent(f"""\
        import asyncio
        import {library}
        import pytest
        import pytest_asyncio

        @pytest_asyncio.fixture
        async def service():
            trigger = asyncio.Event()

            async def fail():
                await trigger.wait()
                raise RuntimeError("background service failed")

            async def forever():
                try:
                    await asyncio.Event().wait()
                finally:
                    print("SIBLING CANCELLED")

            async with {group} as group:
                {start}
                if {phase!r} == "fixture":
                    trigger.set()
                    await asyncio.Event().wait()
                yield trigger

        @pytest_asyncio.fixture
        async def dependent(service):
            if {phase!r} == "setup":
                service.set()
                await asyncio.Event().wait()
            yield service

        @pytest.mark.asyncio(loop_scope="module")
        async def test_service(dependent):
            dependent.set()
            await asyncio.Event().wait()

        @pytest.mark.asyncio(loop_scope="module")
        async def test_after():
            await asyncio.sleep(0)
        """))
    result = pytester.runpytest_subprocess("--asyncio-mode=strict", "-s", timeout=30)
    group_error = [
        "*ExceptionGroup: unhandled errors in a TaskGroup (1 sub-exception)*",
        "*RuntimeError: background service failed*",
    ]
    if phase == "fixture":
        result.assert_outcomes(errors=1, passed=1)
        lines = ["*ERROR at setup of test_service*", *group_error]
    elif phase == "setup":
        result.assert_outcomes(errors=2, failed=1)
        lines = [
            "*ERROR at setup of test_service*",
            *_fixture_cancelled("setup"),
            "*ERROR at teardown of test_service*",
            *group_error,
            "*_ test_after _*",
            _REFUSED,
        ]
    else:
        result.assert_outcomes(failed=2, errors=1)
        lines = [
            "*ERROR at teardown of test_service*",
            *group_error,
            "*_ test_service _*",
            *_CANCELLED,
            "*_ test_after _*",
            _REFUSED,
        ]
    result.stdout.fnmatch_lines(lines)
    assert "SIBLING CANCELLED" in result.stdout.str()


@pytest.mark.parametrize("library", _LIBRARIES)
def test_child_failing_between_tests_refuses_the_next_test(
    pytester: Pytester, library: str
):
    """
    A child failing while no test runs cancels the fixture's task at its yield:
    later tests on the loop are refused, and its teardown reports the error.
    """
    group, start = _task_group(library, "fail")
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = module")
    pytester.makepyfile(dedent(f"""\
        import asyncio
        import {library}
        import pytest
        import pytest_asyncio

        @pytest_asyncio.fixture(scope="module")
        async def service():
            trigger = asyncio.Event()

            async def fail():
                await trigger.wait()
                raise RuntimeError("background service failed")

            async with {group} as group:
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


def test_anyio_scopes_span_the_yield(pytester: Pytester):
    """Issue #1191: every AnyIO scope is exited in the task that entered it."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
        import asyncio
        import anyio
        import pytest
        import pytest_asyncio

        @pytest_asyncio.fixture
        async def scopes():
            with anyio.CancelScope(), anyio.fail_after(60), anyio.move_on_after(60):
                async with anyio.create_task_group():
                    yield

        @pytest.mark.asyncio
        async def test_scopes(scopes):
            await asyncio.sleep(0)
        """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(passed=1)


@pytest.mark.parametrize(
    ("library", "cleanup"),
    [
        pytest.param("asyncio", "await cleanup(connection)", marks=_REQUIRES_311),
        ("anyio", "await cleanup(connection)"),
        ("anyio", "with anyio.CancelScope(shield=True): await cleanup(connection)"),
    ],
    ids=["asyncio", "anyio", "anyio-shielded"],
)
def test_dependent_teardown_outside_cancelled_scope(
    pytester: Pytester, library: str, cleanup: str
):
    """
    The dependent is torn down first, in its own task rather than inside the
    cancelled scope of its parent, so its awaited cleanup completes with either
    library, shielded or not; the parent's teardown then reports the error.
    """
    group, start = _task_group(library, "fail")
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent(f"""\
        import asyncio
        import {library}
        import pytest
        import pytest_asyncio

        async def cleanup(connection):
            for _ in range(5):
                await asyncio.sleep(0)
                assert connection["open"]
            print("TRANSACTION CLEANED UP")

        @pytest_asyncio.fixture
        async def connection():
            state = {{"open": True, "fail": asyncio.Event()}}

            async def fail():
                await state["fail"].wait()
                raise RuntimeError("connection worker failed")

            try:
                async with {group} as group:
                    {start}
                    yield state
            finally:
                state["open"] = False
                print("CONNECTION CLOSED")

        @pytest_asyncio.fixture
        async def transaction(connection):
            try:
                yield connection
            finally:
                {cleanup}

        @pytest.mark.asyncio
        async def test_transaction(transaction):
            transaction["fail"].set()
            await asyncio.Event().wait()
        """))
    result = pytester.runpytest_subprocess("--asyncio-mode=strict", "-s", timeout=30)
    result.assert_outcomes(failed=1, errors=1)
    result.stdout.fnmatch_lines(
        [
            "*TRANSACTION CLEANED UP*",
            "*CONNECTION CLOSED*",
            "*ERROR at teardown of test_transaction*",
            "*RuntimeError: connection worker failed*",
        ]
    )


@pytest.mark.parametrize("library", _LIBRARIES)
def test_timeout_spanning_yield_expires_during_test(pytester: Pytester, library: str):
    """The raw CancelledError can be named by an xfail; the teardown is clean."""
    header, expire = _timeout(library)
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent(f"""\
        import asyncio
        import {library}
        import pytest
        import pytest_asyncio

        @pytest_asyncio.fixture
        async def deadline():
            {header} as scope:
                yield scope
            print("TEARDOWN CLEAN")

        @pytest.mark.xfail(raises=asyncio.CancelledError, strict=True)
        @pytest.mark.asyncio
        async def test_deadline(deadline):
            {expire}
            await asyncio.Event().wait()
        """))
    result = pytester.runpytest_subprocess("--asyncio-mode=strict", "-s", timeout=30)
    result.assert_outcomes(xfailed=1)
    assert "TEARDOWN CLEAN" in result.stdout.str()


@pytest.mark.parametrize("library", _LIBRARIES)
def test_timeout_spanning_yield_expires_while_idle(pytester: Pytester, library: str):
    """Every later test on the loop is refused, even once the timeout has exited."""
    header, expire = _timeout(library)
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = module")
    pytester.makepyfile(dedent(f"""\
        import asyncio
        import {library}
        import pytest
        import pytest_asyncio

        @pytest_asyncio.fixture
        async def deadline():
            {header} as scope:
                yield scope
            print("DEADLINE EXITED")

        @pytest_asyncio.fixture
        async def dependent(deadline):
            {expire}
            yield

        @pytest.mark.asyncio(loop_scope="module")
        async def test_deadline(dependent):
            pass

        @pytest.mark.asyncio(loop_scope="module")
        async def test_after():
            await asyncio.sleep(0)
        """))
    result = pytester.runpytest_subprocess("--asyncio-mode=strict", "-s", timeout=30)
    result.assert_outcomes(failed=2)
    result.stdout.fnmatch_lines(
        [
            "*DEADLINE EXITED*",
            "*_ test_deadline _*",
            _REFUSED,
            "*_ test_after _*",
            _REFUSED,
        ]
    )


def test_overlapping_parametrized_module_fixtures(pytester: Pytester):
    """
    A parametrized module fixture is torn down while a later one is still alive:
    each AnyIO scope is in a task of its own, so no LIFO order binds fixtures.
    """
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = module")
    pytester.makepyfile(dedent("""\
        import anyio
        import pytest
        import pytest_asyncio

        @pytest_asyncio.fixture(scope="module", params=[1, 2])
        async def first(request):
            async with anyio.create_task_group():
                print(f"OPEN FIRST {request.param}")
                yield request.param
                print(f"CLOSE FIRST {request.param}")

        @pytest_asyncio.fixture(scope="module")
        async def second():
            async with anyio.create_task_group():
                print("OPEN SECOND")
                yield
                print("CLOSE SECOND")

        @pytest.mark.asyncio(loop_scope="module")
        async def test_pair(first, second):
            print(f"TEST {first}")
        """))
    result = pytester.runpytest("--asyncio-mode=strict", "-s")
    result.assert_outcomes(passed=2)
    result.stdout.fnmatch_lines(
        [
            "*OPEN FIRST 1*",
            "*OPEN SECOND*",
            "*TEST 1*",
            "*CLOSE FIRST 1*",
            "*OPEN FIRST 2*",
            "*TEST 2*",
            "*CLOSE FIRST 2*",
            "*CLOSE SECOND*",
        ]
    )


@_REQUIRES_311
@pytest.mark.parametrize("failing", ["first", "second"])
def test_two_task_group_fixtures(pytester: Pytester, failing: str):
    """Groups are exited in LIFO order; the failing one reports at its teardown."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent(f"""\
        import asyncio
        import pytest
        import pytest_asyncio

        log = []
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
                    log.append("first teardown")

        @pytest_asyncio.fixture
        async def second(first):
            async with asyncio.TaskGroup() as group:
                group.create_task(child("second"))
                try:
                    yield
                finally:
                    log.append("second teardown")

        @pytest.mark.asyncio
        async def test_services(second):
            try:
                trigger.set()
                await asyncio.Event().wait()
            finally:
                log.append("test")

        def test_order():
            assert log == ["test", "second teardown", "first teardown"]
        """))
    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)
    result.assert_outcomes(passed=1, failed=1, errors=1)
    result.stdout.fnmatch_lines([f"*RuntimeError: {failing} child failed*"])


@pytest.mark.parametrize(
    ("body", "passed", "error"),
    [
        ("return; yield", 1, "*StopAsyncIteration*"),
        ("yield; yield", 2, "*ValueError: Async generator fixture yielded more than*"),
    ],
    ids=["never_yields", "yields_twice"],
)
def test_misbehaving_generators(pytester: Pytester, body: str, passed: int, error: str):
    """A generator that never yields or yields twice is reported; tests go on."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent(f"""\
        import pytest
        import pytest_asyncio

        @pytest_asyncio.fixture
        async def generator():
            {body}

        @pytest.mark.asyncio
        async def test_generator(generator):
            pass

        @pytest.mark.asyncio
        async def test_next():
            pass
        """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(passed=passed, errors=1)
    result.stdout.fnmatch_lines([error])


@_REQUIRES_311
@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(
            "class CustomLoop(asyncio.SelectorEventLoop): pass\nfactory = CustomLoop",
            id="custom",
        ),
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
def test_loop_factories_in_debug_mode(pytester: Pytester, factory: str):
    """A fixture's group cancels the test and refuses the rest, on any loop."""
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
        async def test_debug():
            assert asyncio.get_running_loop().get_debug()

        @pytest.mark.asyncio(loop_scope="module")
        async def test_service(service):
            service.set()
            await asyncio.Event().wait()

        @pytest.mark.asyncio(loop_scope="module")
        async def test_after():
            await asyncio.sleep(0)
        """))
    result = pytester.runpytest_subprocess(
        "--asyncio-mode=strict", "-W", "error", timeout=30
    )
    result.assert_outcomes(passed=1, failed=2, errors=1)
    result.stdout.fnmatch_lines(
        ["*RuntimeError: background service failed*", "*_ test_after _*", _REFUSED]
    )
    output = result.stdout.str() + result.stderr.str()
    for noise in ("Task was destroyed", "never retrieved", "never awaited"):
        assert noise not in output


def test_repeated_cancellation_preserves_cleanup_error(pytester: Pytester):
    """
    A cancelled teardown is an error caused by the CancelledError; the parent
    is still torn down afterwards and its own error reported.
    """
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
        import asyncio
        import pytest
        import pytest_asyncio

        @pytest_asyncio.fixture
        async def parent():
            try:
                yield
            finally:
                print("PARENT CLEANUP RAN")
                raise RuntimeError("parent cleanup failed")

        @pytest_asyncio.fixture
        async def child(parent):
            try:
                yield
            finally:
                for _ in range(3):
                    asyncio.current_task().cancel()
                    await asyncio.sleep(0)
                print("CHILD CLEANUP RAN")

        @pytest.mark.asyncio
        async def test_cancel(child):
            asyncio.current_task().cancel()
            await asyncio.Event().wait()
        """))
    result = pytester.runpytest_subprocess("--asyncio-mode=strict", "-s", timeout=30)
    result.assert_outcomes(failed=1, errors=1)
    result.stdout.fnmatch_lines(
        ["*PARENT CLEANUP RAN*", "*RuntimeError: parent cleanup failed*"]
    )
    result.stdout.fnmatch_lines(_fixture_cancelled("teardown"))
    assert "CHILD CLEANUP RAN" not in result.stdout.str()


def test_no_accumulation_over_many_tests(pytester: Pytester):
    """Jobs leave no coroutines, futures, tasks or contexts behind."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = module")
    pytester.makepyfile(dedent("""\
        import asyncio
        import contextvars
        import gc
        import types
        import pytest
        import pytest_asyncio

        N = 200
        samples = {}

        def sample():
            gc.collect()
            objects = gc.get_objects()
            kinds = (types.CoroutineType, asyncio.Future, contextvars.Context)
            counts = (sum(isinstance(obj, kind) for obj in objects) for kind in kinds)
            return (*counts, len(asyncio.all_tasks()))

        @pytest_asyncio.fixture
        async def resource():
            await asyncio.sleep(0)
            yield object()

        @pytest.mark.asyncio(loop_scope="module")
        @pytest.mark.parametrize("i", range(N))
        async def test_many(i, resource):
            if i in (10, N - 1):
                samples[i] = sample()
            await asyncio.sleep(0)

        def test_samples():
            assert samples[10] == samples[N - 1]
        """))
    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)
    result.assert_outcomes(passed=201)


def test_sync_tests_and_fixtures_with_async_fixtures(pytester: Pytester):
    """Sync code may use async fixtures; an async test cannot request them."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
        import asyncio
        import pytest
        import pytest_asyncio

        calls = []

        @pytest_asyncio.fixture
        async def generator():
            calls.append(asyncio.current_task())
            yield "generator"
            calls.append(asyncio.current_task())

        @pytest.fixture
        def sync_fixture(generator):
            return generator.upper()

        def test_sync(generator, sync_fixture):
            assert (generator, sync_fixture) == ("generator", "GENERATOR")

        @pytest.mark.asyncio
        async def test_dynamic(request):
            request.getfixturevalue("generator")

        def test_calls():
            # The refused request never set the fixture up.
            assert len(calls) == 2
            assert calls[0] is calls[1]
        """))
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(passed=2, failed=1)
    result.stdout.fnmatch_lines(
        ["*cannot start an async fixture or test while the event loop is running*"]
    )
