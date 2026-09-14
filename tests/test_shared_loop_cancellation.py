"""
The other tests on a shared event loop after a cancellation.

Each test and fixture runs in a task of its own, so a cancellation ends the
test or fixture it reached and nothing else: the later tests on the loop
run. The exception is an async generator fixture cancelled as it waits at
its yield, or pytest-asyncio's own task: the loop then runs nothing but
fixture teardowns until it closes, and a refused test or fixture setup
fails with a RuntimeError saying so.
"""

from __future__ import annotations

import sys
from textwrap import dedent

import pytest
from pytest import Pytester

_REQUIRES_311 = pytest.mark.skipif(sys.version_info < (3, 11), reason="needs 3.11")
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


@pytest.mark.parametrize(
    "after",
    [
        pytest.param("await asyncio.sleep(0)", id="propagated"),
        pytest.param("pass", id="at_return"),
    ],
)
def test_a_test_cancelling_its_own_task_fails_alone(pytester: Pytester, after: str):
    """Its fixture is torn down, and the later tests on the loop run."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makeconftest(dedent(_EVENTS_CONFTEST))
    pytester.makepyfile(dedent(f"""\
        import asyncio

        import pytest
        import pytest_asyncio
        from conftest import events

        @pytest_asyncio.fixture(loop_scope="module")
        async def resource():
            yield
            await asyncio.sleep(0)
            events.append("resource closed")

        @pytest.mark.asyncio(loop_scope="module")
        async def test_cancel(resource):
            asyncio.current_task().cancel()
            {after}

        @pytest.mark.asyncio(loop_scope="module")
        async def test_next():
            events.append("next ran")

        @pytest.mark.asyncio(loop_scope="module")
        async def test_with_fixture(resource):
            events.append("with fixture ran")
        """))
    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)
    result.assert_outcomes(failed=1, passed=2)
    result.stdout.fnmatch_lines(["*_ test_cancel _*", "*CancelledError*"])
    assert _read_events(pytester) == [
        "resource closed",
        "next ran",
        "with fixture ran",
        "resource closed",
    ]


def test_a_test_suppressing_its_own_cancellation_passes(pytester: Pytester):
    """Cancelling its own task is the test's business: nothing else notices."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
        import asyncio
        import contextlib

        import pytest

        @pytest.mark.asyncio(loop_scope="module")
        async def test_cancel():
            asyncio.current_task().cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.sleep(0)

        @pytest.mark.asyncio(loop_scope="module")
        async def test_next():
            pass
        """))
    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)
    result.assert_outcomes(passed=2)


def test_an_error_raised_after_a_self_cancellation_is_the_failure(pytester: Pytester):
    """The error wins over the pending cancellation, as in any task."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest

        @pytest.mark.asyncio(loop_scope="module")
        async def test_cancel_then_raise():
            asyncio.current_task().cancel()
            raise RuntimeError("raised after the cancellation")

        @pytest.mark.asyncio(loop_scope="module")
        async def test_next():
            pass
        """))
    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)
    result.assert_outcomes(failed=1, passed=1)
    result.stdout.fnmatch_lines(
        [
            "*_ test_cancel_then_raise _*",
            "*RuntimeError: raised after the cancellation*",
        ]
    )


def test_a_keyboard_interrupt_raised_after_a_self_cancellation_interrupts_the_session(
    pytester: Pytester,
):
    """The interruption wins over the pending cancellation: the session stops."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest

        @pytest.mark.asyncio(loop_scope="module")
        async def test_cancel_then_interrupt():
            asyncio.current_task().cancel()
            raise KeyboardInterrupt

        @pytest.mark.asyncio(loop_scope="module")
        async def test_next():
            pass
        """))
    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)
    assert result.ret == pytest.ExitCode.INTERRUPTED
    result.assert_outcomes()


def test_a_test_cancelling_every_task_makes_the_loop_refuse_later_tests(
    pytester: Pytester,
):
    """
    asyncio.all_tasks() includes the fixture waiting at its yield and
    pytest-asyncio's own task: the loop runs nothing but teardowns after that.
    """
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makeconftest(dedent(_EVENTS_CONFTEST))
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest
        import pytest_asyncio
        from conftest import events

        @pytest_asyncio.fixture(loop_scope="module")
        async def resource():
            yield
            await asyncio.sleep(0)
            events.append("resource closed")

        @pytest.mark.asyncio(loop_scope="module")
        async def test_cancel_all(resource):
            for task in asyncio.all_tasks():
                task.cancel()
            await asyncio.sleep(0)

        @pytest.mark.asyncio(loop_scope="module")
        async def test_next():
            events.append("next ran")

        @pytest.mark.asyncio(loop_scope="module")
        async def test_with_fixture(resource):
            events.append("with fixture ran")
        """))
    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)
    result.assert_outcomes(failed=2, errors=1)
    result.stdout.fnmatch_lines(
        [
            "*ERROR at setup of test_with_fixture*",
            _REFUSED,
            "*_ test_cancel_all _*",
            "*CancelledError*",
            "*_ test_next _*",
            _REFUSED,
        ]
    )
    assert _read_events(pytester) == ["resource closed"]


def test_a_coroutine_fixture_cancelled_during_its_setup_errors_only_its_test(
    pytester: Pytester,
):
    """The cancellation is reported as an ordinary setup error; the loop goes on."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest
        import pytest_asyncio

        @pytest_asyncio.fixture(loop_scope="module")
        async def cancelled():
            asyncio.current_task().cancel()
            await asyncio.sleep(0)

        @pytest.mark.asyncio(loop_scope="module")
        async def test_cancelled(cancelled):
            pass

        @pytest.mark.asyncio(loop_scope="module")
        async def test_next():
            pass
        """))
    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)
    result.assert_outcomes(errors=1, passed=1)
    result.stdout.fnmatch_lines(
        [
            "*ERROR at setup of test_cancelled*",
            "*PytestAsyncioError: The setup of the async fixture was cancelled.*",
        ]
    )


def test_a_generator_fixture_cancelled_during_its_setup_errors_only_its_test(
    pytester: Pytester,
):
    """A generator that never reached its yield is reported the same way."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest
        import pytest_asyncio

        @pytest_asyncio.fixture(loop_scope="module")
        async def cancelled():
            asyncio.current_task().cancel()
            await asyncio.sleep(0)
            yield

        @pytest.mark.asyncio(loop_scope="module")
        async def test_cancelled(cancelled):
            pass

        @pytest.mark.asyncio(loop_scope="module")
        async def test_next():
            pass
        """))
    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)
    result.assert_outcomes(errors=1, passed=1)
    result.stdout.fnmatch_lines(
        [
            "*ERROR at setup of test_cancelled*",
            "*PytestAsyncioError: The setup of the async fixture was cancelled.*",
        ]
    )


def test_a_module_fixture_cancelled_during_its_setup_errors_each_of_its_tests(
    pytester: Pytester,
):
    """The setup error is cached for the fixture's other users; the loop goes on."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest
        import pytest_asyncio

        @pytest_asyncio.fixture(scope="module", loop_scope="module")
        async def cancelled():
            asyncio.current_task().cancel()
            await asyncio.sleep(0)
            yield

        @pytest.mark.asyncio(loop_scope="module")
        async def test_first_user(cancelled):
            pass

        @pytest.mark.asyncio(loop_scope="module")
        async def test_second_user(cancelled):
            pass

        @pytest.mark.asyncio(loop_scope="module")
        async def test_unrelated():
            pass
        """))
    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)
    result.assert_outcomes(errors=2, passed=1)
    result.stdout.fnmatch_lines(
        [
            "*ERROR at setup of test_first_user*",
            "*asyncio.exceptions.CancelledError*",
            "*PytestAsyncioError: The setup of the async fixture was cancelled.*",
            "*ERROR at setup of test_second_user*",
            "*asyncio.exceptions.CancelledError*",
            "*PytestAsyncioError: The setup of the async fixture was cancelled.*",
        ]
    )


@_REQUIRES_311
def test_a_fixture_cancelled_at_its_yield_makes_its_loop_refuse_new_work(
    pytester: Pytester,
):
    """
    Later tests and fixture setups on the loop are refused; teardowns run;
    a test on a loop of its own and the next module's loop are unaffected.
    """
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makeconftest(dedent(_EVENTS_CONFTEST))
    pytester.makepyfile(
        test_a_service=dedent("""\
            import asyncio

            import pytest
            import pytest_asyncio
            from conftest import events

            @pytest_asyncio.fixture(scope="module", loop_scope="module")
            async def service():
                trigger = asyncio.Event()

                async def fail():
                    await trigger.wait()
                    raise RuntimeError("background service failed")

                async with asyncio.TaskGroup() as group:
                    group.create_task(fail())
                    yield trigger

            @pytest_asyncio.fixture(loop_scope="module")
            async def unrelated():
                yield
                await asyncio.sleep(0)
                events.append("unrelated closed")

            @pytest.mark.asyncio(loop_scope="module")
            async def test_trigger(service, unrelated):
                service.set()
                await asyncio.Event().wait()

            @pytest.mark.asyncio(loop_scope="module")
            async def test_uses_service(service):
                events.append("used the service")

            @pytest.mark.asyncio(loop_scope="function")
            async def test_own_loop():
                events.append("own loop ran")

            @pytest.mark.asyncio(loop_scope="module")
            async def test_unrelated(unrelated):
                events.append("unrelated ran")
            """),
        test_b_next_module=dedent("""\
            import pytest
            from conftest import events

            @pytest.mark.asyncio(loop_scope="module")
            async def test_fresh_loop():
                events.append("next module ran")
            """),
    )
    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)
    result.assert_outcomes(failed=2, passed=2, errors=2)
    result.stdout.fnmatch_lines(
        [
            "*ERROR at setup of test_unrelated*",
            _REFUSED,
            "*ERROR at teardown of test_unrelated*",
            "*RuntimeError: background service failed*",
            "*_ test_trigger _*",
            "*CancelledError*",
            "*_ test_uses_service _*",
            _REFUSED,
        ]
    )
    assert _read_events(pytester) == [
        "unrelated closed",
        "own loop ran",
        "next module ran",
    ]


def test_a_test_stopping_the_loop_fails_and_the_loop_serves_the_later_tests(
    pytester: Pytester,
):
    """The stopped loop interrupts the wait: the test is cancelled and fails."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makeconftest(dedent(_EVENTS_CONFTEST))
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest
        import pytest_asyncio
        from conftest import events

        @pytest_asyncio.fixture(loop_scope="module")
        async def resource():
            yield
            await asyncio.sleep(0)
            events.append("resource closed")

        @pytest.mark.asyncio(loop_scope="module")
        async def test_stop(resource):
            asyncio.get_running_loop().stop()
            await asyncio.sleep(0)
            events.append("test continued")

        @pytest.mark.asyncio(loop_scope="module")
        async def test_next(resource):
            events.append("next ran")
        """))
    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)
    result.assert_outcomes(failed=1, passed=1)
    result.stdout.fnmatch_lines(
        [
            "*_ test_stop _*",
            "*RuntimeError: Event loop stopped before Future completed*",
        ]
    )
    assert _read_events(pytester) == ["resource closed", "next ran", "resource closed"]


def test_a_sync_test_closing_the_shared_loop_fails_the_later_async_tests(
    pytester: Pytester,
):
    """They fail with asyncio's error, and pytest-asyncio warns as it closes."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest

        @pytest.mark.asyncio(loop_scope="module")
        async def test_first():
            pass

        def test_close_loop():
            asyncio.get_event_loop().close()

        @pytest.mark.asyncio(loop_scope="module")
        async def test_after_close():
            pass
        """))
    result = pytester.runpytest_subprocess(
        "--asyncio-mode=strict", "-W", "default", timeout=30
    )
    result.assert_outcomes(passed=2, failed=1)
    result.stdout.fnmatch_lines(
        [
            "*_ test_after_close _*",
            "*RuntimeError: Event loop is closed*",
            "*An exception occurred while pytest-asyncio closed the event loop*",
        ]
    )


def test_a_background_task_runs_until_its_loop_closes(pytester: Pytester):
    """A task a fixture leaves running is cancelled when the loop closes."""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makeconftest(dedent(_EVENTS_CONFTEST))
    pytester.makepyfile(dedent("""\
        import asyncio

        import pytest
        import pytest_asyncio
        from conftest import events

        async def run_forever():
            try:
                await asyncio.Event().wait()
            finally:
                events.append("background task cancelled")

        @pytest_asyncio.fixture(scope="module", loop_scope="module")
        async def background():
            return asyncio.create_task(run_forever())

        @pytest.mark.asyncio(loop_scope="module")
        async def test_started(background):
            await asyncio.sleep(0)

        @pytest.mark.asyncio(loop_scope="module")
        async def test_still_running(background):
            assert not background.done()
            events.append("still running in the second test")
        """))
    result = pytester.runpytest_subprocess(
        "--asyncio-mode=strict", "-W", "error", timeout=30
    )
    result.assert_outcomes(passed=2)
    assert _read_events(pytester) == [
        "still running in the second test",
        "background task cancelled",
    ]
    assert "Task was destroyed" not in result.stdout.str() + result.stderr.str()


def test_a_sync_test_can_drive_the_loop_while_an_anyio_scope_is_cancelled(
    pytester: Pytester,
):
    """
    AnyIO keeps cancelling the fixture's task while it waits at its yield;
    a coroutine the sync test runs on the loop still completes.
    """
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(dedent("""\
        import asyncio

        import anyio
        import pytest
        import pytest_asyncio

        @pytest_asyncio.fixture(scope="module", loop_scope="module")
        async def service():
            trigger = asyncio.Event()

            async def fail():
                await trigger.wait()
                raise RuntimeError("background service failed")

            async with anyio.create_task_group() as group:
                group.start_soon(fail)
                yield trigger

        @pytest.mark.asyncio(loop_scope="module")
        async def test_cancelled(service):
            service.set()
            await asyncio.Event().wait()

        def test_drives_the_loop(service):
            loop = asyncio.get_event_loop()
            assert loop.run_until_complete(asyncio.sleep(0, result=42)) == 42

        @pytest.mark.asyncio(loop_scope="module")
        async def test_next(service):
            pass
        """))
    result = pytester.runpytest_subprocess("--asyncio-mode=strict", timeout=30)
    result.assert_outcomes(failed=2, passed=1, errors=1)
    result.stdout.fnmatch_lines(
        [
            "*ERROR at teardown of test_next*",
            "*RuntimeError: background service failed*",
            "*_ test_next _*",
            _REFUSED,
        ]
    )
