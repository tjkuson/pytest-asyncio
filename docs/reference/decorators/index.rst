.. _decorators/pytest_asyncio_fixture:

==========
Decorators
==========
The ``@pytest_asyncio.fixture`` decorator allows coroutines and async generator functions to be used as pytest fixtures.

The decorator takes all arguments supported by `@pytest.fixture`.
Additionally, ``@pytest_asyncio.fixture`` supports the *loop_scope* keyword argument, which selects the event loop in which the fixture is run (see :ref:`concepts/event_loops`).
The default event loop scope is *function* scope.
Possible loop scopes are *session,* *package,* *module,* *class,* and *function*.

The *loop_scope* of a fixture can be chosen independently from its caching *scope*.
However, the event loop scope must be larger or the same as the fixture's caching scope.
In other words, it's possible to reevaluate an async fixture multiple times within the same event loop, but it's not possible to switch out the running event loop in an async fixture.

The setup and teardown of an async generator fixture run in the same asyncio task, which stays alive across the fixture's ``yield``.
This allows an ``asyncio.TaskGroup``, an ``asyncio.timeout`` or a cancel scope to span the ``yield`` (see :ref:`concepts/tasks`).

Examples:

.. include:: pytest_asyncio_fixture_example.py
    :code: python

*auto* mode automatically converts coroutines and async generator functions declared with the standard ``@pytest.fixture`` decorator to pytest-asyncio fixtures.

.. _decorators/pytest_asyncio_fixture/cancellation:

Cancellation and errors
-----------------------

When an async generator fixture is cancelled while it waits at its ``yield``, for example by a task group spanning the ``yield`` whose child failed, the test or fixture setup running at the time is cancelled.
If the ``CancelledError`` escapes the test, the test fails with it, with a note explaining the cause on Python 3.11 and later; a test may catch it, as any asyncio code may.
If ``CancelledError`` escapes a fixture's setup or teardown, pytest-asyncio reports a ``PytestAsyncioError`` instead, so that pytest caches and reports it like any other fixture error.
The fixture cancelled at its ``yield`` itself reports whatever its teardown raises: a task group spanning the ``yield`` reports its children's errors, as it does anywhere; a cancel scope may exit without an error.
A test or fixture setup that cannot start because the event loop no longer accepts new work fails with ``RuntimeError``.

Limitations:

* A cancellation that reaches a fixture's task at its ``yield`` stops all new tests and fixture setups on the whole event loop, including tests that do not use that fixture.
* A timeout spanning a ``yield`` (``asyncio.timeout``, or ``anyio.fail_after``) that expires while a test runs cancels the test; it does not raise ``TimeoutError`` at the fixture's teardown, because the fixture's teardown exits the timeout without an exception.
* An AnyIO cancel scope that is cancelled while spanning a ``yield`` keeps cancelling the fixture's waiting task until the fixture's teardown exits it, which costs CPU time for as long as the event loop runs in between, including the cancelled test's cleanup and dependent fixtures' teardowns.
  Cleanup is not prevented and the errors are reported; the cost is a limitation of pytest-asyncio's integration with AnyIO's cancellation.
* A test's failure is reported by pytest, not by the test's task: a done callback on the task sees it finish without an exception, and a task cancelled while a failure is raised sees the cancellation.
