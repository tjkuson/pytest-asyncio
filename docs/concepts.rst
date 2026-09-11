========
Concepts
========

.. _concepts/event_loops:

asyncio event loops
===================
In order to understand how pytest-asyncio works, it helps to understand how pytest collectors work.
If you already know about pytest collectors, please :ref:`skip ahead <pytest-asyncio-event-loops>`.
Otherwise, continue reading.
Let's assume we have a test suite with a file named *test_all_the_things.py* holding a single test, async or not:

.. include:: concepts_function_scope_example.py
    :code: python

The file *test_all_the_things.py* is a Python module with a Python test function.
When we run pytest, the test runner descends into Python packages, modules, and classes, in order to find all tests, regardless whether the tests will run or not.
This process is referred to as *test collection* by pytest.
In our particular example, pytest will find our test module and the test function.
We can visualize the collection result by running ``pytest --collect-only``::

    <Module test_all_the_things.py>
      <Function test_runs_in_a_loop>

The example illustrates that the code of our test suite is hierarchical.
Pytest uses so called *collectors* for each level of the hierarchy.
Our contrived example test suite uses the *Module* and *Function* collectors, but real world test code may contain additional hierarchy levels via the *Package* or *Class* collectors.
There's also a special *Session* collector at the root of the hierarchy.
You may notice that the individual levels resemble the possible `scopes of a pytest fixture. <https://docs.pytest.org/en/7.4.x/how-to/fixtures.html#scope-sharing-fixtures-across-classes-modules-packages-or-session>`__

.. _pytest-asyncio-event-loops:

Pytest-asyncio provides one asyncio event loop for each pytest collector.
By default, each test runs in the event loop provided by the *Function* collector, i.e. tests use the loop with the narrowest scope.
This gives the highest level of isolation between tests.
If two or more tests share a common ancestor collector, the tests can be configured to run in their ancestor's loop by passing the appropriate *loop_scope* keyword argument to the *asyncio* mark.
For example, the following two tests use the asyncio event loop provided by the *Module* collector:

.. include:: concepts_module_scope_example.py
    :code: python

It's highly recommended for neighboring tests to use the same event loop scope.
For example, all tests in a class or module should use the same scope.
Assigning neighboring tests to different event loop scopes is discouraged as it can make test code hard to follow.

Test discovery modes
====================

Pytest-asyncio provides two modes for test discovery, *strict* and *auto*.
This can be set through Pytest's ``--asyncio-mode`` command line flag,
or through the configuration file.

.. tabs::

   .. group-tab:: Strict mode

      .. code-block:: toml

         [tool.pytest.ini_options]
         asyncio_mode = "strict"

      In strict mode pytest-asyncio will only run tests that have the *asyncio* marker
      and will only evaluate async fixtures decorated with ``@pytest_asyncio.fixture``.
      Test functions and fixtures without these markers and decorators will not be
      handled by pytest-asyncio.

      This mode is intended for projects that want to support multiple asynchronous
      programming libraries as it allows pytest-asyncio to coexist with other async
      testing plugins in the same codebase.

      Pytest automatically enables installed plugins. As a result pytest plugins
      need to coexist peacefully in their default configuration. This is why strict
      mode is the default mode.

   .. group-tab:: Auto mode

      .. code-block:: toml

         [tool.pytest.ini_options]
         asyncio_mode = "auto"

      In *auto* mode pytest-asyncio automatically adds the *asyncio* marker to all
      asynchronous test functions. It will also take ownership of all async fixtures,
      regardless of whether they are decorated with ``@pytest.fixture`` or
      ``@pytest_asyncio.fixture``.

      This mode is intended for projects that use *asyncio* as their only asynchronous
      programming library. Auto mode makes for the simplest test and fixture
      configuration and is the recommended default.

      If you intend to support multiple asynchronous programming libraries,
      e.g. *asyncio* and *trio*, strict mode will be the preferred option.

.. _concepts/concurrent_execution:

Test execution and concurrency
==============================

pytest-asyncio runs async tests sequentially, just like how pytest runs synchronous tests. Each asynchronous test runs within its assigned event loop. For example, consider the following two tests:

.. include:: concepts_concurrent_execution_example.py
    :code: python

This sequential execution is intentional and important for maintaining test isolation. Running tests concurrently could introduce race conditions and side effects where one test could interfere with another, making test results unreliable and difficult to debug.

.. _concepts/tasks:

Tasks, cancellation and context
===============================

pytest-asyncio runs all coroutines that belong to an event loop in a single asyncio task: the setup and teardown of async fixtures, and the tests.
The task that sets up an async generator fixture is therefore the task that tears it down.
A fixture can wrap its ``yield`` in an ``asyncio.TaskGroup``, an ``asyncio.timeout``, or the cancel scope of a library such as AnyIO, so that background tasks run for as long as the fixture is alive:

.. include:: concepts_task_group_fixture_example.py
    :code: python

Cancellation
------------

If a task in the group fails while a test runs, the group cancels the task running the test.
The test fails with ``asyncio.CancelledError``, and the group raises the failure as an ``ExceptionGroup`` when the fixture is torn down, which pytest reports as an error at teardown of that fixture.
A cancellation can also arrive while no test or fixture is running, for example between two tests sharing a module-scoped loop.

Either way, the cancellation ends the normal work of that event loop.
pytest-asyncio still runs fixture teardowns in it, so that the task group, cancel scope or timeout can exit and report what happened, but refuses to run further tests and fixture setups in the loop: they fail with an ``asyncio.CancelledError`` explaining the refusal.
The same happens for any other cancellation of the task that a test or fixture lets propagate, for example a test cancelling its own task.
With the default function-scoped loop, nothing else would have run in the loop anyway.
With a wider loop scope, the remaining tests sharing the loop are refused: pytest-asyncio does not know which fixture's background work failed, and any of those tests may depend on it.

Context variables
-----------------

The task of an event loop has one ``contextvars`` context, in which all async fixtures and tests of the loop run.
Tests with a function-scoped loop therefore start from a fresh context, whereas tests sharing a module, class, package or session loop share theirs: a context variable set by one test is seen by the tests that follow in that loop.
Background tasks started by tests and fixtures run in copies of the context, as always in asyncio.

pytest-asyncio moves values between that context and the context of synchronous pytest code by two rules:

* Before an async fixture or test runs, the variables set in the synchronous context are applied to the loop's context, so that async code sees what synchronous fixtures and tests have set.
  Afterwards, each applied variable is restored to its previous value in the loop's context, unless the fixture or test left it with a different value.
  Values from the synchronous context therefore take precedence for as long as they are set there.
* The variables an async fixture changes during its setup are applied to the synchronous context until the fixture is torn down, so that synchronous fixtures and tests depending on it see them, and so does later async code.

Both rules compare values: pytest-asyncio cannot see assignments, and setting a variable to the value it already has is not a change.
In particular, an async fixture set up more than once in the same loop, such as a function-scoped fixture with a module-scoped loop, propagates a value to synchronous code only the first time, because the loop's context keeps the value between setups.
Reset the variable after the ``yield`` of an async generator fixture if that matters.

The loop scope of a fixture or test is separate from the caching scope of the fixture.
For example, a module-scoped fixture used by tests with function-scoped loops runs in the module's loop, so its context is not the context of any test.
Its values reach the tests through the rules above, not through a shared context.

Interruption
------------

When a test or fixture is interrupted, for example by Ctrl-C or by the signal of a timeout plugin, pytest-asyncio cancels the running coroutine and waits for it to finish, so that ``finally`` blocks and async context managers run before pytest tears down the fixtures the coroutine may be using.
A second interruption abandons the coroutine.
If a task group, cancel scope or timeout cancels the task in the meantime, the loop is treated as cancelled, as described above.
On Python 3.10, asyncio cannot tell such a cancellation from the one pytest-asyncio requested (``Task.uncancel`` was added in Python 3.11), so there an interrupted test or fixture always leaves the loop in that state.

Limitations
-----------

* Task groups, cancel scopes and timeouts spanning a ``yield`` must exit in the reverse order in which they were entered.
  pytest may set up a second instance of a parametrized fixture with a wider caching scope while the first is still alive and tear down the first one afterwards.
  An AnyIO cancel scope spanning the ``yield`` of such a fixture fails at teardown, as it does with AnyIO's own pytest plugin.
* AnyIO delivers a cancellation to every ``await`` inside a cancelled scope until the scope exits.
  If a fixture depends on a fixture whose AnyIO scope was cancelled, any ``await`` in its teardown is cancelled, and the teardown is reported as an error.
  Shield such cleanup with ``anyio.CancelScope(shield=True)``.
* A cancellation that a test or fixture catches and does not re-raise never reaches pytest-asyncio.
  A later test that uses the fixture whose background task failed may then wait forever for that task.
  Let ``asyncio.CancelledError`` propagate.
* A timeout spanning a ``yield`` that expires during a test fails the test with ``asyncio.CancelledError``; it does not raise ``TimeoutError`` at teardown, because the fixture's teardown exits the timeout without an exception.
