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
pytest-asyncio then refuses to run further tests and fixture setups in that loop until the task group, cancel scope or timeout that requested the cancellation has exited, which usually means until the fixture is torn down.
Refused tests fail with an ``asyncio.CancelledError`` that explains the refusal.
Fixture teardowns always run, so that the requesting scope can exit.
With the default function-scoped loop nothing else runs in the loop, so the refusal only ever affects tests that share a loop with the failed fixture.

On Python 3.10, asyncio cannot report whether a cancellation request is still unresolved (``asyncio.Task.cancelling`` was added in Python 3.11).
Once a cancellation has reached the task, a loop on Python 3.10 therefore runs only fixture teardowns until it is closed.

Context variables
-----------------

Each event loop has one ``contextvars`` context, shared by all async fixtures and tests that run in the loop:

* Tests with a function-scoped loop start from a fresh context.
  Tests sharing a module, class, package or session loop share their context, so a context variable set by one test is seen by the tests that follow in that loop.
* Async code sees the context variables set by synchronous fixtures and tests.
* Context variables set by an async fixture are seen by the synchronous fixtures and tests that depend on it, and are restored in synchronous code when the fixture is torn down.
  In the event loop they stay set: an async generator fixture that should not leak a value to the async code that runs after it is torn down resets the variable after its ``yield``.

The loop scope of a fixture or test is separate from the caching scope of the fixture.
For example, a module-scoped fixture used by tests with function-scoped loops runs in the module's loop, so its context is not the context of any test.
Its value is inherited through the mechanism above, not through a shared context.

Interruption
------------

When a test or fixture is interrupted, for example by Ctrl-C, pytest-asyncio cancels the running coroutine and waits for it to finish, so that ``finally`` blocks and async context managers run before pytest tears down the fixtures the coroutine may be using.
A second interruption abandons the coroutine.

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
