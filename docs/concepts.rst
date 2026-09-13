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

pytest-asyncio runs each async fixture and each test in an asyncio task of its own, named after the fixture or test.
The task of an async generator fixture stays alive across the fixture's ``yield``: it runs the setup, waits while the fixture is in use, and runs the teardown.
The task that enters a task group, cancel scope or timeout before the ``yield`` is therefore the task that exits it, so a fixture can wrap its ``yield`` in an ``asyncio.TaskGroup``, an ``asyncio.timeout``, or the cancel scope of a library such as AnyIO, and background tasks run for as long as the fixture is alive:

.. include:: concepts_task_group_fixture_example.py
    :code: python

The tasks of the async generator fixtures of an event loop are the children of an ``asyncio.TaskGroup`` (on Python 3.11 and later) entered by a task of pytest-asyncio's own, named ``pytest-asyncio``, which lives as long as the loop and joins them as it exits.
A fixture's errors go to pytest, at the setup or teardown of that fixture, so one fixture's failure does not cancel the others.
The task of a test is not a child of that group: a test that cancels every task of its loop and waits for them to end could then never end, since the group would be waiting for the test.

Cancellation
------------

If a task in the group fails, the group cancels the fixture's task, which is waiting at the ``yield``.
pytest-asyncio then cancels the test or fixture setup running at the time, which fails with ``asyncio.CancelledError``, and ends the normal work of that event loop: fixture teardowns still run, so that the group can exit and report what happened, but further tests and fixture setups in the loop are refused.
A refused test fails with an ``asyncio.CancelledError`` explaining the refusal; a refused or cancelled fixture setup or teardown errors with a ``PytestAsyncioError`` carrying the cause, so that pytest handles it like any other fixture error.
The group raises the failure as an ``ExceptionGroup`` when the fixture is torn down, which pytest reports as an error at teardown of that fixture.

With the default function-scoped loop, nothing else would have run in the loop anyway.
With a wider loop scope, the remaining tests sharing the loop are refused: pytest-asyncio does not know which of them depend on the failed fixture.
A cancellation of pytest-asyncio's own ``pytest-asyncio`` task, by a test cancelling every task of the loop, say, ends the normal work of the loop in the same way; its task group then waits for the fixtures' tasks, which end when pytest tears the fixtures down, and the task ends after them, at once if there are none.

The teardown of a fixture that depends on the failed one runs in its own task, outside the failed fixture's scope, so that scope does not cancel it.
Until the failed fixture's own teardown exits the scope, an AnyIO cancel scope keeps re-cancelling the fixture's waiting task, which costs CPU time for as long as the dependent fixtures' cleanup takes: a substantial share of one core, varying with the machine and the Python version.

Context variables
-----------------

Each async fixture and each test runs in its own copy of the context (``contextvars``) of the synchronous pytest code that requested it, as it did before:

* Variables set by synchronous fixtures and tests are seen by the async fixtures and tests that follow.
* Variables set by an async fixture are seen by the synchronous fixtures and tests depending on it, and by later async code, until the fixture is torn down.
  The setup and teardown of an async fixture run in the same task and context, so a fixture can reset after its ``yield`` a variable it set before.
* Variables set by an async test are seen by no later fixture or test, whatever the loop scope of the test.
  Tasks and callbacks started by a fixture or test inherit its context, as always in asyncio.

A copy of a context shares the objects its variables refer to: only the bindings are separate.

Interruption
------------

When a test or fixture is interrupted, for example by Ctrl-C or by the signal of a timeout plugin, pytest-asyncio cancels its task, as ``asyncio.Runner`` does, and waits for the coroutine to finish, so that ``finally`` blocks and async context managers run before pytest tears down the fixtures the coroutine may be using.
A second interruption abandons the coroutine: its task is cancelled again and left to end on its own, and an error it ended or ends with is reported through the event loop's exception handler (pytest's logging capture shows it with live logging).
pytest-asyncio keeps the task until the loop closes, where asyncio cancels it once more and waits for it, so its cleanup still runs unless it ignores cancellation.
Other fixtures and tests are unaffected.

Limitations
-----------

* A timeout spanning a ``yield`` that expires while a test runs cancels the fixture's task, and with it the test; it does not raise ``TimeoutError`` at teardown, because the fixture's teardown exits the timeout without an exception.
* A cancellation that reaches a fixture's task at its ``yield`` ends the normal work of the whole loop, including for tests that do not use that fixture.
* pytest-asyncio's own tasks appear in ``asyncio.all_tasks()``: the tasks of the async generator fixtures in use, the ``pytest-asyncio`` task and the task waiting for the test. A test that cancels them ends the normal work of the loop, and a test that waits for the task of a fixture to end waits forever, since that task ends only when pytest tears the fixture down. Cancel and wait for the tasks you created.
