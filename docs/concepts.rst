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

.. _concepts/context_variables:

Context variables
=================

Context variables hold values, such as a request identifier, that can differ between tasks.
Pytest-asyncio follows these fixture propagation and test isolation rules:

* Async fixtures and tests inherit context variables from the synchronous pytest code that requests them.
* Values set during async fixture setup are visible to its synchronous and asynchronous dependents until the fixture is torn down.
* An async test receives its own context, so its assignments are not visible to later fixtures or tests, even when tests share an event loop.

The objects held in context variables can still be shared: changing a mutable object can affect other tests.
See also the experimental runner's :ref:`compatibility notes <configuration/experimental_asyncio_task_group_runner/compatibility>`.

.. _concepts/tasks:

Fixture tasks and cancellation
==============================

This section describes the opt-in :ref:`experimental task group runner <configuration/experimental_asyncio_task_group_runner>`.

An async generator fixture runs in one asyncio task from setup through teardown.
This lets a task group or timeout surround its ``yield``: the task that enters the context manager also exits it.
The default runner uses separate tasks for setup and teardown.

A failed test still gets ordinary fixture cleanup.
A fixture can also receive a request to stop, called *cancellation*, for example when one of its task group's background tasks fails.
Pytest-asyncio lets fixtures that depend on it finish their cleanup before raising cancellation at the cancelled fixture's ``yield``.
Enclosing context managers and ``finally`` blocks run; ordinary statements after ``yield`` are skipped unless the fixture handles cancellation.

For example, this fixture records a heartbeat while the test uses its file:

.. include:: concepts_task_group_fixture_example.py
    :code: python

The ``finally`` block stops the background task, the task group waits for it to finish, and the outer ``with`` closes the file.
If a background write fails, the running test is cancelled and pytest reports the background error at fixture teardown.
The context managers and ``finally`` block also handle cleanup in that case.

See :ref:`configuration/experimental_asyncio_task_group_runner/cancellation` for the effects on other tests and the limits of cancellation.
