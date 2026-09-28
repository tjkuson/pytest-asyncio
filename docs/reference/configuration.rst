=============
Configuration
=============

.. _configuration/asyncio_default_fixture_loop_scope:

asyncio_default_fixture_loop_scope
==================================
Determines the default event loop scope of asynchronous fixtures. When this configuration option is unset, it defaults to the fixture scope. In future versions of pytest-asyncio, the value will default to ``function`` when unset. Possible values are: ``function``, ``class``, ``module``, ``package``, ``session``

.. _configuration/asyncio_default_test_loop_scope:

asyncio_default_test_loop_scope
===============================
Determines the default event loop scope of asynchronous tests. When this configuration option is unset, it defaults to function scope. Possible values are: ``function``, ``class``, ``module``, ``package``, ``session``

.. _configuration/asyncio_debug:

asyncio_debug
=============
Enables `asyncio debug mode <https://docs.python.org/3/library/asyncio-dev.html#debug-mode>`_ for the default event loop used by asynchronous tests and fixtures.

The debug mode can be set by the ``asyncio_debug`` configuration option in the `configuration file
<https://docs.pytest.org/en/latest/reference/customize.html>`_:

.. code-block:: ini

   # pytest.ini
   [pytest]
   asyncio_debug = true

The value can also be set via the ``--asyncio-debug`` command-line option:

.. code-block:: bash

   $ pytest tests --asyncio-debug

By default, asyncio debug mode is disabled.

asyncio_mode
============
The pytest-asyncio mode can be set by the ``asyncio_mode`` configuration option in the `configuration file
<https://docs.pytest.org/en/latest/reference/customize.html>`_:

.. code-block:: ini

   # pytest.ini
   [pytest]
   asyncio_mode = auto

The value can also be set via the ``--asyncio-mode`` command-line option:

.. code-block:: bash

   $ pytest tests --asyncio-mode=strict


If the asyncio mode is set in both the pytest configuration file and the command-line option, the command-line option takes precedence. If no asyncio mode is specified, the mode defaults to `strict`.

.. _configuration/asyncio_experimental_task_per_fixture:

asyncio_experimental_task_per_fixture
=====================================

Runs each async fixture in an asyncio task of its own, from setup to teardown, so that task groups, timeouts and AnyIO cancel scopes can span an async generator fixture's ``yield``.
:ref:`concepts/tasks` explains why this matters.

The option is experimental: its behavior may change in any release.
It is intended to become the default in a future major release.
It requires Python 3.11 or later; enabling it on Python 3.10 is a usage error.
Defaults to ``false``.

To enable it in ``pytest.ini``:

.. code-block:: ini

   [pytest]
   asyncio_experimental_task_per_fixture = true

To enable it for one run:

.. code-block:: console

   $ pytest -o asyncio_experimental_task_per_fixture=true

.. _configuration/asyncio_experimental_task_per_fixture/cancellation:

Cancellation and cleanup
------------------------

A fixture can be cancelled while in use, for example when a background task in its task group fails or its timeout expires.
Then:

* The async test or fixture setup running on the fixture's event loop, if any, is cancelled.
* The fixture's own cleanup waits until pytest tears it down, after its dependents, in pytest's usual order.
  The cancellation is then raised at its ``yield``.
* Until then, new async tests and fixture setups on the fixture's event loop fail with an error that names the fixture.
  A function-scoped fixture is torn down right after its test, so no other test is affected.
  A module-scoped fixture on a module-scoped loop affects the rest of that module's tests on the loop.

On Ctrl-C, pytest-asyncio cancels the running async test or fixture and waits for its cleanup, while the fixtures it uses are still available.
A second Ctrl-C stops waiting, so cleanup may be incomplete.
The default runner does the same.

With this option, pytest-asyncio also handles other exceptions that interrupt the event loop this way, such as a timeout from `pytest-timeout <https://github.com/pytest-dev/pytest-timeout>`_ with the ``signal`` method.
By default, a test interrupted by such a timeout finishes its cleanup only when its event loop closes, after its fixtures have been torn down.

.. _configuration/asyncio_experimental_task_per_fixture/limitations:

Limitations
-----------

Cancel only tasks that your code created: cancelling tasks that pytest-asyncio uses, for example every task in ``asyncio.all_tasks()``, can fail a test or fixture, cut its cleanup short, or make code that then waits for those tasks hang.

A fixture can miss a cancellation if it has an AnyIO cancel scope inside an asyncio task group or ``asyncio.timeout()``, both around ``yield``, and the AnyIO scope is cancelled while the fixture is in use, for example by the deadline of ``anyio.move_on_after()``.
If the asyncio task group or timeout is then triggered before the fixture is torn down, the fixture does not receive that cancellation.
The timeout's ``TimeoutError`` can then be missing from the report.
An async test that is running when the AnyIO scope is cancelled fails, because that cancellation also cancels it.
If both scopes are triggered after the test has finished, for example during the slow teardown of a dependent fixture, the test can pass with no error.
Teardown can also hang if cleanup waits for work that the task group's failed task will never do, for example by calling ``join()`` on a queue that the task consumed.
Pressing Ctrl-C ends the hang.
pytest-timeout does not time teardown after a failed test, whichever method it uses, so it cannot end a hang in that teardown.
To limit such hangs, set a timeout for the whole run, such as your CI job's timeout.

With the nesting reversed, an ``asyncio.timeout()`` inside an AnyIO cancel scope, the fixture's teardown can fail with a cancellation error instead of the timeout's ``TimeoutError``.
AnyIO scopes that libraries open and close within a call are not affected, nor are fixtures that nest only AnyIO scopes or only asyncio ones.
To avoid these problems, use AnyIO for both scopes, or cancel the AnyIO scope only after ``yield``, during teardown.

While a fixture whose AnyIO cancel scope was cancelled waits for teardown, AnyIO keeps cancelling it, which uses CPU whenever other code runs on its event loop, such as the teardown of its dependents.
