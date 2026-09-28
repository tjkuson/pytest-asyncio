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

If a fixture nests AnyIO cancel scopes and asyncio task groups or ``asyncio.timeout()`` around ``yield``, and both kinds are triggered while the fixture is in use, AnyIO bugs can lose or misreport a cancellation: a ``TimeoutError`` can go missing, or teardown can hang or fail with ``CancelledError``.
AnyIO releases after 4.15.1 fix the more serious of these, `anyio#1214 <https://github.com/agronholm/anyio/issues/1214>`__.
Fixtures that use only AnyIO scopes, or only asyncio ones, are not affected.
