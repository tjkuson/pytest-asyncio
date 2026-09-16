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

.. _configuration/experimental_asyncio_task_group_runner:

experimental_asyncio_task_group_runner
======================================

Enables task groups and cancel scopes spanning an async generator fixture's ``yield`` (see :ref:`concepts/tasks`).
Requires Python 3.11 or later and defaults to ``false``.
The experimental behavior may change before becoming the default in pytest-asyncio version 2.

To enable it in ``pytest.ini``:

.. code-block:: ini

   [pytest]
   experimental_asyncio_task_group_runner = true

To enable it for one run:

.. code-block:: console

   $ pytest -o experimental_asyncio_task_group_runner=true

.. _configuration/experimental_asyncio_task_group_runner/cancellation:

Cancellation and cleanup
------------------------

If a fixture is cancelled while in use, any async test or async fixture setup still running on its loop is cancelled too.
Further async tests and fixture setups cannot run on that loop, including tests that do not use the cancelled fixture.
Fixture teardown still follows pytest's usual order.
Further cancellation can interrupt cleanup that awaits.

Pytest reports unhandled cancellation as a test failure or a fixture setup or teardown error.
If a task group or timeout fails while its fixture is suspended at ``yield``, pytest reports the error at fixture teardown.
This also applies when the test handles its own cancellation.

After Ctrl-C or a signal-based timeout, pytest-asyncio waits for async cleanup before allowing fixture resources to close.
A second interruption stops that wait, so cleanup may be incomplete; errors raised later may appear only in captured logs.

If cleanup does not finish after a signal-based timeout, the test run can hang.
For a process-level deadline, use an external watchdog or the ``thread`` method of `pytest-timeout <https://github.com/pytest-dev/pytest-timeout/blob/main/README.rst#timeout-methods>`_.
This can end the test run before cleanup or report generation finishes.

.. _configuration/experimental_asyncio_task_group_runner/compatibility:

Compatibility notes
-------------------

The experimental runner preserves :ref:`fixture context propagation and test isolation <concepts/context_variables>`.
Each Hypothesis example also gets a fresh context; the default runner shares a context between examples of one test.
Custom task factories that modify context variables may produce different values with the two runners.

A cancelled AnyIO scope spanning ``yield`` can repeatedly cancel its fixture while it waits for teardown, consuming CPU.

Code inspecting a test task's exception may not see failures reported by pytest.
Pytest-asyncio also manages internal tasks that may appear in ``asyncio.all_tasks()``.
Their names, number and arrangement are not part of its public API and may change between releases.
