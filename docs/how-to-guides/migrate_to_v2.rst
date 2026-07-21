=======================
How to migrate to v2
=======================

pytest-asyncio v2 replaces its fixture-created runners with a hook-managed event
loop lifecycle. Most tests continue to use ``pytest.mark.asyncio`` and
``pytest_asyncio.fixture`` unchanged, but the following compatibility APIs were
removed.

Replace event loop policies with loop factories
================================================

pytest-asyncio no longer defines or automatically requests an
``event_loop_policy`` fixture. A fixture with that name is now an ordinary user
fixture and has no effect on pytest-asyncio. Implement
``pytest_asyncio_loop_factories`` in ``conftest.py`` instead:

.. code-block:: python

   import asyncio


   def pytest_asyncio_loop_factories(config, item):
       return {"asyncio": asyncio.new_event_loop}

The hook may return multiple named factories. Use
``pytest.mark.asyncio(loop_factories=["name"])`` to select a subset for a test.

Update marker and fixture declarations
======================================

The deprecated ``pytest.mark.asyncio(scope=...)`` spelling is now an error. Use
``loop_scope=...``. The default fixture loop scope is now ``function``; set
``asyncio_default_fixture_loop_scope`` explicitly when a wider default is
required.

In strict mode, async fixtures declared with ``pytest.fixture`` are now an error.
Declare them with ``pytest_asyncio.fixture`` or opt into auto mode. Auto mode owns
plain async tests and async fixtures.

Plugin compatibility
====================

The internal ``PytestAsyncioFunction`` subclasses were removed. Tests remain
normal pytest ``Function`` items and are classified from their callable and
current marker at run time. Plugins should call ``pytest_asyncio.is_async_test``;
it returns ``bool`` and reflects markers applied by late collection hooks.

Loop-scope warnings
===================

pytest-asyncio emits ``PytestAsyncioWarning`` when a managed test or fixture
requests a managed fixture with a different loop scope. The warning follows
dependencies through synchronous fixtures. Align the scopes when the dependency
is expected to share loop-bound resources.

Implementation differences from the prototype
=============================================

The playground prototype keyed runners only by the scope name and sketched a new
fixture wrapper. The production implementation instead keys each runner by its
actual pytest scope root and loop-factory variant, retains the public
``pytest_asyncio.fixture`` decorator, supports package scope, and uses the
standard ``asyncio.Runner`` execution model for each async phase.
