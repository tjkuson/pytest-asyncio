================================================
How to apply the asyncio marker programmatically
================================================
If all asynchronous tests in a test suite should be run by pytest-asyncio, consider enabling the auto mode via the ``asyncio_mode`` configuration option, rather than applying the marker programmatically.

When only some tests should receive the marker, or when the marking logic is more involved, the marker can be applied in a ``pytest_collection_modifyitems`` hook:

.. include:: mark_tests_programmatically_modifyitems_example.py
    :code: python

Markers applied in ``pytest_collection_modifyitems`` are applied *after* test collection. As a result, the marked tests cannot participate in collection-time behavior, such as the parametrization triggered by loop factories. If loop factories are configured via the ``pytest_asyncio_loop_factories`` hook, tests that received their marker after collection report an error.

To use loop factories with programmatically marked tests, apply the marker *during* collection in a ``pytest_pycollect_makeitem`` hook, instead:

.. include:: mark_tests_programmatically_makeitem_example.py
    :code: python
