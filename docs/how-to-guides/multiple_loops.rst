======================================
How to test with different event loops
======================================

Return multiple named factories from ``pytest_asyncio_loop_factories``. The following example causes all async tests to run once with the standard asyncio loop and once with a custom loop.

*conftest.py:*

.. include:: multiple_loops/conftest.py
    :code: python

*test_multiple_loops.py:*

.. include:: multiple_loops/test_multiple_loops.py
    :code: python

The hook receives the test item, so it can return different mappings for different parts of a test suite. A test can also select factory names with ``pytest.mark.asyncio(loop_factories=[...])``.
