import inspect

import pytest


def pytest_pycollect_makeitem(collector, name, obj):
    func = getattr(obj, "__func__", obj)
    if inspect.iscoroutinefunction(func) and collector.istestfunction(obj, name):
        pytest.mark.asyncio(obj)
