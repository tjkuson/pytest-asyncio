import inspect


def pytest_collection_modifyitems(items):
    for item in items:
        if inspect.iscoroutinefunction(getattr(item, "obj", None)):
            item.add_marker("asyncio")
