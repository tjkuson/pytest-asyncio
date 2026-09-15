from pathlib import Path

import pytest


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getini("experimental_asyncio_task_group_runner"):
        return

    example = Path(__file__).with_name("concepts_task_group_fixture_example.py")
    for item in items:
        if item.path == example:
            item.add_marker(
                pytest.mark.skip(reason="Requires the experimental task group runner")
            )
