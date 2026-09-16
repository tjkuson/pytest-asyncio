from __future__ import annotations

import pytest
from pytest import Config, MonkeyPatch, Pytester

pytest_plugins = "pytester"


@pytest.fixture
def pytester(
    pytester: Pytester, pytestconfig: Config, monkeypatch: MonkeyPatch
) -> Pytester:
    """Run shared scenarios with the runner selected for the outer test session."""
    if pytestconfig.getini("experimental_asyncio_task_group_runner"):
        monkeypatch.setenv(
            "PYTEST_ADDOPTS", "-o experimental_asyncio_task_group_runner=true"
        )
    return pytester
