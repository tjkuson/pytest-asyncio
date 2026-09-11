from __future__ import annotations

import pytest
from pytest import Config, MonkeyPatch, Pytester

pytest_plugins = "pytester"


@pytest.fixture
def pytester(
    pytester: Pytester, pytestconfig: Config, monkeypatch: MonkeyPatch
) -> Pytester:
    """Run shared scenarios with the runner selected for the outer test session."""
    if pytestconfig.getini("asyncio_experimental_task_per_fixture"):
        monkeypatch.setenv(
            "PYTEST_ADDOPTS", "-o asyncio_experimental_task_per_fixture=true"
        )
    return pytester
