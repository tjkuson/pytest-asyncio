import asyncio
import sys

import pytest
import pytest_asyncio

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 11), reason="asyncio.TaskGroup requires Python 3.11"
)


async def heartbeat(beats: list[float]) -> None:
    while True:
        beats.append(asyncio.get_running_loop().time())
        await asyncio.sleep(0.01)


@pytest_asyncio.fixture
async def beats():
    beats: list[float] = []
    async with asyncio.TaskGroup() as tg:
        task = tg.create_task(heartbeat(beats))
        yield beats
        task.cancel()


@pytest.mark.asyncio
async def test_heartbeat_runs_during_the_test(beats):
    beats_before = len(beats)
    await asyncio.sleep(0.05)
    assert len(beats) > beats_before
