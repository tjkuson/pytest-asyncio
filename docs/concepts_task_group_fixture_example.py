import asyncio

import pytest

import pytest_asyncio


async def write_heartbeats(output):
    while True:
        output.write("alive\n")
        output.flush()
        await asyncio.sleep(0.01)


@pytest_asyncio.fixture
async def heartbeat_log(tmp_path):
    path = tmp_path / "heartbeat.log"
    with path.open("w") as output:
        async with asyncio.TaskGroup() as group:
            task = group.create_task(write_heartbeats(output))
            try:
                yield path
            finally:
                task.cancel()


@pytest.mark.asyncio
async def test_background_task_records_a_heartbeat(heartbeat_log):
    await asyncio.sleep(0.05)
    assert "alive" in heartbeat_log.read_text()
