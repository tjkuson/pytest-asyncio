import asyncio

import pytest


@pytest.mark.asyncio
async def test_uses_configured_loop():
    assert isinstance(asyncio.get_running_loop(), asyncio.AbstractEventLoop)
