import asyncio
import sys

import pytest

import pytest_asyncio

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 11), reason="asyncio.timeout requires Python 3.11"
)


@pytest_asyncio.fixture
async def deadline():
    async with asyncio.timeout(1):
        yield


@pytest.mark.asyncio
async def test_finishes_before_the_deadline(deadline):
    await asyncio.sleep(0.01)
