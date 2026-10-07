import asyncio

from . import state_machine
from .actions import timer


async def main() -> None:
    async with asyncio.TaskGroup() as tasks:
        tasks.create_task(timer.run_daemon())
        tasks.create_task(state_machine.run())


asyncio.run(main())
