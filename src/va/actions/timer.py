import asyncio

from va import nlp
from va.actions import AssistantResponse


async def set_timer(secs: int | None) -> AssistantResponse:
    if secs is None:
        return AssistantResponse("На какое время поставить таймер?")
    if secs < 0:
        return AssistantResponse("Я не могу поставить таймер на отрицательное время")
    asyncio.create_task(_sleep_and_fire(secs))
    spoken = nlp.secs_to_words(secs, 'accs', digits=False)
    display = nlp.secs_to_words(secs, 'accs', digits=True)
    return AssistantResponse(f"Таймер на {spoken} поставлен", f"Таймер на {display} поставлен")


async def _sleep_and_fire(secs: float):
    await asyncio.sleep(secs)
    print("Timer is up!")
