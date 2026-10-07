import asyncio
import datetime
from dataclasses import dataclass

from va import nlp
from va.actions import AssistantResponse


@dataclass(eq=False)
class Timer:
    until: datetime.datetime
    duration_secs: int


timers: set[Timer] = set()


def set_timer(secs: int | None) -> AssistantResponse:
    if secs is None:
        return AssistantResponse("На какое время поставить таймер?")
    if secs < 0:
        return AssistantResponse("Я не могу поставить таймер на отрицательное время")

    now = datetime.datetime.now(datetime.UTC)
    until = now + datetime.timedelta(seconds=secs)
    timers.add(Timer(until=until, duration_secs=secs))

    spoken = nlp.secs_to_words(secs, "accs", digits=False)
    display = nlp.secs_to_words(secs, "accs", digits=True)
    return AssistantResponse(
        f"Таймер на {spoken} поставлен", f"Таймер на {display} поставлен"
    )


def get_timers() -> AssistantResponse:
    if not timers:
        return AssistantResponse("Нет поставленных таймеров")

    now = datetime.datetime.now(datetime.UTC)

    if len(timers) == 1:
        timer = next(iter(timers))
        left = (timer.until - now).seconds

        duration_spoken = nlp.secs_to_words(timer.duration_secs, "accs", digits=False)
        duration_written = nlp.secs_to_words(timer.duration_secs, "accs", digits=True)
        left_spoken = nlp.secs_to_words(left, "nomn", digits=False)
        left_written = nlp.secs_to_words(left, "nomn", digits=True)

        return AssistantResponse(
            f"Поставлен таймер на {duration_spoken}, осталось {left_spoken}",
            f"Поставлен таймер на {duration_written}, осталось {left_written}",
        )

    spoken = ["Поставлены таймеры:"]
    written = ["Поставлены таймеры:"]
    for timer in timers.copy():
        left = int((timer.until - now).total_seconds())

        duration_spoken = nlp.secs_to_words(timer.duration_secs, "accs", digits=False)
        duration_written = nlp.secs_to_words(timer.duration_secs, "accs", digits=True)
        left_spoken = nlp.secs_to_words(left, "nomn", digits=False)
        left_written = nlp.secs_to_words(left, "nomn", digits=True)

        spoken.append(f"На {duration_spoken}, осталось {left_spoken};")
        written.append(f"На {duration_written}, осталось {left_written};")

    spoken = "\n".join(spoken)
    written = "\n".join(written)
    return AssistantResponse(spoken, written)


async def run_daemon() -> None:
    while True:
        now = datetime.datetime.now(datetime.UTC)
        for timer in timers.copy():
            if now >= timer.until:
                print("timer is up")
                timers.remove(timer)
        await asyncio.sleep(1)
