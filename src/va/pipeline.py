import logging
import re
import typing

import llama_cpp

from va.actions import ActionError, AssistantResponse

from . import intent, models, slots, llm
from .actions import player, time, timer, weather
from .intent import Intent

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = 'Ты - голосовая ассистентка по имени Ада. Ты создана для выполнения простых задач, таких как ставить таймер и говорить погоду. Ты всегда говоришь о себе в женском роде: "я создана", "я думала", "я чувствовала". Не повторяй одни и те же по смыслу фразы.'

CJK_REGEX = re.compile(r"[\u4E00-\u9FFF]+")


class Pipeline:
    def __init__(self) -> None:
        self.messages: list[llama_cpp.ChatCompletionRequestMessage] = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

    async def process(self, text: str) -> AssistantResponse | None:
        self.messages.append({"role": "user", "content": text})

        action = intent.classify(text)
        print(action)
        data = _extract_data(text, action)
        print(data)

        if action == Intent.Unknown:
            response = AssistantResponse(llm.create_chat_completion(self.messages))
        else:
            try:
                response = await _execute_action(action, data)
            except ActionError as e:
                logger.exception(e)
                response = AssistantResponse("Произошла какая-то ошибка, простите")

        if response is not None:
            self.messages.append(
                {
                    "role": "assistant",
                    "content": response.display or response.spoken,
                }
            )
        return response


def _extract_data(s: str, action: intent.Intent) -> dict:
    match action:
        case Intent.Weather | Intent.Time:
            locations = slots.extract_locations(s)
            location = locations[0] if locations else None
            return {"location": location}
        case Intent.Timer:
            durations = slots.extract_durations(s)
            duration = durations[0] if durations else None
            return {"duration": duration}
        case Intent.PlayMusic:
            query = slots.extract_music_artist_and_or_title(s)
            return {"query": query}
    return {}


async def _execute_action(intent: Intent, data: dict) -> AssistantResponse | None:
    match intent:
        case Intent.Weather:
            return await weather.get_weather(data["location"])
        case Intent.Time:
            return await time.get_time(data["location"])
        case Intent.Timer:
            return timer.set_timer(data["duration"])
        case Intent.Timers:
            return timer.get_timers()
        case Intent.PauseMusic:
            return player.pause_music()
        case Intent.NowPlaying:
            return player.now_playing()
        case Intent.PlayMusic:
            return await player.play_music(data["query"])
        case Intent.NextTrack:
            return player.next_track()
        case Intent.PreviousTrack:
            return player.previous_track()
        case Intent.VolumeUp:
            return player.volume_up()
        case Intent.VolumeDown:
            return player.volume_down()
        case Intent.VolumeMuchUp:
            return player.volume_much_up()
        case Intent.VolumeMuchDown:
            return player.volume_much_down()
        case unhandled:
            raise RuntimeError(f"Unknown intent: `{unhandled}`")
