import asyncio
import logging

from va.actions import ActionError, AssistantResponse

from . import intent, slots
from .actions import player, time, timer, weather
from .intent import Intent

logger = logging.getLogger(__name__)


async def process(text: str) -> AssistantResponse | None:
    action = intent.classify(text)
    data = await _extract_data(text, action)
    try:
        return await _execute_action(action, data)
    except ActionError as e:
        logger.exception(e)
        return AssistantResponse("Произошла какая-то ошибка, простите")


async def _extract_data(s: str, action: intent.Intent) -> dict:
    match action:
        case Intent.Weather | Intent.Time:
            locations = await asyncio.to_thread(slots.extract_locations, s)
            location = locations[0] if locations else None
            return {"location": location}
        case Intent.Timer:
            durations = await asyncio.to_thread(slots.extract_durations, s)
            duration = durations[0] if durations else None
            return {"duration": duration}
        case Intent.PlayMusic:
            query = await asyncio.to_thread(slots.extract_music_artist_and_or_title, s)
            return {"query": query}
    return {}


async def _execute_action(intent: Intent, data: dict) -> AssistantResponse | None:
    match intent:
        case Intent.Weather:
            return await weather.get_weather(data["location"])
        case Intent.Time:
            return await time.get_time(data["location"])
        case Intent.Timer:
            return await timer.set_timer(data["duration"])
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
        case Intent.Unknown:
            return AssistantResponse("Я глупая")
        case unhandled:
            raise RuntimeError(f"Unknown intent: `{unhandled}`")
