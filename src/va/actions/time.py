import asyncio
import json
import urllib.parse
import urllib.request
from datetime import UTC, datetime

from va import nlp
from va.actions import ActionError, AssistantResponse
from va.actions.weather import _fetch_weather


def _format_spoken(hour: int, minute: int, area: str | None) -> str:
    head = (
        "Сейчас на этом компьютере — "
        if area is None
        else f"В {nlp.decline(area, 'loct')} сейчас "
    )
    time = nlp.time_to_words(hour, minute, 0, "nomn")
    if minute != 0:
        return f"{head}{time}."
    return f"{head}{time} ровно."


def _format_display(hour: int, minute: int, area: str | None) -> str:
    head = (
        "Сейчас на этом компьютере — "
        if area is None
        else f"В {nlp.decline(area, 'loct')} сейчас: "
    )
    return f"{head}{hour:02d}:{minute:02d}"


def _fetch_local_time(lat: float, lon: float) -> dict:
    url = (
        "https://timeapi.io/api/Time/current/coordinate"
        f"?latitude={lat}&longitude={lon}"
    )
    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            return json.load(response)
    except (OSError, ValueError, KeyError) as e:
        raise ActionError(e) from e


async def get_time(city: str | None = None) -> AssistantResponse:
    if city:
        location = urllib.parse.quote(city)
        data = await asyncio.to_thread(_fetch_weather, location)
        nearest = data["nearest_area"][0]
        tz_data = await asyncio.to_thread(
            _fetch_local_time, float(nearest["latitude"]), float(nearest["longitude"])
        )
        hour, minute = int(tz_data["hour"]), int(tz_data["minute"])
        area = nearest["areaName"][0]["value"]
    else:
        now = datetime.now(UTC).astimezone()
        hour, minute = now.hour, now.minute
        area = None
    return AssistantResponse(
        spoken=_format_spoken(hour, minute, area),
        display=_format_display(hour, minute, area),
    )
