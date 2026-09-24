import asyncio
import json
import logging
import urllib.parse
import urllib.request

from va import nlp
from va.actions import ActionError

logger = logging.getLogger(__name__)

WEATHER_CODES = {
    113: "ясно",
    116: "облачно с прояснениями",
    119: "облачно",
    122: "пасмурно",
    143: "лёгкий туман",
    176: "небольшой дождь",
    179: "небольшой снег",
    182: "небольшой ледяной дождь",
    185: "небольшая ледяная морось",
    200: "гроза вблизи",
    227: "метель",
    230: "пурга",
    248: "туман",
    260: "ледяной туман",
    263: "небольшая морось",
    266: "морось",
    281: "ледяная морось",
    284: "сильная ледяная морось",
    293: "небольшой дождь",
    296: "лёгкий дождь",
    299: "временами дождь",
    302: "дождь",
    305: "временами сильный дождь",
    308: "сильный дождь",
    311: "ледяной дождь",
    314: "сильный ледяной дождь",
    317: "небольшой снег с дождём",
    320: "снег с дождём",
    323: "небольшой снег",
    326: "снег",
    329: "временами снег",
    332: "умеренный снег",
    335: "временами сильный снег",
    338: "сильный снег",
    350: "ледяная крупа",
    353: "небольшой ливень",
    356: "ливень",
    359: "сильный ливень",
    362: "небольшая морось с дождём",
    365: "снег с дождём",
    368: "небольшой снегопад",
    371: "сильный снегопад",
    374: "ледяная крупа",
    377: "сильная ледяная крупа",
    386: "гроза с дождём",
    389: "сильная гроза с дождём",
    392: "гроза со снегом",
    395: "сильная гроза со снегом",
    149: "возможна гроза",
}

RAIN_CODES = {
    176,
    263,
    266,
    281,
    284,
    293,
    296,
    299,
    302,
    305,
    308,
    311,
    314,
    353,
    356,
    359,
    362,
    386,
    389,
}


def _format_temp(temp: int) -> str:
    if temp < 0:
        return f"минус {abs(temp)}"
    return str(temp)


def _weather_desc(cur: dict) -> str:
    code = int(cur["weatherCode"])
    if code in WEATHER_CODES:
        return WEATHER_CODES[code]
    logger.warning("Missing weather code: %s", code)
    return cur["weatherDesc"][0]["value"]


def _wind_strength(kmh: int) -> str:
    if kmh < 5:
        return "Ветра практически нет"
    if kmh < 12:
        return "Ветер лёгкий"
    if kmh < 22:
        return "Ветер умеренный"
    if kmh < 32:
        return "Ветер сильный"
    return "Ветер очень сильный"


def _rain_hours(today: dict) -> list[int]:
    hours = []
    for hour in today["hourly"]:
        code = int(hour.get("weatherCode", -1))
        chance = int(hour.get("chanceofrain", "0"))
        if code in RAIN_CODES or chance >= 40:
            hours.append(int(hour["time"]) // 100)
    return sorted(set(hours))


def _rain_sentence(cur: dict, today: dict) -> str:
    if int(cur.get("weatherCode", -1)) in RAIN_CODES:
        return "Сейчас идёт дождь."
    hours = _rain_hours(today)
    if not hours:
        return "Дождя сегодня не ожидается."
    run = [hours[0]]
    for hour in hours[1:]:
        if hour - run[-1] <= 3:
            run.append(hour)
        else:
            break
    if len(run) > 1:
        return f"Дождь ожидается с {run[0]} до {run[-1]} часов."
    hour = run[0]
    return f"Дождь ожидается в {hour} {nlp.plural(hour, 'час', 'часа', 'часов')}."


def _format_weather(data: dict, area: str | None = None) -> str:
    cur = data["current_condition"][0]
    today = data["weather"][0]
    if area is None:
        area = data["nearest_area"][0]["areaName"][0]["value"]
    area = nlp.decline(area, "loct")

    temp = int(cur["temp_C"])
    feels = int(cur["FeelsLikeC"])
    min_t = int(today["mintempC"])
    max_t = int(today["maxtempC"])

    parts = [
        f"Сейчас в {area} {_format_temp(temp)} ",
        f"{nlp.plural(temp, 'градус', 'градуса', 'градусов')}, {_weather_desc(cur)}.",
    ]
    if feels != temp:
        parts.append(
            f"Ощущается как {_format_temp(feels)} "
            f"{nlp.plural(feels, 'градус', 'градуса', 'градусов')}."
        )
    parts.append(f"Сегодня от {_format_temp(min_t)} до {_format_temp(max_t)} градусов.")
    parts.append(f"{_wind_strength(int(cur['windspeedKmph']))}.")
    parts.append(_rain_sentence(cur, today))
    return " ".join(parts)


def _fetch_location() -> dict | None:
    url = "http://ip-api.com/json/?fields=status,city,regionName,lat,lon&lang=ru"
    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            data = json.load(response)
    except OSError, ValueError:
        return None
    if data.get("status") != "success":
        return None
    return data


def _get(location: str, area: str | None = None) -> str:
    url = f"https://wttr.in/{location}?format=j1&lang=ru"
    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            data = json.load(response)
        return _format_weather(data, area)
    except (OSError, ValueError, KeyError) as e:
        raise ActionError(e) from e


async def get_weather(city: str | None = None) -> str:
    if city:
        location = urllib.parse.quote(city)
        area = None
    else:
        location_data = await asyncio.to_thread(_fetch_location)
        if location_data is None:
            location = ""
            area = None
        else:
            location = f"{location_data['lat']},{location_data['lon']}"
            area = location_data["city"]
    return await asyncio.to_thread(_get, location, area)
