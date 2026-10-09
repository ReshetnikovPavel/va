import enum

import va.models
from va import decima


class Intent(enum.Enum):
    Unknown = enum.auto()
    Weather = enum.auto()
    Time = enum.auto()
    Timer = enum.auto()
    Timers = enum.auto()
    PauseMusic = enum.auto()
    PlayMusic = enum.auto()
    NextTrack = enum.auto()
    PreviousTrack = enum.auto()
    NowPlaying = enum.auto()
    VolumeUp = enum.auto()
    VolumeDown = enum.auto()
    VolumeMuchUp = enum.auto()
    VolumeMuchDown = enum.auto()


CONFIDENCE_THRESHOLD_LAYA = 0.55
CONFIDENCE_THRESHOLD_DECIMA = 0.3
QUESTIONS = {
    "intent": {
        "type": "choice",
        "instructions": (
            "Реплика пользователя голосовому ассистенту. Выбери ровно одну "
            "команду из списка; если команды нет в списке или это не команда, "
            "выбери «другое»."
        ),
        "criteria": {
            "таймеры": (
                "узнать про активные таймеры: какие идут, сколько их, сколько осталось"
            ),
            "таймер": (
                "поставить таймер с длительностью: «на пять минут», "
                "«через полчаса», «таймер на две минуты»"
            ),
            "пауза": (
                "выключить музыку или поставить её на паузу: «хватит», "
                "«стоп», «пауза», «выключи»"
            ),
            "следующий трек": ("переключить на следующий трек или песню, «next»"),
            "предыдущий трек": (
                "переключить на предыдущий трек, вернуться назад, «previous»"
            ),
            "включить музыку": (
                "включить, поставить, сыграть песню, трек, альбом или "
                "исполнителя — по названию или запросу"
            ),
            "погода": ("узнать погоду, температуру, дождь, ветер, что за окном"),
            "время": "узнать текущее время: «сколько времени», «который час»",
            "что играет": "узнать, какая песня или трек сейчас играет",
            "намного громче": "сильно прибавить громкость: «погромче», «навали»",
            "намного тише": "сильно убавить громкость: «потише», «приглуши»",
            "громче": "прибавить громкость: «громче», louder",
            "тише": "убавить громкость: «тише», quieter",
            "другое": (
                "всё остальное: приветствия, вопросы и просьбы, не перечисленные выше"
            ),
        },
    }
}


def classify(s: str) -> Intent:
    if True:
        question = decima.Question(QUESTIONS["intent"]["instructions"], QUESTIONS["intent"]["criteria"].keys(), lang="ru")
        decision = va.models.DECIMA.decide(s, question)
        print(decision)
        choice = decision.top
        conf = max(decision.probs)
        if conf < CONFIDENCE_THRESHOLD_DECIMA:
            return Intent.Unknown
    else:
        result = va.models.LAYA_ROUTER.predict(s, QUESTIONS)
        print(result)
        ans = result["answers"]["intent"]
        choice = ans["choice"]
        conf = ans.get("answer_confidence", ans.get("confidence", 1.0))
        if conf < CONFIDENCE_THRESHOLD_LAYA:
            return Intent.Unknown

    match choice:
        case "таймеры":
            return Intent.Timers
        case "таймер":
            return Intent.Timer
        case "включить музыку":
            return Intent.PlayMusic
        case "погода":
            return Intent.Weather
        case "время":
            return Intent.Time
        case "следующий трек":
            return Intent.NextTrack
        case "предыдущий трек":
            return Intent.PreviousTrack
        case "пауза":
            return Intent.PauseMusic
        case "что играет":
            return Intent.NowPlaying
        case "громче":
            return Intent.VolumeUp
        case "тише":
            return Intent.VolumeDown
        case "намного громче":
            return Intent.VolumeMuchUp
        case "намного тише":
            return Intent.VolumeMuchDown
        case "другое":
            return Intent.Unknown
        case _:
            raise RuntimeError(f"Unknown laya choice `{choice}`")
