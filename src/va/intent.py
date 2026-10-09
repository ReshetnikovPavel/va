import enum

import va.models
from va import decima


class Intent(enum.Enum):
    Unknown = "другое"
    Weather = "погода"
    Time = "время"
    Timer = "таймер"
    Timers = "таймеры"
    PauseMusic = "пауза"
    PlayMusic = "включить музыку"
    NextTrack = "следующий трек"
    PreviousTrack = "предыдущий трек"
    NowPlaying = "что играет"
    VolumeUp = "прибавить громкость"
    VolumeDown = "убавить громкость"
    VolumeMuchUp = "сильно прибавить громкость"
    VolumeMuchDown = "сильно убавить громкость"


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
            Intent.Timers.value: (
                "узнать про активные таймеры: какие идут, сколько их, сколько осталось"
            ),
            Intent.Timer.value: (
                "поставить таймер с длительностью: «на пять минут», "
                "«через полчаса», «таймер на две минуты»"
            ),
            Intent.PauseMusic.value: (
                "выключить музыку или поставить её на паузу: «хватит», "
                "«стоп», «пауза», «выключи»"
            ),
            Intent.NextTrack.value: ("переключить на следующий трек или песню, «next»"),
            Intent.PreviousTrack.value: (
                "переключить на предыдущий трек, вернуться назад, «previous»"
            ),
            Intent.PlayMusic.value: (
                "включить, поставить, сыграть песню, трек, альбом или "
                "исполнителя — по названию или запросу"
            ),
            Intent.Weather.value: ("узнать погоду, температуру, дождь, ветер, что за окном"),
            Intent.Time.value: "узнать текущее время: «сколько времени», «который час»",
            Intent.NowPlaying.value: "узнать, какая песня или трек сейчас играет",
            Intent.VolumeMuchUp.value: "сильно прибавить громкость: «погромче», «навали»",
            Intent.VolumeMuchDown.value: "сильно убавить громкость: «потише», «приглуши»",
            Intent.VolumeUp.value: "прибавить громкость: «громче», louder",
            Intent.VolumeDown.value: "убавить громкость: «тише», quieter",
            Intent.Unknown.value: (
                "всё остальное: приветствия, вопросы и просьбы, не перечисленные выше"
            ),
        },
    }
}


def classify(s: str) -> Intent:
    if False:
        question = decima.Question(
            QUESTIONS["intent"]["instructions"],
            QUESTIONS["intent"]["criteria"].keys(),
            lang="ru",
        )
        decision = va.models.DECIMA.decide(s, question)
        choice = decision.top
        conf = max(decision.probs)
        if conf < CONFIDENCE_THRESHOLD_DECIMA:
            return Intent.Unknown
    else:
        result = va.models.LAYA_ROUTER.predict(s, QUESTIONS)
        ans = result["answers"]["intent"]
        choice = ans["choice"]
        conf = ans.get("answer_confidence", ans.get("confidence", 1.0))
        if conf < CONFIDENCE_THRESHOLD_LAYA:
            return Intent.Unknown

    return Intent(choice)
