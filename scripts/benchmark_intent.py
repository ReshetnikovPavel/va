"""Бенчмарк распознавания интентов: только laya и перебор её конфигов.

Чем этот бенчмарк отличается от scripts/benchmark_classify.py:

- только laya. Ни LLM, ни эмбеддингов, ни keyword-бейзлайна — сравниваются
  конфиги рантайма laya, а не разные классификаторы;
- нет утечки тестовых фраз в промпт. Критерии rich/terse пишутся заново и
  проверяются на совпадение с золотым набором (см. leak_report, для этих
  промптов это падение). Сколько фраз продакшн-промпт цитирует дословно,
  печатается колонкой `lk` — это свойство продакшна, а не ошибки набора;
- золотой набор закрывает все 14 интентов продакшена, включая таймеры,
  которых старый бенчмарк не знал («поставь таймер на пять минут» там
  считался Unknown, хотя продакшен его выполняет);
- метрики берутся из laya.evals — собственного харнесса laya: choice_accuracy,
  ECE, AURC, selective accuracy, перцентили latency, срезы по тегу/языку/
  чекпоинту. Task-метрики (exact/wrong/refused/fpresume) считаются поверх
  его построчных отчётов, руками измеряется ничего;
- воронка конфигов, где каждая стадия питает следующую (см. STAGES);
- прогон пакетный: laya.evals чанкует запросы в Router.predict_batch, одиночные
  вызовы остаются на прогреве и на пробе latency. Без батчей один прогон на
  CPU идёт десятки минут, с батчами — единицы.

Стадии (по умолчанию — только те, что показали себя; --full возвращает все):
  base   — быстрый набор: авто-роутинг на prod и terse (см. FAST_BASE);
           --full даёт 6 конфигов: три промпта + все чекпоинты на prod;
  order  — criteria лучшего base в обратном порядке (--full): проверка на
           позиционный сдвиг; обратный порядок показался хуже (+6 -> -21);
  gate   — свип min_confidence на лучших base, две метрики гейта:
           maxp — штатный answer_confidence laya, entropy — поле confidence;
           считается на готовых ответах base, без повторного прогона;
  lang   — принудительный lang="ru" против авто-роутинга.

Отброшено по итогам первого прогона: rich (acc 0.43), english (0.25),
typed-decisions (0.32) против 0.58 у prod; multilingual повторяет auto
побитово. --full возвращает их и стадию order.

Температура (laya.calibrate) и dev-сплит из прогона убраны: меряем конфиги
как есть, на всём золотом наборе.

Запуск (laya — зависимость проекта, ничего сверх не нужно):
    uv run python scripts/benchmark_intent.py                 # все стадии
    uv run python scripts/benchmark_intent.py --list          # данные и план
    uv run python scripts/benchmark_intent.py --stages base,gate
    uv run python scripts/benchmark_intent.py rich            # только *rich*
    uv run python scripts/benchmark_intent.py --detail base:auto:rich
    uv run python scripts/benchmark_intent.py --batch-size 32 # больше батч
    uv run python scripts/benchmark_intent.py --json output/intent_bench.json

Категория по кейсу (классификатор всегда возвращает метку):
  exact    — метка совпала с ожидаемой (включая «другое»);
  wrong    — вернул не ту команду: выполнится не то — самый вредный исход;
  refused  — ждали команду, а он отказался (выбрал «другое» или сработал гейт);
  fpresume — ждали «другое», а он придумал команду.

Балл: exact +2, wrong -3, refused -1, fpresume -2 (сумма по золотому набору).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path

OTHER = "другое"

# Порядок и состав — как в src/va/intent.py: ключ словаря criteria это ровно
# та строка, которую рантайм потом матчит на Intent.
LABELS: tuple[str, ...] = (
    "таймеры",
    "таймер",
    "пауза",
    "следующий трек",
    "предыдущий трек",
    "музыка",
    "погода",
    "время",
    "что играет",
    "намного громче",
    "намного тише",
    "громче",
    "тише",
    OTHER,
)

# ------------------------------------------------------------------ золотой набор
# (реплика, ожидаемая метка). Фразы писались под то, что продакшен реально
# выполняет: «отмени таймер» ждёт другое — отмены в Intents нет, и классификатор,
# вернувший таймер, заставит set_timer упасть.
CASES: tuple[tuple[str, str], ...] = (
    # музыка
    ("включи платинум", "музыка"),
    ("поставь the void", "музыка"),
    ("включи песню группы спиритбокс", "музыка"),
    ("сыграй что-нибудь", "музыка"),
    ("хочу послушать опет", "музыка"),
    ("давай опет", "музыка"),
    ("врубай спанч боб", "музыка"),
    ("включи монгольский рок", "музыка"),
    ("включи следующие треки линкин парк", "музыка"),
    ("включи ещё опет", "музыка"),
    ("поставь альбом куртейн", "музыка"),
    ("хочу послушать джаз", "музыка"),
    ("запусти плейлист для дороги", "музыка"),
    ("дай музыку понастроению", "музыка"),
    ("включи последний альбом молчат дома", "музыка"),
    ("поставь фоновую музыку", "музыка"),
    ("включи концерт киркорова", "музыка"),
    ("найди и поставь саундтрек из титаника", "музыка"),
    ("вруби рок-н-ролл", "музыка"),
    ("play some music", "музыка"),
    # погода
    ("какая погода", "погода"),
    ("какая погода в москве", "погода"),
    ("что за окном", "погода"),
    ("погода завтра", "погода"),
    ("будет ли дождь", "погода"),
    ("какой сейчас ветер", "погода"),
    ("сколько градусов на улице", "погода"),
    ("нужен ли зонт сегодня", "погода"),
    ("прогноз погоды на неделю", "погода"),
    ("замёрзну ли я на улице", "погода"),
    ("what's the weather like", "погода"),
    # время
    ("сколько времени", "время"),
    ("который час", "время"),
    ("во сколько сейчас время", "время"),
    ("сколько время в лондоне", "время"),
    ("который час в нью-йорке", "время"),
    ("какое сейчас время", "время"),
    ("текущее время скажи", "время"),
    ("сколько минут до полуночи", "время"),
    ("а который час уже", "время"),
    ("what time is it", "время"),
    # таймер (поставить)
    ("поставь таймер на пять минут", "таймер"),
    ("таймер на десять минут", "таймер"),
    ("поставь таймер на полчаса", "таймер"),
    ("запусти таймер на минуту", "таймер"),
    ("на сорок секунд таймер", "таймер"),
    ("таймер на восемь минут", "таймер"),
    ("поставь отсчёт на две минуты", "таймер"),
    ("через сорок минут напомни про кашу", "таймер"),
    ("через пять минут подними", "таймер"),
    # таймеры (посмотреть)
    ("сколько таймеров у меня", "таймеры"),
    ("какие таймеры сейчас идут", "таймеры"),
    ("сколько осталось по таймеру", "таймеры"),
    ("активные таймеры покажи", "таймеры"),
    ("сколько таймеров осталось", "таймеры"),
    ("таймеры вообще есть", "таймеры"),
    ("проверь таймеры", "таймеры"),
    ("сколько их идёт по времени", "таймеры"),
    # пауза
    ("пауза", "пауза"),
    ("поставь на паузу", "пауза"),
    ("выключи музыку", "пауза"),
    ("стоп", "пауза"),
    ("останови музыку", "пауза"),
    ("замолчи", "пауза"),
    ("прекрати воспроизведение", "пауза"),
    ("выключи плеер", "пауза"),
    ("паузу дай", "пауза"),
    ("останови", "пауза"),
    # следующий трек
    ("следующий трек", "следующий трек"),
    ("следующая песня", "следующий трек"),
    ("переключи дальше", "следующий трек"),
    ("следующая", "следующий трек"),
    ("пропусти эту", "следующий трек"),
    ("next song", "следующий трек"),
    # предыдущий трек
    ("предыдущий трек", "предыдущий трек"),
    ("предыдущая песня", "предыдущий трек"),
    ("назад", "предыдущий трек"),
    ("верни назад", "предыдущий трек"),
    ("отмотай назад", "предыдущий трек"),
    ("play the previous song", "предыдущий трек"),
    # что играет
    ("что играет", "что играет"),
    ("что сейчас играет", "что играет"),
    ("какая это песня", "что играет"),
    ("какой трек идёт", "что играет"),
    ("что за музыка играет", "что играет"),
    ("что сейчас звучит", "что играет"),
    ("какая песня играет", "что играет"),
    ("что за трек", "что играет"),
    # громче
    ("сделай громче", "громче"),
    ("прибавь звук", "громче"),
    ("громче", "громче"),
    ("louder", "громче"),
    ("добавь громкости", "громче"),
    ("громкость чуть выше", "громче"),
    ("звук громче сделай", "громче"),
    ("чуть громче", "громче"),
    ("увеличь громкость", "громче"),
    # тише
    ("сделай тише", "тише"),
    ("убавь звук", "тише"),
    ("тише", "тише"),
    ("quieter", "тише"),
    ("снизь громкость", "тише"),
    ("звук тише сделай", "тише"),
    ("чуть тише", "тише"),
    ("уменьши громкость", "тише"),
    # намного громче
    ("погромче", "намного громче"),
    ("навались", "намного громче"),
    ("намного громче", "намного громче"),
    ("врубай на всю", "намного громче"),
    ("максимально громко", "намного громче"),
    ("громче в разы", "намного громче"),
    ("на максимум громкость", "намного громче"),
    ("громко очень", "намного громче"),
    ("давай в полную мощь", "намного громче"),
    # намного тише
    ("потише", "намного тише"),
    ("приглуши", "намного тише"),
    ("намного тише", "намного тише"),
    ("еле слышно сделай", "намного тише"),
    ("тише почти в ноль", "намного тише"),
    ("сделай шёпотом", "намного тише"),
    ("почти без звука", "намного тише"),
    ("прибери звук совсем", "намного тише"),
    ("тише тише", "намного тише"),
    # другое
    ("расскажи анекдот", OTHER),
    ("привет", OTHER),
    ("выключи свет", OTHER),
    ("какой сегодня день", OTHER),
    ("открой браузер", OTHER),
    ("что такое фотосинтез", OTHER),
    ("переведи фразу", OTHER),
    ("сколько будет два плюс два", OTHER),
    ("заведи будильник на семь утра", OTHER),
    ("шутку расскажи", OTHER),
    ("какая сейчас дата", OTHER),
    ("как тебя зовут", OTHER),
    ("спасибо", OTHER),
    ("кто ты", OTHER),
    ("запиши в список покупок молоко", OTHER),
    ("включи фонарик", OTHER),
    ("отмени таймер", OTHER),
    ("похвали меня", OTHER),
    ("какая порода у собак", OTHER),
    ("свари кофе", OTHER),
    ("поставь фильм", OTHER),
    ("найди рецепт пельменей", OTHER),
    ("сколько весит слон", OTHER),
    ("поиграй со мной", OTHER),
)

# -------------------------------------------------------------------- промпты
@dataclass(frozen=True)
class Prompt:
    instructions: str
    criteria: dict[str, str]  # порядок вставки = порядок опций для модели

    def questions(self, reverse: bool = False) -> dict:
        items = list(self.criteria.items())
        if reverse:
            items.reverse()
        return {
            "intent": {
                "type": "choice",
                "instructions": self.instructions,
                "criteria": dict(items),
            }
        }


# prod — дословно из src/va/intent.py, это продакшен-конфиг, его и меряем.
PROD_PROMPT = Prompt(
    instructions=(
        "Реплика пользователя голосовому ассистенту. Выбери ровно одну "
        "команду из списка; если команды нет в списке или это не команда, "
        "выбери «другое»."
    ),
    criteria={
        "таймеры": (
            "узнать про активные таймеры: какие идут, сколько их, "
            "сколько осталось"
        ),
        "таймер": (
            "поставить таймер с длительностью: «на пять минут», "
            "«через полчаса», «таймер на две минуты»"
        ),
        "пауза": (
            "выключить музыку или поставить её на паузу: «хватит», "
            "«стоп», «пауза», «выключи»"
        ),
        "следующий трек": "переключить на следующий трек или песню, «next»",
        "предыдущий трек": (
            "переключить на предыдущий трек, вернуться назад, «previous»"
        ),
        "музыка": (
            "включить, поставить, сыграть песню, трек, альбом или "
            "исполнителя — по названию или запросу"
        ),
        "погода": "узнать погоду, температуру, дождь, ветер, что за окном",
        "время": "узнать текущее время: «сколько времени», «который час»",
        "что играет": "узнать, какая песня или трек сейчас играет",
        "намного громче": "сильно прибавить громкость: «погромче», «навали»",
        "намного тише": "сильно убавить громкость: «потише», «приглуши»",
        "громче": "прибавить громкость: «громче», louder",
        "тише": "убавить громкость: «тише», quieter",
        "другое": (
            "всё остальное: приветствия, вопросы и просьбы, не "
            "перечисленные выше"
        ),
    },
)

# rich — те же 14 меток, но описанными семантикой, без цитат из тестового набора.
RICH_PROMPT = Prompt(
    instructions=(
        "Определи команду голосового ассистента, которую выполняет реплика "
        "пользователя. Если команды нет в списке или это не команда, выбери "
        "«другое»."
    ),
    criteria={
        "таймеры": (
            "вопрос о уже запущенных отсчётах: сколько их, какие идут, "
            "сколько осталось до конца"
        ),
        "таймер": (
            "запуск отсчёта времени с указанной в реплике длительностью: "
            "минуты, секунды, доли часа"
        ),
        "пауза": (
            "прекратить звучание прямо сейчас: остановить проигрывание, "
            "прервать, замолчать"
        ),
        "следующий трек": (
            "перемотать вперёд: пропустить то, что звучит, и перейти "
            "к идущей дальше композиции"
        ),
        "предыдущий трек": (
            "перемотать в обратную сторону: вернуться к композиции, которая "
            "звучала до этой"
        ),
        "музыка": (
            "начать звучание: назвать исполнителя, песню, альбом, жанр "
            "или любой музыкальный запрос"
        ),
        "погода": (
            "узнать, что происходит на улице: температура, дождь, снег, "
            "ветер, прогноз"
        ),
        "время": "узнать, который сейчас час, локально или в другом городе",
        "что играет": "узнать, какая композиция звучит прямо сейчас",
        "намного громче": (
            "резко поднять громкость: заметно, почти до предела, "
            "а не на один шаг"
        ),
        "намного тише": (
            "резко опустить звук: почти до полного исчезновения звука"
        ),
        "громче": "поднять громкость на обычный шаг, без резкого скачка",
        "тише": "опустить громкость на обычный шаг, без резкого скачка",
        "другое": (
            "любое обращение вне списка: приветствие, разговорная фраза, "
            "вопрос о чём-то, что ассистент здесь не делает, просьба "
            "о несуществующем действии"
        ),
    },
)

# terse — минимальная формулировка: скажет ли метка-подпись сама за себя.
TERSE_PROMPT = Prompt(
    instructions="Команда из списка или «другое».",
    criteria={
        "таймеры": "сколько/какие запущенные отсчёты",
        "таймер": "запустить отсчёт на длительность",
        "пауза": "остановить звучание",
        "следующий трек": "перейти к идущей дальше композиции",
        "предыдущий трек": "вернуться к предыдущей композиции",
        "музыка": "начать играть что-нибудь",
        "погода": "узнать погоду на улице",
        "время": "текущие часы и минуты",
        "что играет": "какая композиция звучит сейчас",
        "намного громче": "сильно прибавить громкость",
        "намного тише": "сильно опустить звук",
        "громче": "прибавить громкость",
        "тише": "опустить громкость",
        "другое": "не команда",
    },
)

PROMPTS: dict[str, Prompt] = {"prod": PROD_PROMPT, "rich": RICH_PROMPT, "terse": TERSE_PROMPT}

# Чекпоинты: короткое имя в названии конфига -> значение model= у Router.
# None = авто-роутинг Router (lang_guess="ru", как в продакшене).
MODELS: dict[str, str | None] = {
    "auto": None,
    "multi": "multilingual",
    "en": "english",
    "typed": "typed-decisions",
}

SCORE = {"exact": 2, "wrong": -3, "refused": -1, "fpresume": -2}
GATE_THRESHOLDS = (0.5, 0.7, 0.9)
GATE_METRICS = ("maxp", "entropy")
WARMUP_TEXT = "прогрев роутера"
# Быстрый набор: только конфиги, которые показали себя в прогоне.
# rich (acc 0.43), english (0.25) и typed-decisions (0.32) отбрасываем,
# multilingual повторяет auto побитово — их возвращает --full.
FAST_BASE: tuple[tuple[str, str], ...] = (("auto", "prod"), ("auto", "terse"))
FULL_BASE: tuple[tuple[str, str], ...] = (
    *((short, "prod") for short in MODELS if short != "auto"),
    *(( "auto", prompt) for prompt in ("prod", "rich", "terse")),
)
STAGES = ("base", "gate", "lang")
FULL_STAGES = ("base", "order", "gate", "lang")
BATCH_SIZE = 16
# Сколько одиночных вызовов на конфиг в пробе latency: батчированный отчёт
# меряет время чанка, а не запроса, поэтому p50/p95 берём отсюда.
PROBE_N = 8


# -------------------------------------------------------------------- утечки
def normalize(text: str) -> str:
    return re.sub(r"[\s\".,;!?»«'()\-—:]+", " ", text).strip().lower()


_LABEL_NORM = {normalize(label) for label in LABELS}


def prompt_text(prompt: Prompt) -> str:
    return normalize(prompt.instructions + " " + " ".join(prompt.criteria.values()))


def overlaps(prompt: Prompt, cases: tuple[tuple[str, str], ...]) -> list[str]:
    """Фразы набора, которые лежат в промпте дословно (по границам слов).

    Длина >= 5 отсекает служебные короткие слова, метка отсекает саму себя
    (ключ criteria неизбежно совпадает с меткой).
    """
    text = prompt_text(prompt)
    found = []
    for phrase, _ in cases:
        norm = normalize(phrase)
        if len(norm) < 5 or norm in _LABEL_NORM:
            continue
        if re.search(r"(?<!\w)" + re.escape(norm) + r"(?!\w)", text):
            found.append(phrase)
    return found


def assert_no_leak(prompt: Prompt, name: str, cases: tuple[tuple[str, str], ...]) -> None:
    found = overlaps(prompt, cases)
    if found:
        raise SystemExit(
            f"промпт {name!r} видит тестовые фразы: {found}\n"
            "критерии нужно переписать абстрактно, без цитат из набора"
        )


def lang_of(phrase: str) -> str:
    return "ru" if re.search(r"[а-яё]", phrase, re.IGNORECASE) else "en"


# -------------------------------------------------------------------- конфиги
@dataclass(frozen=True)
class Config:
    name: str
    stage: str
    prompt: str
    model: str | None = None  # None -> авто-роутинг Router
    lang: str | None = None
    gate: float | None = None
    gate_metric: str = "maxp"  # maxp: answer_confidence laya; entropy: confidence
    reverse: bool = False

    @property
    def short_model(self) -> str:
        for short, resolved in MODELS.items():
            if resolved == self.model:
                return short
        return "auto" if self.model is None else self.model


def base_configs(names: tuple[tuple[str, str], ...] = FAST_BASE) -> list[Config]:
    """Базовые конфиги по парам (короткое имя чекпоинта, промпт).

    По умолчанию FAST_BASE — только показавшие себя; --full передаёт FULL_BASE.
    """
    return [
        Config(
            name=f"base:{short}:{prompt}",
            stage="base",
            prompt=prompt,
            model=MODELS[short],
        )
        for short, prompt in names
    ]


def pick_bases(ctx: "Ctx") -> list[Config]:
    """Базовые конфиги, которые питают gate: --gate-on либо top-K по баллу.

    Пусто, если base не запускался: стадии сами печатают, что их пропустили.
    """
    pool = [r.cfg for r in ctx.results.values() if r.cfg.stage == "base"]
    if not pool:
        return []
    if ctx.gate_on:
        by_name = {cfg.name: cfg for cfg in pool}
        missing = [name for name in ctx.gate_on if name not in by_name]
        if missing:
            raise SystemExit(f"--gate-on называет незапущенные конфиги: {missing}")
        return [by_name[name] for name in ctx.gate_on]
    ranked = sorted(pool, key=lambda cfg: (-ctx.results[cfg.name].metrics["score"], cfg.name))
    return ranked[: ctx.top_k]


# ------------------------------------------------------------------- runner
def mark_gate(answer: dict, threshold: float, metric: str) -> None:
    """Контракт laya.confidence.apply_confidence_gate, поле выбирается метрикой.

    maxp — answer_confidence (max p), штатная политика laya. У отгруженных
    чекпоинтов это точечная масса у 1.0, поэтому порог почти не срабатывает.
    entropy — поле confidence (нормированная энтропия): на фиксированном числе
    опций реально разбросана, это отдельная гейт-политика, которую тоже меряем.

    Гейт только маркирует: choice не меняется, поэтому конфиг с гейтом
    получается из уже посчитанного прогона (см. derive_gate) без инференса.
    """
    from laya.confidence import GATE_ABSTAINED, GATE_PASSED, GATE_UNEVALUATED

    conf = answer.get("answer_confidence" if metric == "maxp" else "confidence")
    if isinstance(conf, (int, float)) and not isinstance(conf, bool) and math.isfinite(conf):
        if conf < threshold:
            answer["low_confidence"] = True
            answer["abstention"] = GATE_ABSTAINED
        else:
            answer.pop("low_confidence", None)
            answer["abstention"] = GATE_PASSED
    else:
        answer["abstention"] = GATE_UNEVALUATED
    answer["abstention_threshold"] = float(threshold)


def gate_result(result: dict, threshold: float, metric: str) -> None:
    for answer in (result.get("answers") or {}).values():
        if isinstance(answer, dict):
            mark_gate(answer, threshold, metric)


class BenchRunner:
    """Router в форме, которую ждет laya.evals: язык конфига + метрика гейта."""

    def __init__(self, router, lang: str | None = None, gate_metric: str = "maxp"):
        self.router = router
        self.lang = lang
        self.gate_metric = gate_metric

    def predict(self, state, questions, model=None, min_confidence=None):
        result = self.router.predict(state, questions, model=model, lang=self.lang)
        if min_confidence is not None:
            gate_result(result, min_confidence, self.gate_metric)
        return result

    def predict_batch(
        self, requests, batch_size=None, min_confidence=None, sort_by_length=False
    ):
        """Форма Router.predict_batch, которую ждет laya.evals (словарь на запрос).

        lang в словаре запроса нет — харнесс кладет только state/questions/model,
        поэтому язык конфига дописываем здесь, как это делает predict.
        """
        payload = [dict(request) for request in requests]
        if self.lang is not None:
            for request in payload:
                request["lang"] = self.lang
        results = self.router.predict_batch(
            payload, batch_size=batch_size, sort_by_length=sort_by_length
        )
        if min_confidence is not None:
            for result in results:
                gate_result(result, min_confidence, self.gate_metric)
        return results


# ------------------------------------------------------------------- метрики
def task_metrics(cases: list[dict]) -> dict:
    """exact/wrong/refused/fpresume поверх построчных отчётов laya.evals."""
    counts = {"exact": 0, "wrong": 0, "refused": 0, "fpresume": 0, "abstained": 0}
    for case in cases:
        answer = case["answer"]
        expected = case["expected"]
        if answer.get("abstention") == "abstained":
            counts["abstained"] += 1
        got = answer.get("choice")
        if answer.get("abstention") == "abstained" or got == OTHER:
            got = OTHER
        if got == expected:
            counts["exact"] += 1
        elif got == OTHER:
            counts["refused"] += 1
        elif expected == OTHER:
            counts["fpresume"] += 1
        else:
            counts["wrong"] += 1
    counts["n"] = len(cases)
    counts["score"] = sum(counts[key] * weight for key, weight in SCORE.items())
    counts["cov"] = 1.0 - (counts["abstained"] / counts["n"]) if counts["n"] else 0.0
    return counts


def per_label_recall(cases: list[dict]) -> dict[str, tuple[int, int]]:
    recall: dict[str, list[int]] = {}
    for case in cases:
        answer = case["answer"]
        got = OTHER if answer.get("abstention") == "abstained" else answer.get("choice")
        entry = recall.setdefault(case["expected"], [0, 0])
        entry[1] += 1
        if got == case["expected"]:
            entry[0] += 1
    return {label: (ok, total) for label, (ok, total) in recall.items()}


@dataclass
class Result:
    cfg: Config
    report: object
    metrics: dict


def derive_gate(source: Result, cfg: Config) -> Result:
    """Конфиг с гейтом из уже посчитанного отчёта — без повторного инференса.

    Гейт по контракту laya не меняет choice: он пишет только abstention и
    low_confidence, а choice остаётся сырым argmax. Значит, копируем ответы
    лучшего base, ставим отметки тем же mark_gate, task_metrics считаем заново.
    overall (acc/ece/aurc) от порога не зависит — его наследуем, latency
    тоже: прогон один и тот же.
    """
    cases = deepcopy(source.report.cases)
    for case in cases:
        mark_gate(case["answer"], cfg.gate, cfg.gate_metric)
    report = replace(
        source.report,
        cases=cases,
        config=dict(
            source.report.config,
            va_config=asdict(cfg),
            gate_derived_from=source.cfg.name,
        ),
    )
    metrics = task_metrics(cases)
    metrics["p50"], metrics["p95"] = source.metrics["p50"], source.metrics["p95"]
    metrics["secs"] = 0.0
    return Result(cfg=cfg, report=report, metrics=metrics)


# ------------------------------------------------------------------ прогон
def build_examples(cfg: Config, cases: list[tuple[str, str]]):
    from laya.evals import Dataset, Example

    questions = PROMPTS[cfg.prompt].questions(cfg.reverse)
    return Dataset(
        [
            Example(
                state=phrase,
                questions=questions,
                expected={"intent": label},
                tags=(label,),
                language=lang_of(phrase),
                model=cfg.model,
            )
            for phrase, label in cases
        ]
    )


def probe_latency(runner: BenchRunner, questions, cfg: Config, cases, n: int = PROBE_N):
    """(p50, p95) в мс по одиночным вызовам подряд.

    Батчированный отчёт пишет в latency_p50_ms время всего чанка — это не
    latency запроса, а latency пачки. Здесь один запрос за раз, как в проде.
    """
    step = max(1, len(cases) // n)
    waits = []
    for phrase, _ in cases[::step][:n]:
        started = time.perf_counter()
        runner.predict(phrase, questions, model=cfg.model)
        waits.append((time.perf_counter() - started) * 1000.0)
    waits.sort()
    if not waits:
        return float("nan"), float("nan")

    def nearest_rank(pct: float) -> float:
        index = max(0, min(len(waits) - 1, math.ceil(pct / 100 * len(waits)) - 1))
        return waits[index]

    return nearest_rank(50), nearest_rank(95)


def run_config(router, cfg: Config, eval_cases: list[tuple[str, str]], batch_size: int) -> Result:
    from laya.evals import evaluate

    questions = PROMPTS[cfg.prompt].questions(cfg.reverse)
    runner = BenchRunner(router, lang=cfg.lang, gate_metric=cfg.gate_metric)
    # Прогрев: первый вызов по чекпоинту грузит веса, это секунды, и они не
    # должны попадать в latency-пробу.
    runner.predict(WARMUP_TEXT, questions, model=cfg.model)
    started = time.perf_counter()
    report = evaluate(
        runner,
        build_examples(cfg, eval_cases),
        batch_size=batch_size,
        sort_by_length=True,
        min_confidence=cfg.gate,
        config={"va_config": asdict(cfg)},
    )
    metrics = task_metrics(report.cases)
    metrics["p50"], metrics["p95"] = probe_latency(runner, questions, cfg, eval_cases)
    metrics["secs"] = time.perf_counter() - started
    return Result(cfg=cfg, report=report, metrics=metrics)


def _log(message: str) -> None:
    print(f"  .. {message}", file=sys.stderr)


# -------------------------------------------------------------------- вывод
HEADER = (
    f"{'config':26} {'score':>7} {'ex':>4}{'wr':>4}{'rf':>4}{'fp':>4}"
    f" {'cov':>5} {'acc':>5} {'ece':>5} {'aurc':>5} {'p50ms':>7}{'p95ms':>7} {'lk':>3}"
)


def leak_count(cfg: Config, eval_cases: list[tuple[str, str]]) -> int:
    return len(overlaps(PROMPTS[cfg.prompt], tuple(eval_cases)))


def print_table(results: list[Result], eval_cases: list[tuple[str, str]]) -> None:
    print(HEADER)
    for res in results:
        overall = res.report.overall
        acc = overall.get("choice_accuracy", float("nan"))
        print(
            f"{res.cfg.name:26} {res.metrics['score']:7.1f}"
            f" {res.metrics['exact']:4d}{res.metrics['wrong']:4d}"
            f"{res.metrics['refused']:4d}{res.metrics['fpresume']:4d}"
            f" {res.metrics['cov']:5.2f} {acc:5.2f}"
            f" {overall.get('ece', float('nan')):5.2f}"
            f" {overall.get('aurc', float('nan')):5.2f}"
            f" {res.metrics['p50']:7.1f}"
            f" {res.metrics['p95']:7.1f}"
            f" {leak_count(res.cfg, eval_cases):3d}"
        )
    print()


def print_detail(res: Result, eval_cases: list[tuple[str, str]]) -> None:
    cfg = res.cfg
    print(f"\n=== {cfg.name} ===")
    print(f"    {asdict(cfg)}")
    print(f"    балл {res.metrics['score']:.1f} при n={res.metrics['n']}, "
          f"отказов гейта {res.metrics['abstained']}")
    recall = per_label_recall(res.report.cases)
    print("    по меткам:")
    for label in LABELS:
        ok, total = recall.get(label, (0, 0))
        mark = "  " if ok == total else "!!"
        print(f"      {mark} {label:18} {ok}/{total}")
    printed = 0
    print("    ошибки:")
    for (phrase, expected), case in zip(eval_cases, res.report.cases):
        answer = case["answer"]
        got = OTHER if answer.get("abstention") == "abstained" else answer.get("choice")
        if got == expected:
            continue
        printed += 1
        conf = case.get("confidence")
        conf_s = f"{conf:.2f}" if isinstance(conf, float) else "?"
        print(f"      xx {expected:16} <- {got:16} {phrase!r} (conf {conf_s})")
    if not printed:
        print("      нет")
    hits = overlaps(PROMPTS[cfg.prompt], tuple(eval_cases))
    if hits:
        print(f"    дословно в промпте ({len(hits)}): {', '.join(hits)}")


def print_summary(results: list[Result]) -> None:
    ranked = sorted(results, key=lambda r: (-r.metrics["score"], r.cfg.name))
    print("=== итог: топ-5 по баллу ===")
    for rank, res in enumerate(ranked[:5], 1):
        print(f"  {rank}. {res.cfg.name:26} {res.metrics['score']:7.1f}  {asdict(res.cfg)}")
    best = ranked[0]
    print(f"\nлучший конфиг: {best.cfg.name}")
    print(f"  {json.dumps(asdict(best.cfg), ensure_ascii=False)}")


# -------------------------------------------------------------------- стадии
@dataclass
class Ctx:
    router: object
    ev: list[tuple[str, str]]
    top_k: int
    gate_on: list[str] | None
    batch_size: int
    want: object  # predicate по имени конфига: позиционные фильтры CLI
    base_names: tuple[tuple[str, str], ...]
    results: dict[str, Result]
    ran: list[Result]


def _record(ctx: Ctx, res: Result, note: str = "") -> None:
    ctx.results[res.cfg.name] = res
    ctx.ran.append(res)
    acc = res.report.overall.get("choice_accuracy", float("nan"))
    print(
        f"  .. {res.cfg.name:30} балл {res.metrics['score']:7.1f} "
        f"acc {acc:.2f} {res.metrics['secs']:5.1f}s{note}",
        flush=True,
    )


def _execute(ctx: Ctx, cfg: Config) -> None:
    if cfg.name in ctx.results or not ctx.want(cfg.name):
        return
    started = time.perf_counter()
    res = run_config(ctx.router, cfg, ctx.ev, ctx.batch_size)
    res.metrics["secs"] = time.perf_counter() - started
    _record(ctx, res)


def stage_base(ctx: Ctx) -> None:
    for cfg in base_configs(ctx.base_names):
        _execute(ctx, cfg)


def stage_order(ctx: Ctx) -> None:
    source = _best(ctx, "base")
    if source is None:
        return
    cfg = replace(
        source, name=f"order:{source.short_model}:{source.prompt}-rev", stage="order", reverse=True
    )
    _execute(ctx, cfg)
    if cfg.name not in ctx.results:
        return
    delta = ctx.results[cfg.name].metrics["score"] - ctx.results[source.name].metrics["score"]
    _log(f"обратный порядок опций: {source.name} -> {cfg.name} (Δ балл {delta:+.1f})")


def stage_gate(ctx: Ctx) -> None:
    picks = pick_bases(ctx)
    if not picks:
        print("  .. gate: нет результатов base, пропускаю (запусти base или укажи --gate-on)")
        return
    _log("gate от ответов: " + ", ".join(c.name for c in picks))
    for source in picks:
        for threshold in GATE_THRESHOLDS:
            for metric in GATE_METRICS:
                suffix = "m" if metric == "maxp" else "e"
                cfg = replace(
                    source,
                    name=f"gate:{source.short_model}:{source.prompt}@{threshold:.1f}{suffix}",
                    stage="gate",
                    gate=threshold,
                    gate_metric=metric,
                )
                if cfg.name in ctx.results or not ctx.want(cfg.name):
                    continue
                _record(ctx, derive_gate(ctx.results[source.name], cfg), note="  без прогона")


def stage_lang(ctx: Ctx) -> None:
    source = _best(ctx, "base")
    if source is None:
        return
    cfg = replace(
        source, name=f"lang:ru:{source.short_model}:{source.prompt}", stage="lang", lang="ru"
    )
    _execute(ctx, cfg)
    if cfg.name not in ctx.results:
        return
    delta = ctx.results[cfg.name].metrics["score"] - ctx.results[source.name].metrics["score"]
    _log(f"lang='ru': {source.name} -> {cfg.name} (Δ балл {delta:+.1f})")


def _best(ctx: Ctx, stage: str) -> Config | None:
    pool = [r for r in ctx.results.values() if r.cfg.stage == stage]
    if not pool:
        print(f"  .. стадия: нет результатов {stage!r}, пропускаю")
        return None
    return sorted(pool, key=lambda r: (-r.metrics["score"], r.cfg.name))[0].cfg


STAGE_FUNCS = {
    "base": stage_base,
    "order": stage_order,
    "gate": stage_gate,
    "lang": stage_lang,
}


# ----------------------------------------------------------------------- cli
def print_catalog(base_names, stages) -> None:
    print(f"золотой набор: {len(CASES)} фраз, меток {len(LABELS)} (идёт в eval целиком)")
    counts: dict[str, int] = {}
    for _, label in CASES:
        counts[label] = counts.get(label, 0) + 1
    for label in LABELS:
        print(f"  {label:18} {counts.get(label, 0):3d}")
    print()
    for name, prompt in PROMPTS.items():
        hits = overlaps(prompt, tuple(CASES))
        flag = "OK" if not hits else "ЛИК"
        print(f"промпт {name:6} [{flag}] фраз набора в промпте: {len(hits)}")
        for phrase in hits:
            print(f"    {phrase!r}")
    print()
    print(f"стадии ({', '.join(stages)}):")
    print("  base  базовые конфиги (см. ниже)")
    print("  order обратный порядок criteria у лучшего base (--full)")
    print(f"  gate  min_confidence {GATE_THRESHOLDS} x метрики {GATE_METRICS} на top-K base")
    print("  lang  lang='ru' против авто-роутинга")
    print()
    print(f"конфиги base ({len(base_names)}, --full даёт {len(FULL_BASE)}):")
    for cfg in base_configs(base_names):
        hits = overlaps(PROMPTS[cfg.prompt], tuple(CASES))
        print(f"  {cfg.name:26} model={cfg.model} lk={len(hits)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Бенчмарк интентов на laya")
    parser.add_argument(
        "filters", nargs="*",
        help="запускать только конфиги, чьё имя содержит (фильтрует и прогон, не только таблицу)",
    )
    parser.add_argument("--list", action="store_true", help="данные, промпты, план без прогона")
    parser.add_argument("--full", action="store_true",
                        help="все 6 базовых конфигов и стадия order (по умолчанию — быстрый набор)")
    parser.add_argument("--stages", default="", help="подмножество стадий (по умолчанию все)")
    parser.add_argument("--top-k", type=int, default=2, help="сколько base конфигов питает gate")
    parser.add_argument("--gate-on", default="", help="имена base конфигов для gate вместо top-K")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="сколько запросов в одном forward pass (1 — без батчей)")
    parser.add_argument("--json", default="", help="куда сложить полные отчёты laya.evals")
    parser.add_argument("--detail", default="", help="построчный разбор одного конфига")
    args = parser.parse_args()

    base_names = FULL_BASE if args.full else FAST_BASE
    default_stages = FULL_STAGES if args.full else STAGES
    stages = (
        [s.strip() for s in args.stages.split(",") if s.strip()]
        if args.stages
        else list(default_stages)
    )

    if args.list:
        print_catalog(base_names, stages)
        return

    unknown = [s for s in stages if s not in STAGE_FUNCS]
    if unknown:
        parser.error(f"неизвестные стадии: {unknown}; доступны {list(STAGE_FUNCS)}")
    gate_on = [s.strip() for s in args.gate_on.split(",") if s.strip()] or None
    batch_size = max(1, args.batch_size)
    want = (
        (lambda name: any(f in name for f in args.filters))
        if args.filters
        else (lambda name: True)
    )

    import laya
    from laya import Router

    ev = list(CASES)
    prod_hits = overlaps(PROD_PROMPT, tuple(CASES))
    print(
        f"laya {laya.__version__} | набор {len(CASES)} фраз (eval целиком) "
        f"| батч {batch_size} "
        f"| промпты: prod lk={len(prod_hits)}, rich lk={len(overlaps(RICH_PROMPT, tuple(CASES)))}, "
        f"terse lk={len(overlaps(TERSE_PROMPT, tuple(CASES)))} "
        "(lk — дословные цитаты набора в промпте)",
        flush=True,
    )

    # max_loaded >= число чекпоинтов: вытеснение посреди eval увело бы
    # перезагрузку весов в замеры.
    router = Router(lang_guess="ru", max_loaded=len(MODELS))
    ctx = Ctx(router=router, ev=ev, top_k=args.top_k, gate_on=gate_on,
              batch_size=batch_size, want=want, base_names=base_names,
              results={}, ran=[])

    for stage in stages:
        print(f"\n=== {stage} ===", flush=True)
        before = len(ctx.results)
        STAGE_FUNCS[stage](ctx)
        ran = ctx.ran[before:]
        if not ran:
            print("  .. конфигов не запущено")
            continue
        print_table(ran, ctx.ev)

    results = list(ctx.results.values())
    if not results:
        print(f"под фильтр {args.filters} ничего не подошло")
        return

    print_summary(results)

    if args.detail:
        res = ctx.results.get(args.detail)
        if res is None:
            print(f"конфиг {args.detail!r} не запущен; доступны: {', '.join(ctx.results)}")
        else:
            print_detail(res, ctx.ev)

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "laya_version": laya.__version__,
            "batch_size": batch_size,
            "cases": {"all": list(CASES)},
            "results": {
                name: {"config": asdict(res.cfg), "task": res.metrics,
                       "report": res.report.to_json()}
                for name, res in ctx.results.items()
            },
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nотчёты: {path}")


if __name__ == "__main__":
    main()
