"""Бенчмарк классификаторов интентов на золотом наборе фраз.

Варианты:
- baseline   — текущий intent.classify (ключевые слова + difflib-фолбэк);
- llm        — Qwen (GGUF) few-shot, ответ модели берём как есть;
- llm-gate   — llm + гейт: распознаём метку из закрытого списка, иначе Unknown;
- llm-gate-15b / llm-gate-3b — то же на Qwen2.5-1.5B / Qwen2.5-3B;
- llm-tools / llm-tools-15b / llm-tools-3b — tool calling (Qwen умеет):
  модель сама выбирает функцию (play_music/get_weather/...) и слоты,
  отсутствие вызова = Unknown;
- laya       — System-1 decision-модель (choice) в рантайме laya;
- laya-gate  — laya + confidence-гейт: если уверенность ниже порога, Unknown.

Одной строкой таблица + время (avg/p95 на кейс, сумма) каждой модели,
подробно — построчный разбор и recall по интентам.

Запуск (llama-cpp-python и наташа — зависимости проекта, laya — оверлей):
  uv run --project . \
      --index https://download.pytorch.org/whl/cpu \
      --with "torch==2.14.0+cpu" --with "transformers>=5.17.0" \
      --with "laya" --with "laya[onnx]>=0.3.20" \
      python scripts/benchmark_classify.py

Категория по кейсу (классификатор всегда возвращает Intent):
- exact     — метка совпала с ожидаемой;
- wrong     — вернул метку, но не ту (выполнит не то — самый вредный);
- refused   — ждали интент, вернул Unknown (не понял);
- fpresume  — ждали Unknown, вернул какой-то интент (придумал).

Балл: exact +2, wrong -3, refused -0.5, fpresume -2.
"""

from __future__ import annotations

import contextlib
import enum
import io
import json
import re
import sys
import time
from statistics import mean

from va import intent
from va.intent import Intent

MODELS_DIR = "models"
LLM_MODELS = {
    "anchor": f"{MODELS_DIR}/llm/qwen2.5-0.5b-instruct-q4_k_m.gguf",
    "q25-15b": f"{MODELS_DIR}/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    "q25-3b": f"{MODELS_DIR}/llm/qwen2.5-3b-instruct-q4_k_m.gguf",
}

# Закрытый список меток: (русская метка -> Intent). Классификатор отвечает
# ровно одной меткой из этого списка (или её английским именем).
INTENT_LABELS: list[tuple[str, Intent]] = [
    ("музыка", Intent.PlayMusic),
    ("погода", Intent.Weather),
    ("время", Intent.Time),
    ("следующий трек", Intent.NextTrack),
    ("предыдущий трек", Intent.PreviousTrack),
    ("пауза", Intent.PauseMusic),
    ("что играет", Intent.NowPlaying),
    ("громче", Intent.VolumeUp),
    ("тише", Intent.VolumeDown),
    ("намного громче", Intent.VolumeMuchUp),
    ("намного тише", Intent.VolumeMuchDown),
    ("другое", Intent.Unknown),
]
LABEL_TEXT = ", ".join(label for label, _ in INTENT_LABELS)

LLM_SYSTEM = (
    "Ты классифицируешь короткую реплику пользователя для голосового "
    f"ассистента. Отвечай строго одной меткой из списка: {LABEL_TEXT}."
)
LLM_EXAMPLES = """Реплики и ответы:
"включи платинум" -> музыка
"поставь the void" -> музыка
"громче сделай" -> громче
"сделай потише" -> намного тише
"погромче" -> намного громче
"какая погода в москве" -> погода
"сколько времени" -> время
"пауза" -> пауза
"следующий трек" -> следующий трек
"что играет" -> что играет
"какой сегодня день" -> другое"""

# Qwen3.5 может обернуть ответ в thinking-блоки — срезаем их.
_THINK_TAGS_RE = r"<\|begin_of_think\|>.*?<\|end_of_think\|>|\[thinking\].*?\[/thinking\]"
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "play_music",
            "description": "Включить музыку/трек/исполнителя в YouTube Music",
            "parameters": {
                "type": "object",
                "properties": {"song": {"type": "string", "description": "Название трека или исполнителя"}},
                "required": ["song"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Узнать погоду в городе",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string", "description": "Город"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Узнать время в городе или локально",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "control_player",
            "description": "Управление проигрывателем",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": [
                            "pause", "next", "previous", "volume_up", "volume_down",
                            "volume_much_up", "volume_much_down", "now_playing",
                        ],
                    }
                },
                "required": ["command"],
            },
        },
    },
]

_PLAYER_COMMANDS = {
    "pause": Intent.PauseMusic,
    "next": Intent.NextTrack,
    "previous": Intent.PreviousTrack,
    "volume_up": Intent.VolumeUp,
    "volume_down": Intent.VolumeDown,
    "volume_much_up": Intent.VolumeMuchUp,
    "volume_much_down": Intent.VolumeMuchDown,
    "now_playing": Intent.NowPlaying,
}
_TOOL_TO_INTENT = {
    "play_music": Intent.PlayMusic,
    "get_weather": Intent.Weather,
    "get_time": Intent.Time,
}

CASES: list[tuple[str, Intent]] = [
    # PlayMusic
    ("включи платинум", Intent.PlayMusic),
    ("поставь the void", Intent.PlayMusic),
    ("включи песню группы спиритбокс", Intent.PlayMusic),
    ("сыграй что-нибудь", Intent.PlayMusic),
    ("хочу послушать опет", Intent.PlayMusic),
    ("давай опет", Intent.PlayMusic),
    ("врубай спанч боб", Intent.PlayMusic),
    ("включи монгольский рок", Intent.PlayMusic),
    ("включи следующие треки линкин парк", Intent.PlayMusic),
    ("включи ещё опет", Intent.PlayMusic),
    # Weather
    ("какая погода", Intent.Weather),
    ("какая погода в москве", Intent.Weather),
    ("что за окном", Intent.Weather),
    ("погода завтра", Intent.Weather),
    ("будет ли дождь", Intent.Weather),
    ("какой сейчас ветер", Intent.Weather),
    # Time
    ("сколько времени", Intent.Time),
    ("который час", Intent.Time),
    ("во сколько сейчас время", Intent.Time),
    ("сколько время в лондоне", Intent.Time),
    ("который час в нью-йорке", Intent.Time),
    # NextTrack / PreviousTrack
    ("следующий трек", Intent.NextTrack),
    ("следующая песня", Intent.NextTrack),
    ("переключи дальше", Intent.NextTrack),
    ("вперёд", Intent.NextTrack),
    ("предыдущий трек", Intent.PreviousTrack),
    ("предыдущая песня", Intent.PreviousTrack),
    ("назад", Intent.PreviousTrack),
    ("верни назад", Intent.PreviousTrack),
    # PauseMusic
    ("пауза", Intent.PauseMusic),
    ("поставь на паузу", Intent.PauseMusic),
    ("выключи музыку", Intent.PauseMusic),
    ("стоп", Intent.PauseMusic),
    ("хватит", Intent.PauseMusic),
    # NowPlaying
    ("что играет", Intent.NowPlaying),
    ("что сейчас играет", Intent.NowPlaying),
    ("какая это песня", Intent.NowPlaying),
    # Volume
    ("сделай громче", Intent.VolumeUp),
    ("прибавь звук", Intent.VolumeUp),
    ("louder", Intent.VolumeUp),
    ("сделай тише", Intent.VolumeDown),
    ("убавь звук", Intent.VolumeDown),
    ("quieter", Intent.VolumeDown),
    ("погромче", Intent.VolumeMuchUp),
    ("навались", Intent.VolumeMuchUp),
    ("намного громче", Intent.VolumeMuchUp),
    ("потише", Intent.VolumeMuchDown),
    ("приглуши", Intent.VolumeMuchDown),
    ("намного тише", Intent.VolumeMuchDown),
    # Unknown
    ("расскажи анекдот", Intent.Unknown),
    ("привет", Intent.Unknown),
    ("выключи свет", Intent.Unknown),
    ("поставь таймер на пять минут", Intent.Unknown),
    ("какой сегодня день", Intent.Unknown),
    ("открой браузер", Intent.Unknown),
    ("что такое фотосинтез", Intent.Unknown),
    ("переведи фразу", Intent.Unknown),
    ("сколько будет два плюс два", Intent.Unknown),
    ("заведи будильник на семь утра", Intent.Unknown),
    ("шутку расскажи", Intent.Unknown),
    ("какая сейчас дата", Intent.Unknown),
]

SCORE = {"exact": 2, "wrong": -3, "refused": -0.5, "fpresume": -2}

_LLMS: dict[str, object] = {}


def _get_llm(path: str):
    if path not in _LLMS:
        from llama_cpp import Llama

        _LLMS[path] = Llama(
            model_path=path,
            n_ctx=512,
            n_gpu_layers=0,
            verbose=False,
        )
    return _LLMS[path]


def extract_baseline(s: str) -> Intent:
    with contextlib.redirect_stdout(io.StringIO()):
        return intent.classify(s)


def _lookup_label(text: str) -> Intent | None:
    norm = re.sub(r"^[\s\".,;!?»«]+|[\s\".,;!?»«]+$", "", text).lower()
    for label, value in INTENT_LABELS:
        if norm == label or norm == value.name.lower():
            return value
    return None


def make_llm_classifier(llm, *, verified: bool = True):
    def classify(s: str) -> Intent:
        messages = [
            {"role": "system", "content": LLM_SYSTEM},
            {"role": "user", "content": f"{LLM_EXAMPLES}\n\nРеплика: \"{s}\""},
        ]
        try:
            out = llm.create_chat_completion(
                messages=messages,
                temperature=0,
                max_tokens=32,
                chat_template_kwargs={"enable_thinking": False},
            )
        except TypeError:
            out = llm.create_chat_completion(
                messages=messages, temperature=0, max_tokens=32
            )
        text = out["choices"][0]["message"]["content"]
        text = re.sub(_THINK_TAGS_RE, "", text)

        if verified:
            first_line = text.strip().splitlines()[0] if text.strip() else ""
            return _lookup_label(first_line) or Intent.Unknown
        for label, value in INTENT_LABELS:
            if label in text.lower():
                return value
        return Intent.Unknown

    return classify


def _parse_tool_call(text: str) -> tuple[str, dict] | None:
    match = _TOOL_CALL_RE.search(text)
    if not match:
        return None
    raw = match.group(1)
    # chat-template может продублировать фигурные скобки: {{"name": ...}}.
    if raw.startswith("{{") and raw.endswith("}}"):
        raw = raw[1:-1]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    name = data.get("name")
    args = data.get("arguments") if isinstance(data.get("arguments"), dict) else {}
    if not isinstance(name, str) or not name:
        return None
    return name, args


def make_llm_tools_classifier(llm):
    def classify(s: str) -> Intent:
        messages = [{"role": "user", "content": s}]
        out = llm.create_chat_completion(
            messages=messages,
            tools=TOOLS,
            temperature=0,
            max_tokens=128,
        )
        text = out["choices"][0]["message"].get("content") or ""
        text = re.sub(_THINK_TAGS_RE, "", text)
        parsed = _parse_tool_call(text)
        if parsed is None:
            return Intent.Unknown
        name, _args = parsed
        if name == "control_player":
            return _PLAYER_COMMANDS.get(_args.get("command"), Intent.Unknown)
        return _TOOL_TO_INTENT.get(name, Intent.Unknown)

    return classify


INTENT_CRITERIA = {
    "музыка": "включить, поставить, сыграть песню, трек или исполнителя",
    "погода": "узнать погоду, дождь, ветер, температуру, что за окном",
    "время": "узнать текущее время или час",
    "следующий трек": "переключить на следующий трек или песню дальше",
    "предыдущий трек": "переключить на предыдущий трек, вернуться назад",
    "пауза": "поставить музыку на паузу, выключить музыку, стоп",
    "что играет": "узнать, какая песня или трек сейчас играет",
    "громче": "сделать звук громче",
    "тише": "сделать звук тише",
    "намного громче": "значительно увеличить громкость, погромче",
    "намного тише": "значительно уменьшить громкость, потише",
    "другое": "всё остальное: вопросы, просьбы, не связанные с командами ассистента",
}


def make_laya_classifier(conf_thr: float | None = None):
    from laya import Router

    router = Router(lang_guess="ru")

    def classify(s: str) -> Intent:
        result = router.predict(
            s,
            {
                "intent": {
                    "type": "choice",
                    "instructions": "Какую команду хочет пользователь?",
                    "criteria": dict(INTENT_CRITERIA),
                }
            },
        )
        ans = result["answers"]["intent"]
        choice = ans["choice"]
        conf = ans.get("answer_confidence", ans.get("confidence", 1.0))
        if choice == "другое":
            return Intent.Unknown
        if conf_thr is not None and conf < conf_thr:
            return Intent.Unknown
        return _lookup_label(choice) or Intent.Unknown

    return classify


def summarize(results: list[Intent]) -> dict[str, float]:
    counts = {"exact": 0, "wrong": 0, "refused": 0, "fpresume": 0}
    for got, (_, expected) in zip(results, CASES):
        if got == expected:
            counts["exact"] += 1
        elif got is Intent.Unknown:
            counts["refused"] += 1
        elif expected is Intent.Unknown:
            counts["fpresume"] += 1
        else:
            counts["wrong"] += 1
    counts["score"] = sum(counts[k] * w for k, w in SCORE.items())
    return counts


def per_intent_recall(results: list[Intent]) -> dict[Intent, tuple[int, int]]:
    recall: dict[Intent, list[int]] = {}
    for got, (_, expected) in zip(results, CASES):
        entry = recall.setdefault(expected, [0, 0])
        entry[1] += 1
        if got == expected:
            entry[0] += 1
    return {k: (v[0], v[1]) for k, v in recall.items()}


def run(classifier) -> tuple[list[Intent], list[float]]:
    times = []
    results = []
    for s, _ in CASES:
        t0 = time.perf_counter()
        results.append(classifier(s))
        times.append(time.perf_counter() - t0)
    return results, times


def p95(times: list[float]) -> float:
    ordered = sorted(times)
    return ordered[min(len(ordered) - 1, int(len(ordered) * 0.95) - 1)]


def fmt_enum(v) -> str:
    if isinstance(v, enum.Enum):
        return v.name
    return str(v)


def main() -> None:
    args = sys.argv[1:]

    selected = {
        "baseline": lambda: extract_baseline,
        "llm": lambda: make_llm_classifier(_get_llm(LLM_MODELS["anchor"]), verified=False),
        "llm-gate": lambda: make_llm_classifier(_get_llm(LLM_MODELS["anchor"]), verified=True),
        "llm-gate-15b": lambda: make_llm_classifier(
            _get_llm(LLM_MODELS["q25-15b"]), verified=True
        ),
        "llm-gate-3b": lambda: make_llm_classifier(
            _get_llm(LLM_MODELS["q25-3b"]), verified=True
        ),
        "llm-tools": lambda: make_llm_tools_classifier(_get_llm(LLM_MODELS["anchor"])),
        "llm-tools-15b": lambda: make_llm_tools_classifier(
            _get_llm(LLM_MODELS["q25-15b"])
        ),
        "llm-tools-3b": lambda: make_llm_tools_classifier(
            _get_llm(LLM_MODELS["q25-3b"])
        ),
        "laya": lambda: make_laya_classifier(),
        "laya-gate": lambda: make_laya_classifier(conf_thr=0.6),
    }
    names = list(selected)
    if args:
        unknown = [n for n in args if n not in selected]
        if unknown:
            print(f"неизвестные варианты: {unknown}; доступны: {', '.join(selected)}")
        else:
            names = args

    print(f"{'variant':11} {'exact':>5} {'wrong':>5} {'refus':>5} {'fpres':>5}"
          f" {'score':>6} {'avg_ms':>7} {'p95_ms':>7} {'total_s':>7}")
    runs: dict[str, tuple[list[Intent], list[float]]] = {}
    for name in names:
        results, times = run(selected[name]())
        runs[name] = (results, times)
        c = summarize(results)
        print(
            f"{name:11} {c['exact']:5.0f} {c['wrong']:5.0f} {c['refused']:5.0f} "
            f"{c['fpresume']:5.0f} {c['score']:6.1f} "
            f"{mean(times) * 1000:7.1f} {p95(times) * 1000:7.1f} "
            f"{sum(times):7.1f}"
        )
    print(f"\nметки: {LABEL_TEXT}")
    if args and set(args).issubset(selected):
        for name in names:
            results, _ = runs[name]
            print(f"\n=== {name} ===")
            recall = per_intent_recall(results)
            for expected, (ok, total) in recall.items():
                print(f"  {expected.name:16} {ok}/{total}")
            for (s, expected), got in zip(CASES, results):
                mark = "ok " if got == expected else " xx"
                print(f"{mark} {fmt_enum(got):20} {s!r} (exp {fmt_enum(expected)})")


if __name__ == "__main__":
    main()