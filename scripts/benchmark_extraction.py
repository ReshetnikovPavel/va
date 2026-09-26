"""Бенчмарк экстракторов названия музыки на золотом наборе фраз.

Самодостаточный: вся логика (лемматизация, natasha, триггеры, LLM-экстрактор)
определена прямо здесь, без импортов из пакета va.

Варианты:
- baseline — текущая позиционная логика из extract_data;
- syntax  — синтаксис natasha (поддерево глагола-триггера), локальный код;
- qaN     — RuBERT-QA (SberQuAD) с разными формулировками вопроса;
- llm     — Qwen (GGUF), few-shot, локальный экстрактор;
- llm-sub — llm + детерминированный substring-гейт (только дословная фраза);
- llm-sub-08b / llm-sub-2b — то же самое на Qwen3.5-0.8B / Qwen3.5-2B;
- llm-sub-15b / llm-sub-3b — то же самое на Qwen2.5-1.5B / Qwen2.5-3B.

В сводке — время: среднее и p95 на кейс, сумма по набору.

Запуск (llama-cpp-python, natasha, pymorphy3 — зависимости проекта, torch и
transformers — наташа/QA-варианты, подтягиваются оверлеем):
  uv run --project . \
      --index https://download.pytorch.org/whl/cpu \
      --with "torch==2.14.0+cpu" --with "transformers>=5.17.0" \
      python scripts/benchmark_extraction.py [вариант...] [--log ПУТЬ]

С `--log ПУТЬ` сохраняет построчный разбор (вход -> ожидание -> ответы всех
вариантов) в файл.

Метрики по кейсу: exact / garbage (вернул не то) / miss (ожидали имя, дал None)
/ fpresume (ожидали None, дал имя). Балл: exact +2, garbage -3, miss -0.5,
fpresume -1.
"""

from __future__ import annotations

import importlib.metadata
import re
import sys
import time
from statistics import mean

# natasha использует устаревший pkg_resources; monkey-patch вручную.
import pymorphy2.analyzer as _pymorphy2_analyzer

_pymorphy2_analyzer._iter_entry_points = lambda *groups, **_: (  # type: ignore[attr-defined]
    importlib.metadata.entry_points(group=groups[0])
)

import natasha  # noqa: E402
import pymorphy3.analyzer  # noqa: E402

NATASHA_SEGMENTER = natasha.Segmenter()
NATASHA_MORPH_VOCAB = natasha.MorphVocab()
NATASHA_EMBEDDING = natasha.NewsEmbedding()
NATASHA_MORPH_TAGGER = natasha.NewsMorphTagger(NATASHA_EMBEDDING)
NATASHA_SYNTAX_PARSER = natasha.NewsSyntaxParser(NATASHA_EMBEDDING)

_MORPH = pymorphy3.analyzer.MorphAnalyzer()


def _lemmatize(word: str) -> str:
    return _MORPH.parse(word)[0].normal_form


_PUNCT_RE = re.compile(r"[^\w\s]")


def _remove_punctuation(s: str) -> str:
    return _PUNCT_RE.sub("", s)


MODELS_DIR = "models"
QA_DIR = f"{MODELS_DIR}/rubert_qa"
LLM_MODELS = {
    "anchor": f"{MODELS_DIR}/llm/qwen2.5-0.5b-instruct-q4_k_m.gguf",
    "q35-08b": f"{MODELS_DIR}/llm/Qwen3.5-0.8B-Q4_K_M.gguf",
    "q35-2b": f"{MODELS_DIR}/llm/Qwen3.5-2B-Q4_K_M.gguf",
    "q25-15b": f"{MODELS_DIR}/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    "q25-3b": f"{MODELS_DIR}/llm/qwen2.5-3b-instruct-q4_k_m.gguf",
}

PLAY_TRIGGERS = [
    "включить",
    "включать",
    "поставить",
    "постав",
    "сыграть",
    "запустить",
    "завести",
    "врубить",
    "врубать",
    "давать",
    "дать",
    "послушать",
    "слушать",
    "play",
]

PLAY_STOP_LEMMAS = {
    "музыка",
    "песня",
    "песенка",
    "трек",
    "альбом",
    "группа",
    "композиция",
    "пожалуйста",
    "спасибо",
    "пж",
    "плиз",
    "please",
    "я",
    "мы",
    "мой",
    "один",
    "такой",
    "какой",
    "любой",
    "какой-нибудь",
    "снова",
    "потом",
    "сейчас",
    "хотеть",
    "слушать",
    "послушать",
    "выключить",
    "стоп",
    "пауза",
    "хватить",
    "следующий",
    "предыдущий",
    "далёкий",
    "и",
}

# Слова с дефисами, которые remove_punctuation режет до неузнаваемости.
PLAY_STOP_RAW = {"чтонибудь", "чтото", "какоенибудь", "какиенибудь"}

CASES: list[tuple[str, str | None]] = [
    ("включи платинум", "платинум"),
    ("поставь the void", "the void"),
    ("поставь песню группы спиритбокс", "спиритбокс"),
    ("сыграй что-нибудь из последнего альбома группы опет", "опет"),
    ("платинум хочу послушать", "платинум"),
    ("хочу послушать платинум", "платинум"),
    ("дашь платинум", "платинум"),
    ("включи нью дивайд линкин парк", "нью дивайд линкин парк"),
    ("включи следующий трек опет", "опет"),
    (
        "включи первую песню из нового альбома группы спиритбокс",
        "спиритбокс",
    ),
    ("поставь любимую песню линкин парк", "линкин парк"),
    ("включи песню группу криденс", "криденс"),
    ("послушай новый альбом опета", "опета"),
    ("хочу послушать группу эйвэйз", "эйвэйз"),
    ("включи лучше что-нибудь из опета", "опета"),
    ("включи лучше платинум", "платинум"),
    ("поставь play ролс", "play ролс"),
    ("включи the void и потом опет", "the void"),
    ("включи эйвэйз и опет", "эйвэйз"),
    ("включи мне музыку", None),
    ("поставь музыку", None),
    ("включи что-нибудь", None),
    ("включи что-то другое", None),
    ("переключи на опет", None),
    ("сыграй что нибудь из старого металла", "старого металла"),
    ("включи монгольский рок", "монгольский рок"),
    ("поставь первый альбом блэк саботаж", "блэк саботаж"),
    ("хочу послушать хаммерфол", "хаммерфол"),
    ("включи песню из альбома мёртвые цветы", "мёртвые цветы"),
    ("включи что-нибудь бодрое", None),
    ("включи это", None),
    ("опет включи пожалуйста", "опет"),
    ("включи пожалуйста опет", "опет"),
    ("можешь включить опет", "опет"),
    ("будь добр включи опет", "опет"),
    ("короче включи опет", "опет"),
    ("включи же опет", "опет"),
    ("включи-ка опет", "опет"),
    ("заведи что-нибудь из отверженных", "отверженных"),
    ("вруби самый свежий трек от малярии", "от малярии"),
    ("дам тебе послушать спиритбокс", "спиритбокс"),
    ("послушаем вечером опет", "опет"),
    ("включи что-нибудь похожее на опет", "опет"),
    ("включи песню под названием стекло", "стекло"),
    ("поставь ту самую с криденс", "криденс"),
    ("хочу включить себе опет", "опет"),
    ("включи ремикс на опет", "опет"),
    ("включи опет на повтор", "опет"),
    ("ассистент включи опет", "опет"),
    ("давай опет", "опет"),
    ("включи следующее", None),
    ("включи следующие треки", None),
    ("включи ещё опет", "опет"),
    ("включи опет а потом криденс", "опет"),
    ("поставь что-нибудь в том же духе", None),
    ("поставь что-нибудь потише", None),
    ("включи как можно громче опет", "опет"),
    ("поставь песню про зомби", None),
    ("дали б послушать криденс", "криденс"),
    ("the void хочу послушать", "the void"),
]

SCORE = {"exact": 2, "garbage": -3, "miss": -0.5, "fpresume": -1}

QA_QUESTIONS = [
    "какую музыку хочет услышать пользователь?",
    "что хочет послушать пользователь?",
    "какой трек просят включить?",
]

EXTRA_STOP = {
    "из",
    "последний",
    "лучше",
    "новый",
    "любимый",
    "первый",
    "старый",
    "что",
    "нибудь",
    "другой",
}
STOP = PLAY_STOP_LEMMAS | EXTRA_STOP


def _syntax_doc(s: str):
    doc = natasha.Doc(s)
    doc.segment(NATASHA_SEGMENTER)
    doc.tag_morph(NATASHA_MORPH_TAGGER)
    doc.parse_syntax(NATASHA_SYNTAX_PARSER)
    return doc


def _verb_chain(token, index) -> str:
    if token.rel != "xcomp":
        return token.id
    parent = index.get(token.head_id)
    if parent is None or (
        parent.text in PLAY_TRIGGERS or _lemmatize(parent.text) in PLAY_TRIGGERS
    ):
        return token.id
    return parent.id


def extract_song(s: str) -> str | None:
    doc = _syntax_doc(s)
    tokens = list(doc.syntax.tokens)
    positions = {t.id: i for i, t in enumerate(tokens)}
    index = {t.id: t for t in tokens if t.id != t.head_id or t.rel != "root"}

    def lemma(t) -> str:
        return _lemmatize(t.text)

    def is_stop(word: str, word_lemma: str) -> bool:
        norm = word.replace("-", "")
        return (
            norm in PLAY_STOP_RAW
            or word in STOP
            or norm in STOP
            or word_lemma in STOP
        )

    triggers = [
        (i, t)
        for i, t in enumerate(tokens)
        if t.text in PLAY_TRIGGERS or lemma(t) in PLAY_TRIGGERS
    ]
    if not triggers:
        return None

    def subtree(root_id: str) -> list:
        def reaches(tid: str) -> bool:
            visited: set[str] = set()
            cur = tid
            while cur != root_id:
                if cur in visited or cur not in index:
                    return False
                visited.add(cur)
                cur = index[cur].head_id
            return True

        return [t for t in tokens if reaches(t.id)]

    def content(sub: list) -> int:
        return sum(1 for t in sub if not is_stop(t.text, lemma(t)))

    best = max(
        triggers,
        key=lambda p: (content(subtree(_verb_chain(p[1], index))), p[0]),
    )[1]
    root_id = _verb_chain(best, index)
    base = subtree(root_id) or [best]
    lo = min(positions[t.id] for t in base)
    hi = max(positions[t.id] for t in base)
    while hi + 1 < len(tokens) and not is_stop(
        tokens[hi + 1].text, lemma(tokens[hi + 1])
    ):
        hi += 1
    kept: list[str] = []
    for t in tokens:
        if not (lo <= positions[t.id] <= hi):
            continue
        if t.id == best.id or t.rel == "punct":
            continue
        if t.rel == "cc":
            break
        if is_stop(t.text, lemma(t)):
            continue
        kept.append(t.text)
    query = " ".join(kept)
    return query or None


def extract_baseline(s: str) -> str | None:
    words = _remove_punctuation(s).lower().split()
    lemmas = [_lemmatize(word) for word in words]
    index = next(
        (
            i
            for i, (word, lemma) in enumerate(zip(words, lemmas))
            if word in PLAY_TRIGGERS or lemma in PLAY_TRIGGERS
        ),
        None,
    )
    if index is not None:
        words = words[index + 1 :]
        lemmas = lemmas[index + 1 :]
    query = " ".join(
        word
        for word, lemma in zip(words, lemmas)
        if word not in PLAY_STOP_RAW
        and lemma not in PLAY_STOP_LEMMAS
        and word not in PLAY_STOP_LEMMAS
    )
    return query or None


def make_qa_extractor(idx: int):
    import torch
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(QA_DIR)
    model = AutoModelForQuestionAnswering.from_pretrained(QA_DIR)

    def extract(s: str) -> str | None:
        question = QA_QUESTIONS[idx]
        inputs = tokenizer(question, s, return_tensors="pt", truncation=True)
        with torch.no_grad():
            out = model(**inputs)
        start = int(out.start_logits.argmax())
        end = int(out.end_logits.argmax())
        if start > end or end >= inputs["input_ids"].shape[1]:
            return None
        answer = tokenizer.decode(
            inputs["input_ids"][0, start : end + 1],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        if _is_generic(answer):
            return None
        return answer or None

    return extract


def _is_generic(answer: str) -> bool:
    words = [_lemmatize(w) for w in _remove_punctuation(answer).lower().split()]
    if not words or all(w in STOP or w in PLAY_STOP_LEMMAS for w in words):
        return True
    return answer.lower() in {"песня", "песню", "музыка", "музыку"}


LLM_SYSTEM_PROMPT = (
    "Ты помогаешь извлечь название трека или исполнителя из реплики "
    "пользователя для поиска в YouTube Music. Отвечай строго одной строкой: "
    "название или НЕТ. Название должно быть дословной фразой из самой реплики "
    "пользователя."
)

LLM_EXAMPLES = """
Фразы и ответы:
"включи песню группы спиритбокс" -> спиритбокс
"хочу послушать платинум" -> платинум
"поставь музыку" -> НЕТ
"""

_THINK_TAGS_RE = re.compile(r"<(?:\s*/?)(?:think|reasoning|thinking)[^>]*>", re.I)


def _span_of(haystack: str, needle: str) -> str | None:
    """Дословная непрерывная фраза из haystack, совпавшая с needle."""
    h_words = haystack.split()
    n_words = needle.split()
    if not n_words:
        return None
    n_low = [w.lower() for w in n_words]
    for i in range(len(h_words) - len(n_words) + 1):
        if [w.lower() for w in h_words[i : i + len(n_words)]] == n_low:
            return " ".join(h_words[i : i + len(n_words)])
    return None


def make_song_extractor(llm, verified: bool = False):
    def extract(s: str) -> str | None:
        user = f'{LLM_EXAMPLES}\n\nФраза: "{s}"\nОтвет:'
        messages = [
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        out = llm.create_chat_completion(
            messages=messages,
            temperature=0,
            max_tokens=32,
        )
        text = out["choices"][0]["message"]["content"]
        if not text:
            return None
        text = _THINK_TAGS_RE.sub("", text).strip().splitlines()[0]
        if not text or text.upper() == "НЕТ":
            return None
        text = " ".join(text.split())
        if verified:
            return _span_of(s, text)
        return text

    return extract


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


def summarize(results: list[str | None]) -> dict[str, float]:
    counts = {"exact": 0, "garbage": 0, "miss": 0, "fpresume": 0}
    for got, (_, expected) in zip(results, CASES):
        if got == expected:
            counts["exact"] += 1
        elif got is None:
            counts["miss"] += 1
        elif expected is None:
            counts["fpresume"] += 1
        else:
            counts["garbage"] += 1
    counts["score"] = sum(
        counts[k] * w for k, w in SCORE.items()
    )
    return counts


def run(extractor) -> tuple[list[str | None], list[float]]:
    times = []
    results = []
    for s, _ in CASES:
        t0 = time.perf_counter()
        results.append(extractor(s))
        times.append(time.perf_counter() - t0)
    return results, times


def p95(times: list[float]) -> float:
    ordered = sorted(times)
    return ordered[min(len(ordered) - 1, int(len(ordered) * 0.95) - 1)]


def fmt(r) -> str:
    if r is None:
        return "~"
    return repr(r)


def main() -> None:
    argv = sys.argv[1:]
    log_path: str | None = None
    if "--log" in argv:
        i = argv.index("--log")
        if i + 1 >= len(argv):
            print("--log требует путь к файлу")
            return
        log_path = argv[i + 1]
        argv = argv[:i] + argv[i + 2 :]
    args = argv

    def make_syntax():
        return extract_song

    selected = {
        "baseline": lambda: extract_baseline,
        "syntax": make_syntax,
        **{f"qa{i + 1}": (lambda i=i: make_qa_extractor(i)) for i in range(len(QA_QUESTIONS))},
        "llm": lambda: make_song_extractor(_get_llm(LLM_MODELS["anchor"]), verified=False),
        "llm-sub": lambda: make_song_extractor(_get_llm(LLM_MODELS["anchor"]), verified=True),
        "llm-sub-08b": lambda: make_song_extractor(_get_llm(LLM_MODELS["q35-08b"]), verified=True),
        "llm-sub-2b": lambda: make_song_extractor(_get_llm(LLM_MODELS["q35-2b"]), verified=True),
        "llm-sub-15b": lambda: make_song_extractor(_get_llm(LLM_MODELS["q25-15b"]), verified=True),
        "llm-sub-3b": lambda: make_song_extractor(_get_llm(LLM_MODELS["q25-3b"]), verified=True),
    }
    names = list(selected)
    if args:
        unknown = [n for n in args if n not in selected]
        if unknown:
            print(
                f"неизвестные варианты: {unknown}; "
                f"доступны: {', '.join(selected)}"
            )
        else:
            names = args

    print(f"{'variant':11} {'exact':>5} {'garb':>4} {'miss':>4} {'fpres':>5}"
          f" {'score':>6} {'avg_ms':>7} {'p95_ms':>7} {'total_s':>7}")
    runs: dict[str, tuple[list[str | None], list[float]]] = {}
    for name in names:
        results, times = run(selected[name]())
        runs[name] = (results, times)
        c = summarize(results)
        print(
            f"{name:11} {c['exact']:5d} {c['garbage']:4d} {c['miss']:4d} "
            f"{c['fpresume']:5d} {c['score']:6.1f} "
            f"{mean(times) * 1000:7.1f} {p95(times) * 1000:7.1f} "
            f"{sum(times):7.1f}"
        )
    print(f"\nвопросы QA: {QA_QUESTIONS}")
    if log_path:
        _write_log(log_path, names, runs)


def _write_log(
    path: str,
    names: list[str],
    runs: dict[str, tuple[list[str | None], list[float]]],
) -> None:
    import os

    runs_list = [(name, r[0]) for name, r in runs.items()]
    lines: list[str] = []
    for i, (s, expected) in enumerate(CASES):
        lines.append(f"[{i:02d}] in:  {s!r}")
        lines.append(f"      exp: {fmt(expected)}")
        for name, results in runs_list:
            got = results[i]
            mark = "ok" if got == expected else "xx"
            lines.append(f"      {mark} {name:11} -> {fmt(got)}")
        lines.append("")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"\nлог сохранён: {path}")


if __name__ == "__main__":
    main()