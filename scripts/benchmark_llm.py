"""Бенчмарк производительности установленных LLM (GGUF) на русских промптах.

Оценивается только скорость, без качества ответов. Каждая модель поднимается
через llama-cpp-python на CPU (n_gpu_layers=0, как в src/va/models.py),
прогревается и прогоняется по набору русских реплик с продакшн-системным
промптом Ады из src/va/pipeline.py. Генерация потоковая, поэтому отдельно
видно время до первого токена (TTFT, включает препроцессинг промпта) и чистую
скорость генерации (tokens/s) без препроцессинга.

Метрики по модели:
  size_mb   — размер файла весов на диске;
  load_s    — время загрузки весов (cold start, вне замеров генерации);
  ttft50/95 — время до первого токена, p50/p95, мс;
  tot50/95  — полное время ответа, p50/p95, мс;
  gen_tps   — токенов в секунду на генерации, без препроцессинга;
  out_tok   — среднее число токенов в ответе;
  n         — сколько ответов замерено.

Перед каждым ответом KV-кеш сбрасывается (llm.reset()), чтобы TTFT мерил
полный препроцессинг системного и пользовательского промпта, а не горячий
кэш от соседнего запроса. max_tokens общий для всех моделей, чтобы длина
генерации была сравнима.

Ответы ограничены грамматикой RU_GRAMMAR из src/va/grammars/ru.py (той же,
что в проде): на сэмплинге разрешены только кириллица, латиница, цифры и
базовая пунктуация, поэтому чужие письменности (CJK и т.п.) сгенерировать
нельзя. --no-grammar отключает ограничение.

Модели ищутся автоматически в models/llm/*.gguf (--models-dir), короткие
имена берутся из KNOWN или из имени файла.

Запуск (llama-cpp-python — зависимость проекта, больше ничего не нужно):
    uv run python scripts/benchmark_llm.py                     # все найденные
    uv run python scripts/benchmark_llm.py --models q25-1.5b,q35-2b
    uv run python scripts/benchmark_llm.py --reps 5 --max-tokens 48
    uv run python scripts/benchmark_llm.py --no-grammar        # без ограничения
    uv run python scripts/benchmark_llm.py --detail
    uv run python scripts/benchmark_llm.py --json output/llm_perf.json
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path

from va.grammars.ru import RU_GRAMMAR

# Продакшн-промпт Ады. Скопирован дословно из src/va/pipeline.py: импортировать
# его оттуда нельзя — va.pipeline тянет va.models и грузит whisper, piper, laya
# и decima, это лишние минуты и память для бенчмарка.
SYSTEM_PROMPT = (
    "Ты - голосовая ассистентка по имени Ада. Ты создана для выполнения "
    "простых задач, таких как ставить таймер и говорить погоду. Ты всегда "
    "говоришь о себе в женском роде: \"я создана\", \"я думала\", "
    "\"я чувствовала\". Не повторяй одни и те же по смыслу фразы."
)

MODELS_DIR = "models/llm"

# Короткое имя -> файл. Всё, что не перечислено, получает имя из файла.
KNOWN = {
    "qwen2.5-0.5b-instruct-q4_k_m.gguf": "q25-0.5b",
    "qwen2.5-1.5b-instruct-q4_k_m.gguf": "q25-1.5b",
    "qwen2.5-3b-instruct-q4_k_m.gguf": "q25-3b",
    "Qwen3.5-0.8B-Q4_K_M.gguf": "q35-0.8b",
    "Qwen3.5-2B-Q4_K_M.gguf": "q35-2b",
    "RuadaptQwen3-4B-Instruct-Q4_K_M.gguf": "ruadapt-q35-4b",
    "Vikhr-Qwen-2.5-1.5b-Instruct-Q4_K_M.gguf": "vikhr-q25-15b",
}

# Реплики без интента: в продакшене именно они уходят в LLM.
PROMPTS: tuple[str, ...] = (
    "Привет! Как тебя зовут?",
    "Расскажи коротко, что ты умеешь.",
    "Сколько будет семнадцать умножить на три?",
    "Придумай одну шутку про роботов.",
    "Что делать, если я поставил таймер и забыл про него?",
    "Посоветуй, какую книгу почитать вечером.",
)


# --------------------------------------------------------------------- модели
def _short_name(filename: str) -> str:
    if filename in KNOWN:
        return KNOWN[filename]
    stem = filename[:-5] if filename.lower().endswith(".gguf") else filename
    return "".join(ch if ch.isalnum() else "-" for ch in stem.lower()).strip("-")


def discover_models(models_dir: Path) -> dict[str, Path]:
    """Короткое имя -> путь для всех *.gguf в каталоге."""
    models: dict[str, Path] = {}
    for path in sorted(models_dir.iterdir()):
        if path.is_file() and path.suffix.lower() == ".gguf":
            models[_short_name(path.name)] = path
    return models


# --------------------------------------------------------------------- замеры
def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank перцентиль, как в остальных бенчмарках репозитория."""
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    index = max(0, min(len(ordered) - 1, math.ceil(pct / 100 * len(ordered)) - 1))
    return ordered[index]


def generate(llm, messages: list[dict], max_tokens: int, grammar=None) -> dict:
    """Один потоковый ответ: TTFT, полное время и число токенов.

    Токены считаем по непустым чанкам — llama.cpp отдаёт по чанку на токен,
    это ближе к реальной генерации, чем повторная токенизация готового текста.
    grammar (LlamaGrammar | None) маскирует чужие письменности на сэмплинге.
    """
    started = time.perf_counter()
    first_token: float | None = None
    tokens = 0
    parts: list[str] = []

    stream = llm.create_chat_completion(
        messages=messages,
        stream=True,
        temperature=0.0,
        max_tokens=max_tokens,
        grammar=grammar,
    )
    for chunk in stream:
        content = (chunk["choices"][0].get("delta") or {}).get("content")
        if not content:
            continue
        if first_token is None:
            first_token = time.perf_counter()
        tokens += 1
        parts.append(content)
    ended = time.perf_counter()

    return {
        "ttft_ms": (first_token - started) * 1000.0 if first_token is not None else float("nan"),
        "total_ms": (ended - started) * 1000.0,
        "tokens": tokens,
        "text": "".join(parts),
    }


def run_model(name: str, path: Path, args: argparse.Namespace) -> dict:
    from llama_cpp import Llama, LlamaGrammar

    kwargs: dict = {
        "model_path": str(path),
        "n_ctx": args.n_ctx,
        "n_gpu_layers": args.n_gpu_layers,
        "verbose": False,
    }
    if args.threads > 0:
        kwargs["n_threads"] = args.threads

    started = time.perf_counter()
    llm = Llama(**kwargs)
    load_s = time.perf_counter() - started
    grammar = None if args.no_grammar else LlamaGrammar.from_string(RU_GRAMMAR)

    try:
        # Прогрев: первый вызов по модели грузит граф и прогревает кэши.
        # Реплика обязательна — шаблон Qwen3.5 падает без user-сообщения.
        llm.reset()
        generate(
            llm,
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "привет"},
            ],
            max_tokens=1,
            grammar=grammar,
        )

        samples: list[dict] = []
        for rep in range(args.reps):
            for prompt in PROMPTS:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ]
                llm.reset()
                row = generate(llm, messages, args.max_tokens, grammar=grammar)
                row["prompt"] = prompt
                row["rep"] = rep
                samples.append(row)
    finally:
        llm.close()
        del llm
        gc.collect()

    ttfts = [s["ttft_ms"] for s in samples]
    totals = [s["total_ms"] for s in samples]
    gen_seconds = sum(
        (s["total_ms"] - s["ttft_ms"]) / 1000.0
        for s in samples
        if not math.isnan(s["ttft_ms"])
    )
    gen_tokens = sum(max(0, s["tokens"] - 1) for s in samples)
    metrics = {
        "size_mb": path.stat().st_size / 1e6,
        "load_s": load_s,
        "ttft50": _percentile(ttfts, 50),
        "ttft95": _percentile(ttfts, 95),
        "tot50": _percentile(totals, 50),
        "tot95": _percentile(totals, 95),
        "gen_tps": gen_tokens / gen_seconds if gen_seconds > 0 else float("nan"),
        "out_tok": sum(s["tokens"] for s in samples) / len(samples) if samples else float("nan"),
        "n": len(samples),
    }
    return {"name": name, "path": str(path), "metrics": metrics, "samples": samples}


# ---------------------------------------------------------------------- вывод
HEADER = (
    f"{'model':12} {'size_mb':>8} {'load_s':>7} {'ttft50':>7} {'ttft95':>7}"
    f" {'tot50':>7} {'tot95':>7} {'gen_tps':>8} {'out_tok':>7}"
)


def print_table(results: list[dict]) -> None:
    print(HEADER)
    for res in sorted(results, key=lambda r: -r["metrics"]["gen_tps"]):
        m = res["metrics"]
        print(
            f"{res['name']:12} {m['size_mb']:8.0f} {m['load_s']:7.2f}"
            f" {m['ttft50']:7.1f} {m['ttft95']:7.1f}"
            f" {m['tot50']:7.1f} {m['tot95']:7.1f}"
            f" {m['gen_tps']:8.1f} {m['out_tok']:7.1f}"
        )
    print()


def print_detail(results: list[dict]) -> None:
    for res in results:
        print(f"=== {res['name']} ({res['path']}) ===")
        print(f"    {'prompt':52} {'rep':>3} {'ttft':>7} {'total':>7} {'tok':>4}")
        for s in res["samples"]:
            prompt = s["prompt"][:50]
            print(
                f"    {prompt:52} {s['rep']:3d} {s['ttft_ms']:7.1f}"
                f" {s['total_ms']:7.1f} {s['tokens']:4d}"
            )
        print()


def dump_json(results: list[dict], path: Path, args: argparse.Namespace) -> None:
    payload = {
        "system_prompt": SYSTEM_PROMPT,
        "prompts": list(PROMPTS),
        "config": {
            "reps": args.reps,
            "max_tokens": args.max_tokens,
            "n_ctx": args.n_ctx,
            "n_gpu_layers": args.n_gpu_layers,
            "threads": args.threads,
            "grammar": not args.no_grammar,
        },
        "models": {
            res["name"]: {
                "path": res["path"],
                "metrics": res["metrics"],
                "samples": res["samples"],
            }
            for res in results
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"отчёт: {path}")


# ------------------------------------------------------------------------ cli
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Бенчмарк скорости установленных LLM (GGUF) на промптах Ады"
    )
    parser.add_argument("--models", default="",
                        help="какие модели брать (короткие имена через запятую; по умолчанию все)")
    parser.add_argument("--models-dir", default=MODELS_DIR, help="каталог с *.gguf")
    parser.add_argument("--reps", type=int, default=3, help="повторов на промпт (по умолчанию 3)")
    parser.add_argument("--max-tokens", type=int, default=64,
                        help="лимит генерации на ответ (по умолчанию 64)")
    parser.add_argument("--n-ctx", type=int, default=512, help="размер контекста (по умолчанию 512)")
    parser.add_argument("--n-gpu-layers", type=int, default=0,
                        help="слоёв на GPU, 0 — только CPU (по умолчанию 0)")
    parser.add_argument("--threads", type=int, default=0,
                        help="число потоков, 0 — выбор llama.cpp по умолчанию")
    parser.add_argument("--no-grammar", action="store_true",
                        help="не ограничивать символы ответа грамматикой (по умолчанию — RU_GRAMMAR)")
    parser.add_argument("--detail", action="store_true", help="построчно по каждому ответу")
    parser.add_argument("--json", default="", help="куда сложить полный отчёт")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    if not models_dir.is_dir():
        parser.error(f"каталог моделей не найден: {models_dir}")

    found = discover_models(models_dir)
    if not found:
        parser.error(f"в {models_dir} нет *.gguf")

    if args.models:
        wanted = [name.strip() for name in args.models.split(",") if name.strip()]
        unknown = [name for name in wanted if name not in found]
        if unknown:
            parser.error(f"неизвестные модели: {unknown}; найдены: {list(found)}")
        found = {name: found[name] for name in wanted}

    args.reps = max(1, args.reps)
    args.max_tokens = max(1, args.max_tokens)

    print(
        f"моделей {len(found)} | промптов {len(PROMPTS)} | повторов {args.reps}"
        f" | ответов {len(found) * len(PROMPTS) * args.reps}"
        f" | max_tokens {args.max_tokens} | n_ctx {args.n_ctx}"
        f" | gpu_layers {args.n_gpu_layers}"
        f" | threads {args.threads or 'auto'}"
        f" | grammar {'off' if args.no_grammar else 'RU'}",
        flush=True,
    )

    results: list[dict] = []
    for name, path in found.items():
        print(f"\n=== {name} ({path.name}) ===", flush=True)
        try:
            res = run_model(name, path, args)
        except Exception as exc:  # noqa: BLE001 — модель не поднялась, не роняем остальные
            print(f"  .. {name}: пропуск — {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        results.append(res)
        m = res["metrics"]
        print(
            f"  .. load {m['load_s']:.1f}s | ttft {m['ttft50']:.0f}/{m['ttft95']:.0f} ms"
            f" | total {m['tot50']:.0f}/{m['tot95']:.0f} ms"
            f" | gen {m['gen_tps']:.1f} tok/s | out {m['out_tok']:.0f} tok",
            flush=True,
        )

    if not results:
        print("ни одной модели не прогнано")
        return

    print()
    print_table(results)
    if args.detail:
        print_detail(results)
    if args.json:
        dump_json(results, Path(args.json), args)


if __name__ == "__main__":
    main()
