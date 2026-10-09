"""Сравнение лучшего конфига laya с Kev-0.8B, Decima-base и GLiNER2.5-Decide.decima.DecimaDECIMA.

Один золотой набор и те же метрики, что и в benchmark_intent.py: 147 фраз,
14 меток, score exact +2 / wrong -3 / refused -1 / fpresume -2. Каждая модель
получает те же 14 меток и тот же смысл критериев; laya и Decima — русскую
инструкцию, Kev и GLiNER — английский перевод тех же критериев (это английские
модели, так их и используют).

Раннеры:
  laya   — Router(lang_guess="ru") с промптом prod из benchmark_intent (тот же,
           что в src/va/intent.py): авто-роутинг, как в продакшене;
  decima — amyrmahdy/decima-base, int8 ONNX на CPU, Question с русской
           инструкцией и метками в порядке прод-критериев;
  kev    — jaredpalmer/kev-0.8b через kev.serve (POST /v1/systemone); состояние —
           фраза как есть, инструкция и описания критериев на английском;
  gliner — fastino/GLiNER2.5-Decide через gliner2 AutoExtractor; английские
           описания меток, сами метки русские, как у всех;
  gliner-multi — fastino/GLiNER2.5-multi-Decide (mDeBERTa-v3-base): тот же
           API, но многоязычная модель — описания меток берём русские, из
           промпта prod (как у laya). Добавлена, потому что базовый Decide
           английский, а русские фразы ведёт именно многовариант.

Гейт считается на готовых ответах, без повторного инференса: для каждого
классификатора свип порога по уверенности выбранной метки (у laya — по двум
осям: maxp = answer_confidence, entropy = поле confidence; у остальных — по
вероятности выбранной метки, у GLiNER — по некалиброванному sigmoid-скору).
Ответы ниже порога считаются отказом (другое + abstention) и пересчитывают
те же task-метрики. ECE — 10 бинов равной ширины по той же оси уверенности.

Запуск:
    uv run python scripts/benchmark_compare.py                           # все 4
    uv run python scripts/benchmark_compare.py --models laya,decima
    uv run python scripts/benchmark_compare.py --json output/compare.json
    uv run python scripts/benchmark_compare.py --report 'output/compare_*.json'

Kev подаётся как локальный сервер (отдельная venv в каталоге проекта):
    cd .bench-deps/kev && uv run --extra serve python -m kev.serve --run jaredpalmer/kev-0.8b --port 8008
Без сервера kev пропускается с предупреждением (см. KEV_BASE_URL).

Зависимости сверх проекта: decima, gliner2, peft, onnxruntime (uv pip install),
плюс свой venv для kev. Остальное — из venv проекта.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

import benchmark_intent as bi

KEV_BASE_URL = os.environ.get("KEV_BASE_URL", "http://127.0.0.1:8008")
KEV_MODEL_NAME = "kev-latest"
THRESHOLDS = (0.5, 0.7, 0.9)

# Английский перевод прод-критериев (PROD_PROMPT из benchmark_intent): те же
# 14 меток, та же семантика. Для Kev и GLiNER — в их рабочем языке.
EN_INSTRUCTIONS = (
    "A user's utterance to a voice assistant. Choose exactly one command from "
    "the list; if the command is not in the list or it is not a command, "
    "choose «другое»."
)
EN_CRITERIA: dict[str, str] = {
    "таймеры": "ask about active countdowns: which are running, how many, time left",
    "таймер": "start a countdown with a duration: «for five minutes», «in half an hour», «timer for two minutes»",
    "пауза": "stop the music or pause it: «stop», «pause», «enough»",
    "следующий трек": "switch to the next track or song, «next»",
    "предыдущий трек": "switch back to the previous track, «previous»",
    "музыка": "play a song, track, album or artist — by name or by request",
    "погода": "ask about the weather, temperature, rain, wind, what it is like outside",
    "время": "ask the current time: «what time is it», «what's the hour»",
    "что играет": "ask which song or track is playing right now",
    "намного громче": "raise the volume a lot: «much louder», «max volume»",
    "намного тише": "lower the volume a lot: «much quieter», «almost silent»",
    "громче": "raise the volume: «louder»",
    "тише": "lower the volume: «quieter»",
    "другое": "everything else: greetings, questions and requests not listed above",
}


# ---------------------------------------------------------------------- метрики
def task_counts(rows: list[dict]) -> dict:
    cnt = {"n": len(rows), "exact": 0, "wrong": 0, "refused": 0, "fpresume": 0}
    for r in rows:
        got, exp = r["choice"], r["expected"]
        if got == bi.OTHER:
            if exp == bi.OTHER:
                cnt["exact"] += 1
            else:
                cnt["refused"] += 1
        elif got == exp:
            cnt["exact"] += 1
        elif exp == bi.OTHER:
            cnt["fpresume"] += 1
        else:
            cnt["wrong"] += 1
    cnt["score"] = sum(cnt[k] * w for k, w in bi.SCORE.items())
    return cnt


def ece10(confs, corrects, bins: int = 10) -> float:
    """Expected calibration error, 10 бинов равной ширины по оси уверенности."""
    confs = np.asarray(confs, dtype=float)
    corrects = np.asarray(corrects, dtype=float)
    if not corrects.size:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.searchsorted(edges[1:-1], confs, "right"), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = idx == b
        if not mask.any():
            continue
        acc = corrects[mask].mean()
        conf = confs[mask].mean()
        total += abs(acc - conf) * (mask.sum() / confs.size)
    return total


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = max(0, min(len(xs) - 1, math.ceil(p / 100 * len(xs)) - 1))
    return xs[k]


def metrics(rows: list[dict]) -> dict:
    m = task_counts(rows)
    m["acc"] = m["exact"] / m["n"] if m["n"] else float("nan")
    m["ece"] = ece10([r["conf"] for r in rows], [r["choice"] == r["expected"] for r in rows])
    lat = [r["latency_ms"] for r in rows]
    m["p50"] = pct(lat, 50)
    m["p95"] = pct(lat, 95)
    return m


def gate_sweep(rows: list[dict], axes: list[tuple[str, str]]) -> list[dict]:
    """Свип порога уверенности: отказ ниже порога = выбор «другое».

    Ось — имя поля строки (conf/p), имя в выводе — первое в паре.
    """
    out = []
    for label, key in axes:
        for thr in THRESHOLDS:
            gated = [
                dict(r, choice=bi.OTHER) if r[key] < thr else dict(r)
                for r in rows
            ]
            m = metrics(gated)
            m["axis"] = label
            m["threshold"] = thr
            m["cov"] = sum(1 for r in rows if r[key] >= thr) / len(rows) if rows else 0.0
            out.append(m)
    return out


def best_gate(gates: list[dict]) -> dict | None:
    if not gates:
        return None
    return max(gates, key=lambda g: (g["score"], -g["threshold"]))


# ---------------------------------------------------------------------- раннеры
class LayaRunner:
    key = "laya"
    label = "laya prod (авто-роутинг)"
    axes = [("entropy", "conf"), ("maxp", "p")]

    def __init__(self) -> None:
        from laya import Router

        self.router = Router(lang_guess="ru")
        self.questions = bi.PROD_PROMPT.questions()

    def predict(self, phrase: str) -> dict:
        started = time.perf_counter()
        answer = self.router.predict(phrase, self.questions)["answers"]["intent"]
        return {
            "choice": answer["choice"],
            "p": answer["answer_confidence"],
            "conf": answer["confidence"],
            "latency_ms": (time.perf_counter() - started) * 1000.0,
        }


class DecimaRunner:
    key = "decima"
    label = "Decima-base int8 (321M)"
    axes = [("maxp", "conf")]
    model_id = "amyrmahdy/decima-base"

    def __init__(self) -> None:
        from decima import Decima, Question

        self.model = Decima.from_pretrained(self.model_id, threads=1)
        self.question = Question(bi.PROD_PROMPT.instructions, list(bi.LABELS), lang="ru")


class DecimaSmallRunner(DecimaRunner):
    """amyrmahdy/decima-small: младшая версия decima на multilingual-e5-small (122M)."""

    key = "decima-small"
    label = "Decima-small int8 (122M)"
    model_id = "amyrmahdy/decima-small"

    def predict(self, phrase: str) -> dict:
        started = time.perf_counter()
        decision = self.model.decide(phrase, self.question)
        p = max(decision.probs)
        return {
            "choice": decision.top,
            "p": p,
            "conf": p,
            "latency_ms": (time.perf_counter() - started) * 1000.0,
        }


class KevRunner:
    key = "kev"
    label = "Kev-0.8B (kev.serve)"
    axes = [("maxp", "conf")]

    def __init__(self, base_url: str = KEV_BASE_URL, timeout: float = 300.0) -> None:
        import httpx

        self.client = httpx.Client(base_url=base_url, timeout=timeout)
        self.questions = {
            "intent": {
                "type": "choice",
                "instructions": EN_INSTRUCTIONS,
                "criteria": dict(EN_CRITERIA),
            }
        }
        # Проверка, что сервер отвечает, до того как прогонять набор.
        self.client.get("/v1/models").raise_for_status()

    def predict(self, phrase: str) -> dict:
        started = time.perf_counter()
        response = self.client.post(
            "/v1/systemone",
            json={"state": phrase, "model": KEV_MODEL_NAME, "questions": self.questions},
        )
        response.raise_for_status()
        answer = response.json()["answers"]["intent"]
        probs = answer.get("probabilities") or {}
        p = max(probs.values()) if probs else float(answer["confidence"])
        return {
            "choice": answer["choice"],
            "p": p,
            "conf": p,
            "latency_ms": (time.perf_counter() - started) * 1000.0,
        }


class EdgeKevRunner(KevRunner):
    """jaredpalmer/kev-0.5b (Qwen2.5-0.5B + LoRA) через edgejev ONNX.

    Тот же /v1/systemone, что у kev.serve, но сервер edgejev не отдаёт
    /v1/models — убираем стартовый пинг; поле "model" в запросе он игнорирует.
    """

    key = "kev-0.5b"
    label = "Kev-0.5B Qwen2.5 (edgejev ONNX)"

    def __init__(self, base_url: str = "http://127.0.0.1:8009", timeout: float = 120.0) -> None:
        import httpx

        self.client = httpx.Client(base_url=base_url, timeout=timeout)
        self.questions = {
            "intent": {
                "type": "choice",
                "instructions": EN_INSTRUCTIONS,
                "criteria": dict(EN_CRITERIA),
            }
        }


class VonRunner:
    """Von 1.3 (wfzyx/von, 395M ModernBERT-large): неавторегрессионный
    System One решатель. Один проход энкодера на все опции — [MASK]-маркеры
    в одной последовательности. Англоязычный, поэтому критерии — английский
    перевод прод-промпта (как у Kev/GLiNER); фразы остаются русскими.

    confidence — входно-условная откалиброванная уверенность (маркерная
    температура), не меняет аргмакс, только «насколько уверен ответ».
    """

    key = "von"
    label = "Von 1.3 (395M ModernBERT, OpenVINO)"
    axes = [("conf", "conf"), ("maxp", "p")]

    def __init__(self) -> None:
        import von

        # Модульный клиент лениво создаёт движок при первом вызове decide и
        # держит его в _default_client; на этой машине бэкенд выберется как
        # OpenVINO CPU (VON_DEVICE=auto, GPU нет).
        self._von = von
        # choices — {метка: англ. описание}; ключи — русские метки, чтобы
        # answer.choice совпадал с bi.LABELS.
        self.choices = dict(EN_CRITERIA)
        self.instructions = EN_INSTRUCTIONS

    def predict(self, phrase: str) -> dict:
        started = time.perf_counter()
        answer = self._von.decide(
            state=phrase,
            choices=self.choices,
            instructions=self.instructions,
        )
        p = max(answer.probabilities.values())
        return {
            "choice": answer.choice,
            "p": p,
            "conf": answer.confidence,
            "latency_ms": (time.perf_counter() - started) * 1000.0,
        }


class GlinerRunner:
    key = "gliner"
    label = "GLiNER2.5-Decide (340M)"
    axes = [("score", "conf")]

    def __init__(self) -> None:
        from gliner2 import AutoExtractor

        self.model = AutoExtractor.from_pretrained("fastino/GLiNER2.5-Decide")
        self.tasks = self._tasks(EN_CRITERIA)

    @staticmethod
    def _tasks(descriptions: dict[str, str]) -> dict:
        return {
            "intent": {
                "labels": {label: descriptions.get(label, "") for label in bi.LABELS}
            }
        }

    def predict(self, phrase: str) -> dict:
        started = time.perf_counter()
        result = self.model.classify_text(
            phrase, self.tasks, threshold=0.0, include_confidence=True
        )
        item = result.get("intent") or {}
        if not item.get("label"):
            # Метка не набрала порог — считаем отказом, как «другое».
            return {"choice": bi.OTHER, "p": None, "conf": 0.0,
                    "latency_ms": (time.perf_counter() - started) * 1000.0}
        score = float(item["confidence"])
        return {
            "choice": item["label"],
            "p": score,
            "conf": score,
            "latency_ms": (time.perf_counter() - started) * 1000.0,
        }


class GlinerMultiRunner(GlinerRunner):
    """fastino/GLiNER2.5-multi-Decide: многоязычный вариант на mDeBERTa-v3-base.

    Описания меток — русские, из промпта prod (та же семантика, что у laya),
    раз модель многоязычная; сам API не отличается от английской версии.
    """

    key = "gliner-multi"
    label = "GLiNER2.5-multi-Decide (287M)"

    def __init__(self) -> None:
        from gliner2 import AutoExtractor

        self.model = AutoExtractor.from_pretrained("fastino/GLiNER2.5-multi-Decide")
        self.tasks = self._tasks(bi.PROD_PROMPT.criteria)


RUNNERS: dict[str, type] = {
    "laya": LayaRunner,
    "decima": DecimaRunner,
    "decima-small": DecimaSmallRunner,
    "kev": KevRunner,
    "kev-0.5b": EdgeKevRunner,
    "gliner": GlinerRunner,
    "gliner-multi": GlinerMultiRunner,
    "von": VonRunner,
}


# ------------------------------------------------------------------------ прогон
def run_model(key: str, cases: list[tuple[str, str]], secs_between: float = 9.0) -> dict:
    if key == "kev":
        try:
            runner = KevRunner()
        except Exception as exc:  # сервер не поднят или не отвечает
            print(f"  .. kev: пропуск — сервер недоступен ({exc.__class__.__name__}: {exc})",
                  file=sys.stderr)
            return {}
    else:
        runner = RUNNERS[key]()

    runner.predict("прогрев роутера")
    rows: list[dict] = []
    for index, (phrase, expected) in enumerate(cases):
        row = runner.predict(phrase)
        row["phrase"] = phrase
        row["expected"] = expected
        rows.append(row)
        if (index + 1) % 25 == 0:
            print(
                f"  .. {key}: {index + 1}/{len(cases)} фраз"
                f" (последняя {row['latency_ms']:.0f} мс)",
                file=sys.stderr,
            )
    m = metrics(rows)
    m["secs"] = sum(r["latency_ms"] for r in rows) / 1000.0
    return {
        "label": runner.label,
        "axes": runner.axes,
        "metrics": m,
        "gates": gate_sweep(rows, runner.axes),
        "rows": rows,
    }


# ------------------------------------------------------------------------- вывод
HEADER = (
    f"{'модель':26} {'score':>7} {'ex':>4}{'wr':>4}{'rf':>4}{'fp':>4}"
    f" {'acc':>5} {'ece':>5} {'p50ms':>7} {'p95ms':>7}"
)


def gate_header() -> str:
    return (
        f"{'модель':26} {'порог':>9} {'score':>7} {'ex':>4}{'wr':>4}{'rf':>4}{'fp':>4}"
        f" {'cov':>5} {'acc':>5}"
    )


def print_compare(results: dict[str, dict]) -> None:
    rows = sorted(results.items(), key=lambda kv: (-kv[1]["metrics"]["score"], kv[0]))
    print(HEADER)
    for key, res in rows:
        m = res["metrics"]
        best = best_gate(res["gates"])
        print(
            f"{key:26} {m['score']:7.1f} {m['exact']:4d}{m['wrong']:4d}"
            f"{m['refused']:4d}{m['fpresume']:4d} {m['acc']:5.2f} {m['ece']:5.2f}"
            f" {m['p50']:7.1f} {m['p95']:7.1f}"
        )
        if best:
            g = best
            print(
                f"{'  лучший гейт':26} {'@' + g['axis'] + str(g['threshold']):>9}"
                f" {g['score']:7.1f} {g['exact']:4d}{g['wrong']:4d}"
                f"{g['refused']:4d}{g['fpresume']:4d} {g['cov']:5.2f} {g['acc']:5.2f}"
            )
    print()

    print("свип порога уверенности (отказ ниже порога):")
    print(gate_header())
    for key in sorted(results, key=lambda k: results[k]["metrics"]["score"], reverse=True):
        for g in sorted(results[key]["gates"], key=lambda g: g["score"], reverse=True):
            print(
                f"{key:26} {g['axis'] + '@' + str(g['threshold']):>9} {g['score']:7.1f}"
                f" {g['exact']:4d}{g['wrong']:4d}{g['refused']:4d}{g['fpresume']:4d}"
                f" {g['cov']:5.2f} {g['acc']:5.2f}"
            )
    print()


def print_errors(results: dict[str, dict], key: str, max_rows: int = 12) -> None:
    res = results[key]
    cases = [
        (row["phrase"], row["expected"], row["choice"], row["conf"])
        for row in res["rows"]
        if row["choice"] != row["expected"]
    ]
    print(f"  {key}: {len(cases)} ошибок (без гейта):")
    for phrase, exp, got, conf in cases[:max_rows]:
        print(f"    xx {exp:16} <- {got:16} {phrase!r} (p {conf:.2f})")
    if len(cases) > max_rows:
        print(f"    ... ещё {len(cases) - max_rows}")


def dump_json(results: dict[str, dict], path: Path) -> None:
    payload = {
        "cases_n": len(bi.CASES),
        "scores": bi.SCORE,
        "models": {
            key: {
                "label": res["label"],
                "axes": res["axes"],
                "metrics": res["metrics"],
                "gates": res["gates"],
                "rows": res["rows"],
            }
            for key, res in results.items()
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"отчёты: {path}")


def load_json(paths: list[str]) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for raw in paths:
        data = json.loads(Path(raw).read_text(encoding="utf-8"))
        for key, res in data.get("models", {}).items():
            results[key] = res
    return results


# -------------------------------------------------------------------------- cli
def main() -> None:
    parser = argparse.ArgumentParser(description="Сравнение лучшего laya с Kev/Decima/GLiNER")
    parser.add_argument("--models", default="",
                        help="что прогонять: laya,decima,kev,gliner (по умолчанию все)")
    parser.add_argument("--json", default="", help="куда сложить полные отчёты")
    parser.add_argument("--report", default="",
                        help="не прогонять, а собрать таблицу из готовых --json "
                             "(имена через запятую или glob)")
    parser.add_argument("--errors", default="",
                        help="построчный разбор ошибок указанных моделей (через запятую)")
    args = parser.parse_args()

    if args.report:
        paths: list[str] = []
        for part in args.report.split(","):
            part = part.strip()
            if not part:
                continue
            if any(ch in part for ch in "*?["):
                paths.extend(str(p) for p in sorted(Path(".").glob(part)))
            else:
                paths.append(part)
        if not paths:
            parser.error("--report не нашёл файлов")
        results = load_json(paths)
        if not results:
            parser.error("в отчётах нет ни одной модели")
        print_compare(results)
        if args.errors:
            for key in args.errors.split(","):
                key = key.strip()
                if key in results:
                    print_errors(results, key)
        return

    keys = [k.strip() for k in args.models.split(",") if k.strip()] or list(RUNNERS)
    unknown = [k for k in keys if k not in RUNNERS]
    if unknown:
        parser.error(f"неизвестные модели: {unknown}; доступны {list(RUNNERS)}")

    cases = list(bi.CASES)
    print(
        f"набор {len(cases)} фраз / {len(bi.LABELS)} меток | модели: {', '.join(keys)}"
        f" | пороги гейта {THRESHOLDS}",
        flush=True,
    )

    results: dict[str, dict] = {}
    for i, key in enumerate(keys):
        print(f"\n=== {key} ===", flush=True)
        res = run_model(key, cases)
        if res:
            results[key] = res

    if not results:
        print("ни одного прогона: проверь сети/серверы")
        return

    print()
    print_compare(results)

    if args.errors:
        for key in args.errors.split(","):
            key = key.strip()
            if key in results:
                print_errors(results, key)

    if args.json:
        dump_json(results, Path(args.json))


if __name__ == "__main__":
    main()
