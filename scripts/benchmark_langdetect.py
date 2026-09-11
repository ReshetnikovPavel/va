import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from pywhispercpp.model import Model as Whisper

from benchmark_stt import char_error_rate, load_clips, wer

MODELS = ["tiny", "base", "small"]
LANGS = {"ru", "en"}


def clip_language(reference: str) -> str:
    return "ru" if any("\u0400" <= ch <= "\u04ff" for ch in reference) else "en"


def detect(model: Whisper, audio: np.ndarray) -> tuple[str, dict[str, float], float]:
    start = time.perf_counter()
    (lang, conf), all_langs = model.auto_detect_language(audio)
    duration = time.perf_counter() - start
    return lang, dict(all_langs), duration


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark whisper.cpp language detection + two-stage RU/EN transcription."
    )
    parser.add_argument("data_dir", type=Path)
    parser.add_argument(
        "--models",
        default=",".join(MODELS),
        help="Models to test detection with (default: tiny,base,small).",
    )
    parser.add_argument(
        "--stt",
        default="base",
        help="STT model used in the transcribe stage (default: base).",
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    clips = load_clips(args.data_dir)
    print(f"Loaded {len(clips)} clips:")
    for i, c in enumerate(clips):
        lang = clip_language(c["reference"])
        print(f"  {i + 1}. [{lang}] {c['reference'][:60]}")
    langs = {clip_language(c["reference"]) for c in clips}
    if not (langs & LANGS):
        raise SystemExit("No ru/en clips found; reference must contain the language.")

    detect_models = [m.strip() for m in args.models.split(",") if m.strip()]
    all_results = []

    print("\n=== Language detection (per-model) ===")
    wers_detected = {}
    for name in detect_models:
        model_dir = Path.home() / ".local" / "share" / "pywhispercpp" / "models"
        path = model_dir / f"ggml-{name}.bin"
        model = Whisper(str(path) if path.exists() else name, models_dir=str(model_dir))
        correct = 0
        total_time = 0.0
        for c in clips:
            expected = clip_language(c["reference"])
            lang, langs, duration = detect(model, c["audio"])
            total_time += duration
            ok = lang in LANGS and lang == expected
            correct += ok
            print(
                f"  {name:<6} expected={expected} detected={lang:<3} "
                f"conf={langs.get(lang, 0):.3f} ok={ok} ({duration:.2f}s)"
            )
            all_results.append(
                {
                    "stage": "detect",
                    "model": name,
                    "expected": expected,
                    "detected": lang,
                    "conf": round(langs.get(lang, 0), 3),
                    "ok": ok,
                    "time_s": round(duration, 2),
                    "top": {k: round(v, 3) for k, v in sorted(langs.items(), key=lambda x: -x[1])[:3]},
                }
            )
        acc = correct / len(clips)
        wers_detected[name] = acc
        print(f"  -> {name}: accuracy={acc:.0%} avg_time={total_time / len(clips):.2f}s")

    print("\n=== Transcription: auto-detect vs two-stage (detect then force lang) ===")
    stt_dir = Path.home() / ".local" / "share" / "pywhispercpp" / "models"
    stt_path = stt_dir / f"ggml-{args.stt}.bin"
    stt = Whisper(str(stt_path) if stt_path.exists() else args.stt, models_dir=str(stt_dir))

    summary = []

    def run_case(label: str, language: str) -> tuple[float, float, float]:
        errors_w = errors_c = tot = 0.0
        for c in clips:
            seconds = len(c["audio"]) / 16000.0
            start = time.perf_counter()
            segs = stt.transcribe(c["audio"], single_segment=True, translate=False, language=language)
            taken = time.perf_counter() - start
            hyp = "".join(s.text for s in segs) if segs else ""
            errors_w += wer(c["reference"], hyp)
            errors_c += char_error_rate(c["reference"], hyp)
            tot += taken / seconds
        n = len(clips)
        return errors_w / n, errors_c / n, tot / n

    wer_a, cer_a, rtf_a = run_case("auto", "")
    summary.append(("auto-detect (baseline)", wer_a, cer_a, rtf_a))
    print(f"  auto-detect (baseline): WER={wer_a:.3f} CER={cer_a:.3f} RTF={rtf_a:.2f}")
    all_results.append({"stage": "transcribe", "pipeline": "auto", "wer": round(wer_a, 3), "cer": round(cer_a, 3), "rtf": round(rtf_a, 2)})

    for name in detect_models:
        model_dir = Path.home() / ".local" / "share" / "pywhispercpp" / "models"
        path = model_dir / f"ggml-{name}.bin"
        det = Whisper(str(path) if path.exists() else name, models_dir=str(model_dir))
        errors_w = errors_c = 0.0
        rtf_sum = 0.0
        per_clip = []
        for c in clips:
            expected = clip_language(c["reference"])
            lang, _, _ = detect(det, c["audio"])
            forced = lang if lang in LANGS else "ru"
            seconds = len(c["audio"]) / 16000.0
            start = time.perf_counter()
            segs = stt.transcribe(c["audio"], single_segment=True, translate=False, language=forced)
            taken = time.perf_counter() - start
            hyp = "".join(s.text for s in segs) if segs else ""
            errors_w += wer(c["reference"], hyp)
            errors_c += char_error_rate(c["reference"], hyp)
            rtf_sum += taken / seconds
            per_clip.append({"expected": expected, "detected": lang, "forced": forced, "hyp": hyp})
        n = len(clips)
        w, c_cer = errors_w / n, errors_c / n
        rtf = rtf_sum / n
        summary.append((f"{name}-detect -> {args.stt}", w, c_cer, rtf))
        print(f"  {name}-detect -> {args.stt}: WER={w:.3f} CER={c_cer:.3f} RTF={rtf:.2f}")
        all_results.append(
            {"stage": "transcribe", "pipeline": f"{name}-detect", "wer": round(w, 3), "cer": round(c_cer, 3), "rtf": round(rtf, 2), "clips": per_clip}
        )

    print("\n=== Summary ===")
    for label, w, c_cer, rtf in sorted(summary, key=lambda x: (x[2], x[1])):
        print(f"  {label:<28} WER={w:.3f} CER={c_cer:.3f} RTF={rtf:.2f}")

    if args.json:
        args.json.write_text(json.dumps(all_results, ensure_ascii=False, indent=2))
        print(f"\nWrote results to {args.json}")


if __name__ == "__main__":
    main()