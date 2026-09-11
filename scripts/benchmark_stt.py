import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from pywhispercpp.model import Model as Whisper


MODELS = ["tiny", "base", "base-ru", "small"]


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zа-яё0-9\s]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def wer(reference: str, hypothesis: str) -> float:
    ref = normalize(reference).split()
    hyp = normalize(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    dp = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        dp[i][0] = i
    for j in range(len(hyp) + 1):
        dp[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[len(ref)][len(hyp)] / len(ref)


def char_error_rate(reference: str, hypothesis: str) -> float:
    ref = normalize(reference)
    hyp = normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    dp = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        dp[i][0] = i
    for j in range(len(hyp) + 1):
        dp[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[len(ref)][len(hyp)] / len(ref)


def load_clips(data_dir: Path) -> list[dict]:
    clips = []
    for wav in sorted(data_dir.glob("*.wav")):
        txt = wav.with_suffix(".txt")
        if not txt.exists():
            raise FileNotFoundError(f"Missing reference transcript: {txt}")
        audio, sr = sf.read(wav, dtype="float32", always_2d=True)
        if sr != 16000:
            audio = np.mean(audio, axis=1) if audio.shape[1] > 1 else audio[:, 0]
            raise ValueError(
                f"{wav.name}: expected 16kHz mono, got sr={sr}. "
                "Please resample to 16kHz mono before benchmarking."
            )
        audio = np.mean(audio, axis=1) if audio.shape[1] > 1 else audio[:, 0]
        clips.append(
            {"audio": audio.astype(np.float32), "reference": txt.read_text().strip()}
        )
    return clips


def bench_model(model_name: str, clips: list[dict], task: str, language: str) -> dict:
    model_dir = Path.home() / ".local" / "share" / "pywhispercpp" / "models"
    candidate = model_dir / f"ggml-{model_name}.bin"
    model_arg = str(candidate) if candidate.exists() else model_name
    try:
        model = Whisper(
            model_arg,
            models_dir=str(model_dir),
            n_threads=4,
            single_segment=True,
            translate=(task == "translate"),
            language=language,
        )
        return _bench_model(model, model_name, clips, task, language)
    except Exception as exc:  # noqa: BLE001 - report and continue with other models
        return {
            "model": model_name,
            "task": task,
            "language": language or "auto",
            "error": str(exc),
        }


def _bench_model(model: Whisper, model_name: str, clips: list[dict], task: str, language: str) -> dict:
    word_errors = 0
    char_errors = 0
    total_chars = 0
    timed = 0.0
    results = []
    for clip in clips:
        seconds = len(clip["audio"]) / 16000.0
        start = time.perf_counter()
        segments = model.transcribe(clip["audio"], single_segment=True, translate=(task == "translate"), language=language)
        duration = time.perf_counter() - start
        hyp = "".join(s.text for s in segments) if segments else ""
        timed += duration
        rtf = duration / seconds
        wer_v = wer(clip["reference"], hyp)
        cer_v = char_error_rate(clip["reference"], hyp)
        word_errors += wer_v
        char_errors += cer_v
        total_chars += len(normalize(clip["reference"]))
        results.append(
            {"clip": str(Path(clip["reference"])), "hypothesis": hyp, "wer": round(wer_v, 3), "cer": round(cer_v, 3), "rtf": round(rtf, 2)}
        )
    n = len(clips)
    return {
        "model": model_name,
        "task": task,
        "language": language or "auto",
        "avg_wer": round(word_errors / n, 3),
        "avg_cer": round(char_errors / n, 3),
        "avg_rtf": round(timed / n, 2),
        "total_seconds_audio": round(sum(len(c["audio"]) / 16000.0 for c in clips), 1),
        "transcribe_seconds": round(timed, 1),
        "clips": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark whisper.cpp models for Russian + English speech recognition on CPU."
    )
    parser.add_argument(
        "data_dir",
        type=Path,
        help="Folder with .wav (16kHz mono) clips and matching .txt reference transcripts.",
    )
    parser.add_argument(
        "--models",
        default=",".join(MODELS),
        help="Comma-separated models to test (default: tiny,base,small).",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Write full per-clip results to a JSON file.",
    )
    args = parser.parse_args()

    clips = load_clips(args.data_dir)
    print(f"Loaded {len(clips)} clips:")
    for i, c in enumerate(clips):
        print(f"  {i + 1}. {c['reference'][:60]}")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    all_results = []
    for model in models:
        for task, language in (("transcribe", ""), ("transcribe", "ru"), ("translate", "")):
            print(f"\n=== {model} | task={task} | language={language or 'auto'} ===")
            r = bench_model(model, clips, task, language)
            if "error" in r:
                print(f"  FAILED: {r['error']}")
            else:
                print(
                    f"  avg WER={r['avg_wer']:.3f}  avg CER={r['avg_cer']:.3f}  RTF={r['avg_rtf']:.2f}"
                )
            all_results.append(r)

    print("\n=== Summary (best first) ===")
    valid = [r for r in all_results if "error" not in r]
    ranked = sorted(valid, key=lambda r: (r["avg_cer"], r["avg_wer"]))
    for i, r in enumerate(ranked, 1):
        print(
            f"  {i}. {r['model']:<6} task={r['task']:<10} lang={r['language']:<4} "
            f"WER={r['avg_wer']:.3f} CER={r['avg_cer']:.3f} RTF={r['avg_rtf']:.2f}"
        )

    if valid:
        print("\nRecommendation: {}".format(ranked[0]["model"]))
    else:
        print("\nNo model completed successfully.")

    if args.json:
        args.json.write_text(json.dumps(all_results, ensure_ascii=False, indent=2))
        print(f"Wrote results to {args.json}")


if __name__ == "__main__":
    main()
