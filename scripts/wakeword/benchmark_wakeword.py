import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from livekit.wakeword import WakeWordModel

SAMPLE_RATE = 16000
WINDOW_SECS = 2.0
STEP_SECS = 0.25
DEFAULTS = [0.05, 0.1, 0.2, 0.3, 0.5]


def load_models(models_dir: Path) -> list[Path]:
    seen = set()
    paths = []
    for p in sorted(models_dir.rglob("*.onnx")):
        digest = hashlib.md5(p.read_bytes()).hexdigest()
        if digest in seen:
            print(f"  (skipping duplicate by content: {p})")
            continue
        seen.add(digest)
        paths.append(p)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark wake word models via livekit-wakeword on 16kHz clips."
    )
    parser.add_argument(
        "data_dir",
        type=Path,
        help=(
            "Folder with .wav clips. Name clips <wakeword>.wav (e.g. alexa.wav). "
            "Multiple clips per wakeword are allowed with a suffix: "
            "<wakeword>.<variant>.wav (e.g. alexa.acc.wav)."
        ),
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Folder with ONNX wake word models (default: models/).",
    )
    parser.add_argument(
        "--window-ms", type=int, default=int(WINDOW_SECS * 1000), help="Detection window in ms."
    )
    parser.add_argument(
        "--step-ms", type=int, default=int(STEP_SECS * 1000), help="Window step in ms."
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    clips: list[tuple[str, str, np.ndarray]] = []
    for wav in sorted(args.data_dir.glob("*.wav")):
        audio, sr = sf.read(wav, dtype="float32")
        audio = audio.flatten()
        if sr != SAMPLE_RATE:
            raise SystemExit(f"{wav.name}: expected {SAMPLE_RATE} Hz, got {sr}")
        stem = wav.stem
        wakeword = stem.split(".")[0]
        clips.append((wakeword, stem, audio))
    if not clips:
        raise SystemExit(f"No .wav clips found in {args.data_dir}")

    window = args.window_ms * SAMPLE_RATE // 1000
    step = args.step_ms * SAMPLE_RATE // 1000
    print(f"Loaded {len(clips)} clips, window={args.window_ms}ms step={args.step_ms}ms")

    models = load_models(args.models_dir)
    print(f"Loaded {len(models)} models:")
    for m in models:
        print(f"  {m}")
    names = [m.stem for m in models]

    all_results = []
    matrix = {}
    for model_path in models:
        name = model_path.stem
        wwm = WakeWordModel(models=[model_path])
        row = []
        for wakeword, stem, audio in clips:
            if len(audio) < window:
                print(f"  WARNING: {stem} shorter than window ({len(audio)/SAMPLE_RATE:.1f}s); use >= {WINDOW_SECS}s clips")
            start = time.perf_counter()
            best = 0.0
            hit_at = None
            for offset in range(0, max(len(audio) - window, 1), step):
                win = audio[offset : offset + window]
                if len(win) < window:
                    win = np.pad(win, (0, window - len(win)))
                scores = wwm.predict(win)
                score = scores[name]
                if score > best:
                    best = score
                    hit_at = (offset + window) / SAMPLE_RATE
            elapsed = time.perf_counter() - start
            row.append(
                {
                    "clip": stem,
                    "wakeword": wakeword,
                    "score": round(best, 3),
                    "hit_at_s": round(hit_at, 2) if hit_at else None,
                    "time_s": round(elapsed, 3),
                }
            )
        matrix[name] = row

    print("\n=== Peak scores (rows=clips, cols=models) ===")
    header = "clip".ljust(14) + "".join(n.ljust(14) for n in names)
    print(header)
    for _, stem, _ in clips:
        cells = []
        for name in names:
            entry = next(e for e in matrix[name] if e["clip"] == stem)
            cells.append(f"{entry['score']:.3f}".ljust(14))
        print(stem.ljust(14) + "".join(cells))

    print("\n=== Accuracy vs threshold ===")
    all_thresholds = DEFAULTS + [0.0005]
    for threshold in all_thresholds:
        correct = total = 0
        for name, row in matrix.items():
            for entry in row:
                triggered = entry["score"] >= threshold
                is_own = entry["wakeword"] == name
                correct += int(triggered == is_own)
                total += 1
        acc = correct / total
        mark = " (current app threshold)" if threshold == 0.0005 else ""
        print(f"  threshold={threshold:.4f}: accuracy={acc:.0%}{mark}")

    if args.json:
        args.json.write_text(json.dumps({"matrix": matrix}, indent=2))
        print(f"\nWrote results to {args.json}")


if __name__ == "__main__":
    main()