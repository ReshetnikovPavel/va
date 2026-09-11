"""Benchmark STT engines/models for short-utterance (command-like) transcription.

Pixel for the voice assistant: a user says a short phrase (~1-6 s), we transcribe
the whole captured buffer, and get the full message.  No streaming callbacks, no
tightening of VAD.  Instead we test the effect of trimming trailing silence.

Benchmark matrix (all n_threads=8, RU, 20 clips):
    whisper.cpp        : base, base-ru, small, large-v3-turbo-q5_0   (batch)
    faster-whisper     : large-v3-turbo (int8)
Every config runs on (a) raw utterances and (b) trailing-silence-trimmed audio.

Dataset: built from the scraped RU youtube audio + auto-captions in
data/wakeword/scraped/ (word-accurate timestamps -> clean short slices with
reference transcripts).  Cached under data/wakeword/benchmark/.

Reuses 16 kHz clip loading + WER/CER helpers from benchmark_stt.py (read-only).
Usage:
    python scripts/benchmark_stt_v2.py [--configs base,base-ru,small,medium,turbo,fw-turbo,fw-medium,fw-distil,fw-distil35]
                                       [--clips PATH] [--no-trim] [--rebuild]
                                       [--max-clips 20] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from benchmark_stt import char_error_rate, load_clips, normalize, wer

MODELS_DIR = Path.home() / ".local" / "share" / "pywhispercpp" / "models"

# config id -> (engine, model)
CONFIGS = {
    "base": ("whisper.cpp", "base"),
    "base-ru": ("whisper.cpp", "base-ru"),
    "small": ("whisper.cpp", "small"),
    "turbo": ("whisper.cpp", "large-v3-turbo-q5_0"),
    "medium": ("whisper.cpp", "medium-q5_0"),
    "fw-turbo": ("faster-whisper", "large-v3-turbo"),
    "fw-medium": ("faster-whisper", "medium"),
    "fw-distil": ("faster-whisper", "distil-large-v3"),
    "fw-distil35": ("faster-whisper", "distil-large-v3.5"),
}

N_THREADS = 8
SR = 16000

_TS = re.compile(r"(\d{2}):(\d{2}):(\d{2}\.\d{3})")
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def _stamp_to_sec(s: str) -> float:
    m = _TS.match(s.strip())
    if not m:
        raise ValueError(f"bad timestamp: {s!r}")
    h, mi, sec = int(m.group(1)), int(m.group(2)), float(m.group(3))
    return h * 3600 + mi * 60 + sec


def _clean_text(raw: str) -> str:
    txt = _WS.sub(" ", _TAG.sub(" ", raw)).strip()
    return txt


def _parse_vtt(path: Path) -> list[dict]:
    """Return list of {start, end, text} plain-sentence cues.

    Handles both plain VTTs and YouTube word-aligned VTTs.  For word-aligned
    files the rolling plain-text cues give exact (already-spoken) windows: the
    reference text of a plain cue describes the audio between the previous
    block start and that cue's own start.
    """
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    cues: list[dict] = []
    blocks: list[dict] = []

    i = 0
    while i < len(lines):
        line = lines[i]
        if "-->" not in line:
            i += 1
            continue
        parts = line.split("-->")
        start = _stamp_to_sec(parts[0])
        end = _stamp_to_sec(parts[1]) or start
        j = i + 1
        text_lines: list[str] = []
        while j < len(lines) and lines[j].strip() != "":
            text_lines.append(lines[j])
            j += 1
        raw = "\n".join(text_lines)
        blocks.append({"start": start, "end": end, "raw": raw, "plain": _clean_text(raw)})
        i = j

    has_tags = any("<c>" in b["raw"] or "<00:" in b["raw"] for b in blocks)

    if not has_tags:
        # simple format: each cue is a sentence
        for b in blocks:
            if not b["plain"] or re.match(r"^\(.*\)$", b["plain"]):
                continue
            cues.append({"start": b["start"], "end": b["end"], "text": b["plain"]})
        return cues

    # word-aligned: for each plain (rolling) cue, the audio window it describes
    # is [previous block start, this block start]; its text is the reference.
    prev_start: float | None = None
    for b in blocks:
        if "<" in b["raw"]:
            prev_start = b["start"]
            continue
        # plain rolling cue: reference for [prev_start, b.start]
        txt = b["plain"]
        if not txt or re.match(r"^\(.*\)$", txt):
            continue
        if prev_start is not None and b["start"] > prev_start:
            cues.append({"start": prev_start, "end": b["start"], "text": txt})
            prev_start = b["start"]
    return cues


def build_dataset(out_dir: Path, scraped_dir: Path, max_clips: int, seed: int) -> list[Path]:
    """Slice short clean utterances from scraped webm+vtt into wav+txt pairs."""
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    audio_dir = scraped_dir / "audio"
    cap_dir = scraped_dir / "captions"

    clips: list[Path] = []
    for webm in sorted(audio_dir.glob("*.webm")):
        stem = webm.stem
        # find a matching RU caption: exact {stem}.ru.vtt, else a RU-titled
        # caption that contains the literal "[{stem}]" id (glob [..] would be a
        # char class, so scan + check the string directly)
        vtt = cap_dir / f"{stem}.ru.vtt"
        if not vtt.exists():
            vtt = next(
                (p for p in cap_dir.glob("*.ru.vtt") if f"[{stem}]" in p.name),
                None,
            )
        if vtt is None or not vtt.exists():
            print(f"  skip {stem}: no RU caption")
            continue

        duration = float(
            subprocess.run(  # noqa: S602 - fixed argv, no shell
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", str(webm)],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )

        slices = []
        for cue in _parse_vtt(vtt):
            dur = cue["end"] - cue["start"]
            words = normalize(cue["text"]).split()
            if not 1.2 <= dur <= 6.0:
                continue
            if not 2 <= len(words) <= 16:
                continue
            if cue["end"] > duration - 0.5:
                continue
            slices.append(cue)

        rng.shuffle(slices)
        # cap per video, then global
        need = min(max(max_clips // len(list(audio_dir.glob("*.webm"))), 1), max_clips - len(clips))
        for cue in slices[:need]:
            if len(clips) >= max_clips:
                break
            name = f"{stem}_{len(clips):03d}"
            wav_path = out_dir / f"{name}.wav"
            txt_path = out_dir / f"{name}.txt"
            pad = 0.15
            start = max(0.0, cue["start"] - pad)
            end = min(duration, cue["end"] + pad)
            subprocess.run(  # noqa: S603 - fixed argv
                [
                    "ffmpeg", "-y", "-v", "error",
                    "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
                    "-i", str(webm),
                    "-ar", str(SR), "-ac", "1",
                    str(wav_path),
                ],
                check=True,
            )
            txt_path.write_text(cue["text"], encoding="utf-8")
            clips.append(wav_path)

    print(f"Built {len(clips)} benchmark clips in {out_dir}")
    return clips


def trim_trailing_silence(audio: np.ndarray, sr: int = SR, win: int = 20, thresh: float = 0.015, keep: float = 0.2) -> np.ndarray:
    """Remove trailing silence (RMS below thresh in `win` ms windows), keep `keep` s tail."""
    n = len(audio)
    step = int(sr * win / 1000)
    if n < step * 2:
        return audio
    frame = audio[: n // step * step].reshape(-1, step)
    rms = np.sqrt(np.mean(frame**2, axis=1))
    idx = np.where(rms >= thresh)[0]
    if idx.size == 0:
        return audio
    end = (idx[-1] + 1) * step
    end = min(n, end + int(sr * keep))
    return audio[:end]


def _whisper_model(name: str):
    from pywhispercpp.model import Model as Whisper
    return Whisper(str(MODELS_DIR / f"ggml-{name}.bin"), n_threads=N_THREADS, language="ru", single_segment=True, print_progress=False)


def _fw_model(name: str):
    from faster_whisper import WhisperModel
    return WhisperModel(name, device="cpu", compute_type="int8", cpu_threads=N_THREADS)


def _transcribe(engine: str, model, audio: np.ndarray) -> str:
    if engine == "whisper.cpp":
        segs = model.transcribe(audio, language="ru", single_segment=True)
        return "".join(s.text for s in segs) if segs else ""
    from faster_whisper import WhisperModel as _  # noqa: F401  (type check)
    segments, _info = model.transcribe(audio, language="ru", beam_size=1, condition_on_previous_text=False, vad_filter=False)
    return "".join(s.text for s in segments)


def run_config(cfg_id: str, engine: str, model_name: str, clips: list[dict], trim: bool) -> dict:
    print(f"\n=== {cfg_id} ({engine}: {model_name}) trim={trim} ===")
    model = (_whisper_model if engine == "whisper.cpp" else _fw_model)(model_name)
    word_errs = char_errs = total_chars = 0.0
    timed = 0.0
    rtfs = []
    per_clip = []
    for clip in clips:
        audio = clip["audio"]
        if trim:
            audio = trim_trailing_silence(audio)
        seconds = len(audio) / SR
        start = time.perf_counter()
        hyp = _transcribe(engine, model, audio)
        elapsed = time.perf_counter() - start
        timed += elapsed
        rtf = elapsed / seconds
        rtfs.append(rtf)
        word_errs += wer(clip["reference"], hyp)
        char_errs += char_error_rate(clip["reference"], hyp)
        total_chars += len(normalize(clip["reference"]))
        per_clip.append({"hyp": hyp, "wer": round(wer(clip["reference"], hyp), 3), "rtf": round(rtf, 2)})
    n = len(clips)
    summary = {
        "config": cfg_id,
        "engine": engine,
        "model": model_name,
        "trim": trim,
        "n": n,
        "avg_wer": round(word_errs / n, 3),
        "avg_cer": round(char_errs / n, 3),
        "avg_rtf": round(np.mean(rtfs), 2),
        "transcribe_seconds": round(timed, 1),
    }
    print(
        f"  WER={summary['avg_wer']:.3f}  CER={summary['avg_cer']:.3f}  "
        f"RTF={summary['avg_rtf']:.2f}  ({timed:.1f}s for {n} clips, {total_chars:.0f} ref chars)"
    )
    return {"summary": summary, "clips": per_clip}


def main() -> None:
    ap = argparse.ArgumentParser(description="Short-utterance STT benchmark harness.")
    ap.add_argument("--configs", default=",".join(CONFIGS),
                    help="comma-separated config ids (default: all)")
    ap.add_argument("--clips", type=Path, default=None,
                    help="use an existing wav+txt clip dir instead of building from scraped")
    ap.add_argument("--scraped", type=Path, default=Path("data/wakeword/scraped"))
    ap.add_argument("--out", type=Path, default=Path("data/wakeword/benchmark"))
    ap.add_argument("--rebuild", action="store_true", help="rebuild the benchmark clip set")
    ap.add_argument("--max-clips", type=int, default=20)
    ap.add_argument("--no-trim", action="store_true", help="skip trailing-silence-trimmed runs")
    ap.add_argument("--json", type=Path, default=None, help="write full per-clip results")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.clips is not None:
        clip_dir = args.clips
    else:
        clip_dir = args.out
        if args.rebuild or not any(clip_dir.glob("*.wav")):
            build_dataset(clip_dir, args.scraped, args.max_clips, args.seed)

    clips = load_clips(clip_dir)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(clips)
    if len(clips) > args.max_clips:
        clips = clips[: args.max_clips]
    clips.sort(key=lambda c: c["reference"])  # stable order for display
    print(f"Loaded {min(len(clips), args.max_clips)} clips")
    for i, c in enumerate(clips):
        print(f"  {i + 1}. {c['reference'][:70]}")

    results = []
    for cfg_id in [c.strip() for c in args.configs.split(",") if c.strip()]:
        if cfg_id not in CONFIGS:
            print(f"  ! unknown config {cfg_id!r}, skipping")
            continue
        engine, model = CONFIGS[cfg_id]
        results.append(run_config(cfg_id, engine, model, clips, trim=False))
        if not args.no_trim:
            results.append(run_config(cfg_id, engine, model, clips, trim=True))

    print("\n=== Summary (best CER first) ===")
    rows = []
    for r in results:
        s = r["summary"]
        label = f"{s['config']}+trim" if s["trim"] else s["config"]
        rows.append((s["avg_cer"], s["avg_wer"], s["avg_rtf"], label, s))
    for cer, wer_v, rtf, label, s in sorted(rows, key=lambda x: (x[0], x[1])):
        print(f"  {label:<30} CER={cer:.3f} WER={wer_v:.3f} RTF={rtf:.2f}")

    if args.json:
        args.json.write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(f"Wrote results to {args.json}")


if __name__ == "__main__":
    main()