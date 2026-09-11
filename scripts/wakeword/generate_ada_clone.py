"""Generate cloned-voice positive clips via XTTS v2 (Coqui TTS).

Replaces the piper-based generic-voice positives with clips that sound like
the real owner.  Uses zero-shot voice cloning from the user's existing
recordings as reference audio.

Writes clips into the output/<model>/{positive_train,positive_test}/ dirs
at indices >= TRAIN_CLONE_START / TEST_CLONE_START (4000 / 3000) so they
never collide with real recordings (3000-3063 train / 2000-2015 test) or
legacy synthetic clips (000000-002899).

The augment pipeline then normalises every clip to 2.0s (aligning
positives to the END of the window), which is how the live rolling-window
classifier sees audio.

Usage:
    python scripts/wakeword/generate_ada_clone.py --config scripts/wakeword/configs/config_ada_ru.yaml
    python scripts/wakeword/generate_ada_clone.py --config scripts/wakeword/configs/config_ada_en.yaml
"""

from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("COQUI_TOS_AGREED", "1")

import numpy as np
import soundfile as sf

from livekit.wakeword.config import load_config

# ---------------------------------------------------------------------------

TRAIN_CLONE_START = 4000
TEST_CLONE_START = 3000

# --- phrase lists (word-final dominant for augment align-to-end) ----------

RU_PHRASES = [
    "Ада.",
    "Ада",
    "Ада!",
    "Ада?",
    "Ада, Ада.",
    "Ада-Ада.",
]

EN_PHRASES = [
    "Ada.",
    "Ada",
    "Ada!",
    "Ada?",
    "Ada, Ada.",
    "Ada-Ada.",
]

# ---------------------------------------------------------------------------

LOUDNESS_TARGET_DB = -16.0
SAMPLE_RATE = 16000


def load_refs(paths: list[str | Path]) -> list[np.ndarray]:
    """Load reference WAVs at 24kHz float32 mono."""
    refs = []
    for p in paths:
        data, sr = sf.read(str(p), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        refs.append(data)
    return refs


def resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio
    duration = len(audio) / src_sr
    target_n = int(round(duration * dst_sr))
    x = np.linspace(0, len(audio) - 1, num=target_n)
    return np.interp(x, np.arange(len(audio)), audio).astype(np.float32)


def normalize(audio: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(audio)) + 1e-8
    peak_gain = 10 ** ((-3.0 - 20 * np.log10(peak)) / 20)
    audio = audio * peak_gain
    rms = np.sqrt(np.mean(audio**2)) + 1e-8
    rms_gain = 10 ** ((LOUDNESS_TARGET_DB - 20 * np.log10(rms)) / 20)
    audio = audio * rms_gain
    peak = np.max(np.abs(audio))
    if peak > 0.95:
        audio = audio * (0.95 / peak)
    return audio.astype(np.float32)


def write_clip(path: Path, audio: np.ndarray) -> None:
    sf.write(str(path), audio, SAMPLE_RATE, subtype="PCM_16")


def next_index(split_dir: Path, start: int) -> int:
    existing = []
    for p in split_dir.glob("clip_*.wav"):
        name = p.stem  # clip_XXXXXX
        idx = int(name.split("_")[1])
        if idx >= start:
            existing.append(idx)
    return max(existing, start - 1) + 1 if existing else start


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--n-test", type=int, default=30)
    args = ap.parse_args()

    cfg = load_config(args.config)
    model_dir = cfg.model_output_dir
    lang = "ru" if "_ru" in cfg.model_name else "en"
    phrases = RU_PHRASES if lang == "ru" else EN_PHRASES

    # Use the long recording as primary reference, short clips as fallback
    ref_dir = Path("data/wakeword/reference")
    long_ref = ref_dir / f"ada_long_{'ru' if lang == 'ru' else 'en'}.wav"
    if long_ref.exists():
        refs = [str(long_ref)]
        print(f"Using long reference: {long_ref}")
    else:
        # fallback to short clips
        short_dir = Path("data/wakeword/positives") / ("ru" if lang == "ru" else "en")
        if lang == "ru":
            refs = [str(p) for p in sorted(short_dir.glob("ada_0[1-9].wav"))] + [str(p) for p in sorted(short_dir.glob("ada_010.wav"))]
        else:
            refs = [str(p) for p in sorted(short_dir.glob("ada_01[1-6].wav"))]
        print(f"Using {len(refs)} short reference clips")
    if not refs:
        sys.exit(f"No reference wavs found for lang={lang}")

    import torch
    from TTS.api import TTS

    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cpu")

    for split, n_target, start_idx in [
        ("positive_train", args.n_train, TRAIN_CLONE_START),
        ("positive_test", args.n_test, TEST_CLONE_START),
    ]:
        split_dir = model_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        idx = next_index(split_dir, start_idx)
        remaining = n_target
        print(f"{split}: generating {remaining} clips starting at clip_{idx:06d}")

        while remaining > 0:
            phrase = random.choice(phrases)
            ref = random.choice(refs)

            t0 = time.time()
            tts.tts_to_file(
                text=phrase,
                speaker_wav=[ref],
                language=lang,
                file_path="/tmp/opencode/_clone_tmp.wav",
                temperature=random.uniform(0.6, 0.9),
            )
            gen_time = time.time() - t0

            data, sr = sf.read("/tmp/opencode/_clone_tmp.wav", dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            data = resample(data, sr, SAMPLE_RATE)
            data = normalize(data)

            out_path = split_dir / f"clip_{idx:06d}.wav"
            write_clip(out_path, data)
            idx += 1
            remaining -= 1

            if remaining % 20 == 0 or remaining == 0:
                print(
                    f"  {split}: {idx - start_idx}/{n_target}  "
                    f"(gen={gen_time:.1f}s)  phrase={phrase!r}"
                )

    print("Clone generation complete!")


if __name__ == "__main__":
    main()
