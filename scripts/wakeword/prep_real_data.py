"""Copy real voice recordings into per-model clip sets.

Splits the user's ``data/wakeword`` recordings by language:
  * positives/ru/ada_01..010  -> Russian "Ада"  (used by the ``ru`` model)
  * positives/en/ada_011..016 -> English "Ada"  (used by the ``en`` model)
  * negatives/near_miss       -> Russian near-miss negatives (rejection data)

Each base is turned into 8 on-domain variants (pitch +-1/+-2 semitones,
time-stretch x0.85/x1.15, pitch+stretch combo) so the raw real-voice domain
has enough weight in training.  The original recording is written alongside
as ``clip_######.wav``; the pipeline ``augment`` then add noise/RIR/background
diversity and ``balance_real.py`` injects the un-augmented originals as raw
features (fixing the raw-vs-augmented inference mismatch).

Numbering keeps synthetic clips (train ~0..2999, test 0..499) untouched:
real clips start at 003000 (train) / 002000 (test).
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf

sys = __import__("sys")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.wakeword.generate_ada_clips import normalize_audio

SR = 16000
VARIANT_RULES = [
    ("orig", None),
    ("p+1", "pitch+1"),
    ("p-1", "pitch-1"),
    ("p+2", "pitch+2"),
    ("p-2", "pitch-2"),
    ("t85", "stretch0.85"),
    ("t115", "stretch1.15"),
    ("pt", "pitch+stretch"),
]

ADA_REAL = Path("data/wakeword")

SPLITS = {
    "ru": {  # (base source, home dir) for each split
        "positive_train": [
            "ada_01.wav", "ada_02.wav", "ada_04.wav", "ada_05.wav",
            "ada_06.wav", "ada_09.wav", "ada_010.wav",
        ],
        # NOTE: ada_07.wav is intentionally EXCLUDED - it was a whispered
        # "Ада" that the model cannot reliably separate from speech negatives.
        "positive_test": ["ada_03.wav", "ada_08.wav"],
        # NOTE: the user's near-miss *_neg recordings are intentionally NOT
        # used as training negatives (they are too strong / too close to the
        # target word from the owner's own voice).
        # Conversational speech (recorded from the owner while talking) IS
        # used as negatives so the model rejects ordinary speech.
        "negative_train": (["conversation_02.wav", "conversation_03.wav",
                            "conversation_04.wav", "conversation_05.wav",
                            "conversation_06.wav"]
                           + [f"myspeech_{i:03d}.wav" for i in range(1, 81)]),
        "negative_test": (["conversation_03.wav", "conversation_06.wav"]
                          + [f"myspeech_{i:03d}.wav" for i in range(81, 100)]),
    },
    "en": {
        "positive_train": [
            "ada_011.wav", "ada_012.wav", "ada_014.wav", "ada_015.wav",
            "ada_016.wav",
        ],
        "positive_test": ["ada_013.wav"],
        "negative_train": [],
        "negative_test": [],
    },
}

# Extra sources appended to absolute (language-crossing) negatives:
#   ru model rejects English "Ada" and other wake words,
#   en model rejects Russian "Ада".
CROSS_NEG = {
    "ru": {
        "negative_train": ["ada_011.wav", "ada_012.wav", "ada_013.wav",
                           "ada_014.wav", "ada_015.wav", "ada_016.wav"]
        + ["alexa.wav", "hey_billy.wav", "hey_jarvis.wav", "hey_livekit.wav",
           "hey_mycroft.wav", "hey_rhasspy.wav"],
        "negative_test": ["alexa.acc.wav", "hey_billy.acc.wav"],
    },
    "en": {
        "negative_train":
            ["ada_01.wav", "ada_02.wav", "ada_03.wav", "ada_04.wav",
             "ada_05.wav", "ada_06.wav", "ada_07.wav", "ada_08.wav",
             "ada_09.wav", "ada_010.wav"]
            + ["alexa.wav", "hey_billy.wav", "hey_jarvis.wav",
               "hey_livekit.wav", "hey_mycroft.wav", "hey_rhasspy.wav"],
        "negative_test": ["alexa.acc.wav", "hey_billy.acc.wav"],
    },
}

TRAIN_REAL_START = 3000
TEST_REAL_START = 2000


def resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    n = len(audio)
    if n < 2 or sr_in == sr_out:
        return audio
    out_n = int(round(n * sr_out / sr_in))
    x_old = np.arange(n)
    x_new = np.linspace(0, n - 1, num=out_n)
    return np.interp(x_new, x_old, audio)


def pitch_shift(audio: np.ndarray, semitones: int) -> np.ndarray:
    factor = 2 ** (semitones / 12.0)
    sped = resample(audio, SR, int(SR * factor))
    return resample(sped, int(SR * factor), SR)


def time_stretch(audio: np.ndarray, rate: float) -> np.ndarray:
    return resample(audio, SR, int(round(SR / rate)))


def apply_variant(audio: np.ndarray, rule: str | None) -> np.ndarray:
    if rule is None:
        return audio
    if rule == "pitch+stretch":
        return time_stretch(pitch_shift(audio, 2), 1.15)
    if rule.startswith("pitch"):
        return pitch_shift(audio, int(rule[5:]))
    if rule.startswith("stretch"):
        return time_stretch(audio, float(rule[7:]))
    raise ValueError(rule)


def trim_silence(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    peak = np.max(np.abs(audio))
    if peak <= 0:
        return audio
    audio = audio / peak
    frame_len = int(sr * 0.02)
    n_frames = len(audio) // frame_len
    frames = audio[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    thr = 0.02
    on = np.where(rms > thr)[0]
    if on.size == 0:
        return audio
    margin = int(sr * 0.1)
    start = max(0, on[0] * frame_len - margin)
    end = min(len(audio), (on[-1] + 1) * frame_len + margin)
    return audio[start:end]


def load_base(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path))
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = audio.astype(np.float32)
    if sr != SR:
        audio = resample(audio, sr, SR)
    audio = trim_silence(audio)
    return normalize_audio(audio)


def write_clip(out: Path, audio: np.ndarray) -> None:
    sf.write(str(out), audio, SR)


def find_base(base: str) -> Path:
    """Locate a base recording by name anywhere under ``data/wakeword``.

    Picks the deepest match so a name like ``ada_01.wav`` that could exist in
    both ``positives/ru`` and ``positives/en`` resolves to the language dir.
    """
    ww = Path("data/wakeword")
    hits: list[Path] = []
    for p in ww.rglob(base):
        if p.is_file():
            hits.append(p)
    if not hits:
        raise FileNotFoundError(f"base {base!r} not found under {ww}")
    return max(hits, key=lambda p: len(p.parts))


def main(model: str) -> None:
    assert model in ("ru", "en"), f"unknown model {model}"
    out_root = Path("output") / f"ada_{model}"

    spec = SPLITS[model]
    extra = CROSS_NEG[model]

    for split in ["positive_train", "positive_test", "negative_train", "negative_test"]:
        bases: list[str] = list(spec.get(split, [])) + list(extra.get(split, []))
        if not bases:
            continue
        start = TRAIN_REAL_START if "train" in split else TEST_REAL_START
        home = out_root / split

        idx = start
        for base in bases:
            path = find_base(base)

            audio = load_base(path)
            for tag, rule in VARIANT_RULES:
                v = apply_variant(audio, rule) if rule else audio
                v = normalize_audio(v)
                name = f"clip_{idx:06d}.wav"
                write_clip(home / name, v)
                idx += 1
        print(f"{split}: added {idx - start} real clips ({len(bases)} bases x {len(VARIANT_RULES)})")

    print(f"ada_{model} real data prep complete.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["ru", "en"], required=True)
    args = ap.parse_args()
    main(args.model)