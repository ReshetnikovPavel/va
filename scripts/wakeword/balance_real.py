"""Balance and inject RAW (un-augmented) real-voice features into training.

Why:
  * The ``augment`` step normalizes every clip to exactly 2.0s and adds
    noise/EQ/RIR/background before feature extraction, producing ``_r0.wav``.
  * At inference the app runs the SAME frontend directly on manufacturing
    audio, so the effective input distribution differs (raw vs augmented).
  * Real recordings are also heavily in the pipeline, but only via the
    augmented path.  This script extracts features straight from the RAW
    real clips and injects them as extra rows into the training .npy, so the
    classifier sees genuine raw-domain real-voice samples.

This also re-balances: raw hints are written as a ja/max-oversampling budget
via ``--dup`` which repeats each raw real sample N extra times (default 4).

Backups of the pre-injection arrays are saved as ``*.npy.pre-raw.npy`` so the
injection is reproducible/revertible.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import soundfile as sf

from livekit.wakeword.models.feature_extractor import MelSpectrogramFrontend, SpeechEmbedding
from livekit.wakeword.resources import get_embedding_model_path, get_mel_model_path

# The clip files that correspond to REAL-voice clips are identifiable by their
# source property: real bases were written starting at clip index 3000 (train)
# and 2000 (test).  Everything authored by prep_real_data.py lives at/above
# these indices.
TRAIN_REAL_START = 3000
TEST_REAL_START = 2000
N_EMBED_TIMESTEPS = 16

# Inclusive max clip index authored by prep_real_data.py (user recordings).
# prep numbers its variant files first; scraped positives (scrape_youtube.py)
# and cross-model mirrors (mirror_cross_negatives.py) append at/above these.
USER_INDEX_MAX = {"ru": 3063, "en": 3039}

# Cloned-voice synthetic clips (generate_ada_clone.py) are written at/above
# these indices so they are clearly distinct from real recordings and are
# NEVER raw-injected (raw injection is reserved for the owner's mic audio).
TRAIN_CLONE_START = 4000
TEST_CLONE_START = 3000


def standardize_audio(audio: np.ndarray, target_duration: float = 2.0, sr: int = 16000) -> np.ndarray:
    """Reshape arbitrary-length mono audio to the frontend/eval input shape.

    The app feeds a fixed 32256-sample window (2.016s).  For training we use
    the same 2.0s target as the augment step: pad-end or center-crop.
    """
    n = audio.size
    target_n = int(round(target_duration * sr))
    if n < target_n:
        padded = np.zeros(target_n, dtype=np.float32)
        padded[target_n - n :] = audio
        return padded
    if n > target_n:
        return audio[-target_n:]
    return audio


def compute_raw_feature(path: Path, mel: MelSpectrogramFrontend, emb: SpeechEmbedding) -> np.ndarray:
    audio, sr = sf.read(str(path))
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = audio.astype(np.float32)
    audio = standardize_audio(audio)
    mel_feats = mel(audio[np.newaxis, :])
    e = emb.extract_embeddings(mel_feats)[0]  # (n_windows, 96)
    if e.shape[0] >= N_EMBED_TIMESTEPS:
        return e[-N_EMBED_TIMESTEPS:]
    pad = np.zeros((N_EMBED_TIMESTEPS - e.shape[0], 96), dtype=np.float32)
    return np.concatenate([pad, e], axis=0)


def real_indices(clip_dir: Path, start_idx: int, tag: str) -> list[int]:
    """Return sorted clip indices of RAW real-voice originals in a directory."""
    out: list[int] = []
    for p in clip_dir.glob("clip_*.wav"):
        m = re.match(r"clip_(\d{6})\.wav$", p.name)
        if m and int(m.group(1)) >= start_idx and m.group(1).startswith(tag):
            out.append(int(m.group(1)))
    return sorted(out)


def main(model: str, dup: int, user_dup: int) -> None:
    mel = MelSpectrogramFrontend(get_mel_model_path())
    emb = SpeechEmbedding(get_embedding_model_path())
    out = Path("output") / f"ada_{model}"
    user_max = USER_INDEX_MAX[model]

    for split, feature_file, start_idx, end_idx in [
        ("positive_train", "positive_features_train.npy", TRAIN_REAL_START, TRAIN_CLONE_START),
        ("negative_train", "negative_features_train.npy", TRAIN_REAL_START, TRAIN_CLONE_START),
        ("positive_test", "positive_features_test.npy", TEST_REAL_START, TEST_CLONE_START),
        ("negative_test", "negative_features_test.npy", TEST_REAL_START, TEST_CLONE_START),
    ]:
        clip_dir = out / split
        fpath = out / feature_file
        if not clip_dir.exists() or not fpath.exists():
            print(f"[skip] {split}: missing dir/features")
            continue

        feats = np.load(str(fpath), mmap_mode=None)
        print(f"{split}: loaded features {feats.shape}")

        # Collect raw real clips (real-use zone: [start_idx, end_idx) in this split dir).
        raws: list[tuple[int, np.ndarray]] = []
        for p in sorted(clip_dir.glob("clip_*.wav")):
            m = re.match(r"clip_(\d{6})\.wav$", p.name)
            if m and start_idx <= int(m.group(1)) < end_idx:
                raws.append((int(m.group(1)), compute_raw_feature(p, mel, emb)))
        if not raws:
            print(f"{split}: no raw real clips to inject")
            continue

        # Per-row duplication: user-recorded POSITIVES get extra weight so the
        # model stays sensitive to the owner's voice; scraped/external voices
        # and all negatives get the base dup.
        rows = np.stack([f for _, f in raws], axis=0).astype(np.float32)
        n_user_pos = sum(1 for idx, _ in raws if idx <= user_max and "positive" in split)
        print(f"{split}: {rows.shape[0]} raw real features ({n_user_pos} user positives) @ {feature_file}")

        n_dup = dup if "train" in split else 1
        n_user_dup = user_dup if "positive" in split and "train" in split else n_dup

        injected = [feats]
        for i, (idx, f) in enumerate(raws):
            rep = n_user_dup if idx <= user_max and "positive" in split else n_dup
            injected.append(rows[i : i + 1])
            for _ in range(rep - 1):
                injected.append(rows[i : i + 1])
        injected = np.concatenate(injected, axis=0)

        # Back up pre-injection array once (first injection only).
        backup = fpath.with_suffix(".npy.pre-raw.npy")
        if not backup.exists():
            np.save(str(backup), feats)
        np.save(str(fpath), injected)
        print(f"{split}: features {feats.shape} -> {injected.shape} (dup={n_dup}, user_pos_dup={n_user_dup})")

    print("balance_real done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["ru", "en"], required=True)
    ap.add_argument("--dup", type=int, default=4, help="repeat count for real rows except user positives")
    ap.add_argument("--user-dup", type=int, default=4, help="repeat count for user-recorded positive rows")
    args = ap.parse_args()
    main(args.model, args.dup, args.user_dup)