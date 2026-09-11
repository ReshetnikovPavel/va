"""Generate synthetic wake word clips using piper-tts with multiple voices.

Bypasses livekit-wakeword's own ``generate`` (which needs a multi-speaker VITS
state_dict + espeak-ng) and instead synthesizes clips directly with piper-tts
against every voice in ``models/piper/*.onnx`` for the requested language,
using randomized SynthesisConfig (length scale, noise, volume) for further
speaker/pace/volume diversity.  A random voice is picked per clip.

Writes clips in the exact layout the ``augment`` command expects:

    output/<model_name>/{positive,negative}_{train,test}/clip_######.wav
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from piper import PiperVoice
from piper.config import SynthesisConfig
from livekit.wakeword.config import load_config
from livekit.wakeword.data.generate import _generate_background_clips

LOUDNESS_TARGET = -16.0

# Per-language voice pools (included in models/piper/).  Every synthetic clip
# uses a randomly chosen voice from its language pool for speaker diversity.
RU_VOICES = [
    "ru_RU-irina-medium.onnx",
    "ru_RU-denis-medium.onnx",
    "ru_RU-dmitri-medium.onnx",
    "ru_RU-ruslan-medium.onnx",
]
EN_VOICES = [
    "en_US-lessac-medium.onnx",
    "en_US-ryan-medium.onnx",
    "en_US-joe-medium.onnx",
]

_LANG_BY_VOICE = {v: ("ru" if v.startswith("ru_") else "en") for v in RU_VOICES + EN_VOICES}


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Normalize to dBFS peak and RMS loudness target."""
    audio = np.asarray(audio, dtype=np.float64)
    if audio.size == 0:
        return audio

    peak = np.max(np.abs(audio))
    if peak > 0:
        db_peak = 20 * np.log10(max(peak, 1e-8))
        target_peak_db = -3.0
        peak_gain = target_peak_db - db_peak
        audio = audio * (10 ** (peak_gain / 20))

    rms = np.sqrt(np.mean(audio**2)) + 1e-8
    db_rms = 20 * np.log10(rms)
    rms_gain = LOUDNESS_TARGET - db_rms
    audio = audio * (10 ** (rms_gain / 20))

    peak = np.max(np.abs(audio))
    if peak > 0.95:
        audio = audio * (0.95 / peak)

    return audio.astype(np.float32)


def synthesize_clip(
    voice: PiperVoice,
    text: str,
    sample_rate: int = 16000,
    synth_config: SynthesisConfig | None = None,
) -> np.ndarray:
    """Synthesize text to a 16kHz mono float32 array."""
    chunks = list(voice.synthesize(text, syn_config=synth_config))
    if not chunks:
        raise RuntimeError(f"Empty synthesis for {text!r}")

    int16 = np.concatenate([chunk.audio_int16_array for chunk in chunks])
    audio = int16.astype(np.float32) / 32768.0
    audio = normalize_audio(audio)

    if chunks[0].sample_rate != sample_rate:
        n = audio.shape[0]
        duration = n / chunks[0].sample_rate
        target_n = int(round(duration * sample_rate))
        x = np.linspace(0, n - 1, num=target_n)
        audio = np.interp(x, np.arange(n), audio)

    return audio.astype(np.float32)


def write_clip(out_path: Path, audio: np.ndarray, sample_rate: int = 16000) -> None:
    import soundfile as sf

    sf.write(str(out_path), audio, sample_rate)


def random_synth_config() -> SynthesisConfig:
    """Randomized synthesis parameters for speaker/pace/volume diversity."""
    return SynthesisConfig(
        length_scale=random.choice([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]),
        noise_scale=random.choice([0.3, 0.4, 0.5, 0.667, 0.8, 0.9]),
        noise_w_scale=random.choice([0.3, 0.5, 0.7, 0.8, 0.9]),
        normalize_audio=True,
        volume=random.choice([0.7, 0.8, 0.9, 1.0, 1.1, 1.2]),
    )


def load_voices(lang: str) -> dict[str, PiperVoice]:
    """Load all piper voices for a language from models/piper/."""
    names = [n for n, l in _LANG_BY_VOICE.items() if l == lang]
    return {n: PiperVoice.load(Path("models", "piper", n)) for n in names}


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    lang = cfg.model_name.split("_")[-1]  # "ada_ru" -> "ru", "ada_en" -> "en"
    voices = load_voices(lang)
    if not voices:
        raise RuntimeError(f"no piper voices registered for language {lang!r}")
    print(f"using {len(voices)} piper voices: {list(voices)}")

    splits = [
        ("positive_train", cfg.target_phrases, cfg.n_samples),
        ("positive_test", cfg.target_phrases, cfg.n_samples_val),
        ("negative_train", cfg.custom_negative_phrases, cfg.n_samples),
        ("negative_test", cfg.custom_negative_phrases, cfg.n_samples_val),
    ]

    for split_name, phrases, n_target in splits:
        split_dir = cfg.model_output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        existing = len(list(split_dir.glob("clip_*.wav")))
        if existing >= n_target:
            print(f"{split_name}: already complete ({existing}/{n_target})")
            continue
        print(f"{split_name}: generating {n_target - existing} clips...")

        i = existing
        while i < n_target:
            voice = random.choice(list(voices.values()))
            phrase = random.choice(phrases)
            try:
                audio = synthesize_clip(voice, phrase, synth_config=random_synth_config())
            except Exception as e:
                print(f"  synth failed for {phrase!r}: {e}")
                continue
            out_path = split_dir / f"clip_{i:06d}.wav"
            write_clip(out_path, audio)
            i += 1
            if i % 100 == 0:
                print(f"  {i}/{n_target}")

        print(f"{split_name}: done ({i} clips)")

    if cfg.n_background_samples > 0:
        _generate_background_clips(cfg, "background_train", cfg.n_background_samples)
    if cfg.n_background_samples_val > 0:
        _generate_background_clips(cfg, "background_test", cfg.n_background_samples_val)

    print("Generation complete!")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="scripts/wakeword/configs/config_ada.yaml")
    args = ap.parse_args()
    main(args.config)