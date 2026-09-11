import asyncio
import ctypes
import enum
import time
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd
from livekit.wakeword import WakeWordModel
from piper import PiperVoice
from pywhispercpp.model import Model as Whisper
from silero_vad_notorch.model import load_silero_vad
from silero_vad_notorch.utils_vad import VADIterator


class State(enum.Enum):
    Idle = enum.auto()
    Listening = enum.auto()
    Processing = enum.auto()
    Speaking = enum.auto()


TICK_SECS = 0.1
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SIZE = 512

WAKEWORD_BLOCK_SIZE = 32000
# Per-model trigger thresholds (tuned against real recordings,
# multi-piper-voice + owner-voice models):
#   ada_ru - Russian "Ада": 10/10 owner clips >= 0.415, worst negative 0.185,
#            English "Ada" cross-trigger max 0.320 -> use 0.35.
#   ada_en - English "Ada": 6/6 owner clips >= 0.505, worst negative 0.162,
#            Russian "Ада" cross-trigger max 0.143 -> use 0.35.
WAKEWORD_THRESHOLDS = {
    "ada_ru": 0.40,
    "ada_en": 0.40,
}


def _predict_wakeword(model: WakeWordModel, samples: deque[np.ndarray]) -> bool:
    if len(samples) * BLOCK_SIZE >= WAKEWORD_BLOCK_SIZE:
        sample = np.concat(samples)
        result = model.predict(sample)
        return any(
            score > WAKEWORD_THRESHOLDS.get(name, 0.40)
            for name, score in result.items()
        )
    return False


async def _transcribe(stt: Whisper, lang_detect: Whisper, audio: np.ndarray) -> str:
    lang = (await asyncio.to_thread(lang_detect.auto_detect_language, audio))[0][0]
    segments = await asyncio.to_thread(
        stt.transcribe,
        audio,
        single_segment=True,
        translate=False,
        language=lang,
    )
    if segments:
        return segments[0].text
    return ""


async def _say(tts: PiperVoice, text: str) -> None:
    for chunk in await asyncio.to_thread(tts.synthesize, text):
        await asyncio.to_thread(
            sd.play,
            chunk.audio_int16_array,
            samplerate=chunk.sample_rate,
            blocking=True,
        )


async def run() -> None:
    wakeword = WakeWordModel(
        models=[
            Path("models", "wakeword", "ada_ru.onnx"),
            Path("models", "wakeword", "ada_en.onnx"),
        ]
    )
    vad = VADIterator(load_silero_vad(), min_silence_duration_ms=800, speech_pad_ms=300)
    stt = Whisper(
        str(
            Path.home()
            / ".local"
            / "share"
            / "pywhispercpp"
            / "models"
            / "ggml-base.bin"
        )
    )
    lang_detect = Whisper(
        str(
            Path.home()
            / ".local"
            / "share"
            / "pywhispercpp"
            / "models"
            / "ggml-tiny.bin"
        )
    )
    tts = PiperVoice.load(Path("models", "piper", "ru_RU-irina-medium.onnx"))

    wakeword_samples = deque()
    vad_samples = []
    recording = []
    is_recording = False

    def callback(
        indata: np.ndarray,
        frames: int,
        time: ctypes._CData,
        status: sd.CallbackFlags,
    ):

        if len(wakeword_samples) * BLOCK_SIZE >= WAKEWORD_BLOCK_SIZE:
            wakeword_samples.popleft()
        block = indata.copy()
        wakeword_samples.append(block)
        vad_samples.append(block)
        if is_recording:
            recording.append(block)

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        channels=CHANNELS,
        callback=callback,
    ):
        state = State.Idle
        stt_task = None
        tts_task = None

        while True:
            start_time = time.monotonic()

            is_wakeword = _predict_wakeword(wakeword, wakeword_samples)

            match state:
                case State.Idle:
                    if is_wakeword:
                        is_recording = True
                        state = State.Listening
                        print(state)

                case State.Listening:
                    for sample in vad_samples:
                        vad_result = vad(sample.T)
                        if vad_result:
                            if "start" in vad_result:
                                print("start")
                                is_recording = True
                            elif "end" in vad_result:
                                print("end")
                                is_recording = False

                    if recording and not is_recording:
                        rec = np.concat(recording).flatten()
                        recording.clear()
                        stt_task = asyncio.create_task(
                            _transcribe(stt, lang_detect, rec)
                        )
                        state = State.Processing
                        print(state)

                case State.Processing:
                    assert stt_task is not None
                    if stt_task.done():
                        transcribed = stt_task.result()
                        print(transcribed)
                        tts_task = asyncio.create_task(_say(tts, transcribed))
                        state = State.Speaking
                        print(state)

                case State.Speaking:
                    assert tts_task is not None
                    if tts_task.done():
                        state = State.Idle
                        print(state)

            vad_samples.clear()

            elapsed = time.monotonic() - start_time
            await asyncio.sleep(max(TICK_SECS - elapsed, 0))
