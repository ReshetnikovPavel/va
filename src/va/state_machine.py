import asyncio
import ctypes
import enum
import time
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from livekit.wakeword import WakeWordModel
from piper import PiperVoice
from silero_vad_notorch.model import load_silero_vad
from silero_vad_notorch.utils_vad import VADIterator

from va.actions import ActionError, AssistantResponse

from . import intent, nlp
from .actions.music import player
from .actions.weather import get_weather


class State(enum.Enum):
    Idle = enum.auto()
    Listening = enum.auto()
    Processing = enum.auto()
    Speaking = enum.auto()


TICK_SECS = 0.1
LISTENING_WAIT_SECS = 2
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


async def _transcribe(stt: WhisperModel, audio: np.ndarray) -> str:
    segments, _info = await asyncio.to_thread(
        stt.transcribe,
        audio,
        language="ru",
        beam_size=1,
        condition_on_previous_text=False,
        vad_filter=False,
    )
    return "".join(s.text for s in segments)


async def _say(ru_tts: PiperVoice, en_tts: PiperVoice, text: str) -> None:
    for language, part in nlp.split_by_script(text):
        tts = en_tts if language == "en" else ru_tts
        for chunk in await asyncio.to_thread(tts.synthesize, part):
            await asyncio.to_thread(
                sd.play,
                chunk.audio_int16_array,
                samplerate=chunk.sample_rate,
                blocking=True,
            )


async def _execute_action(
    action: intent.Intent, data: dict
) -> AssistantResponse | None:
    match action:
        case intent.Intent.Weather:
            return await get_weather(data["location"])
        case intent.Intent.PauseMusic:
            return player.pause_music()
        case intent.Intent.NowPlaying:
            return player.now_playing()
        case intent.Intent.PlayMusic:
            return await player.play_music(data["song"])
        case intent.Intent.NextTrack:
            return player.next_track()
        case intent.Intent.PreviousTrack:
            return player.previous_track()
        case intent.Intent.VolumeUp:
            return player.volume_up()
        case intent.Intent.VolumeDown:
            return player.volume_down()
        case intent.Intent.VolumeMuchUp:
            return player.volume_much_up()
        case intent.Intent.VolumeMuchDown:
            return player.volume_much_down()
        case intent.Intent.Unknown:
            return AssistantResponse("Я глупая")
        case unhandled:
            raise RuntimeError(f"Unknown intent: `{unhandled}`")


async def run() -> None:
    wakeword = WakeWordModel(
        models=[
            Path("models", "wakeword", "ada_ru.onnx"),
            Path("models", "wakeword", "ada_en.onnx"),
        ]
    )
    vad = VADIterator(load_silero_vad(), min_silence_duration_ms=800, speech_pad_ms=300)
    stt = WhisperModel(
        "medium",
        device="cpu",
        compute_type="int8",
        cpu_threads=8,
    )
    ru_tts = PiperVoice.load(Path("models", "piper", "ru_RU-irina-medium.onnx"))
    en_tts = PiperVoice.load(Path("models", "piper", "en_US-amy-medium.onnx"))

    wakeword_samples = deque()
    vad_samples = []
    recording = []
    is_recording = False
    listening_start = None

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

        print(state)
        while True:
            start_time = time.monotonic()

            is_wakeword = _predict_wakeword(wakeword, wakeword_samples)

            match state:
                case State.Idle:
                    if is_wakeword:
                        listening_start = time.monotonic()
                        state = State.Listening
                        print(state)

                case State.Listening:
                    assert listening_start is not None
                    if (
                        not recording
                        and time.monotonic() - listening_start > LISTENING_WAIT_SECS
                    ):
                        state = State.Idle
                        print(state)
                    else:
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
                            stt_task = asyncio.create_task(_transcribe(stt, rec))
                            state = State.Processing
                            print(state)

                case State.Processing:
                    assert stt_task is not None
                    if is_wakeword:
                        stt_task.cancel()
                        state = state.Idle
                        print(state)
                    elif stt_task.done():
                        transcribed = stt_task.result()
                        print(transcribed)

                        try:
                            action = intent.classify(transcribed)
                            data = intent.extract_data(transcribed, action)
                            response = await _execute_action(action, data)
                        except ActionError as e:
                            print(e)
                            response = AssistantResponse(
                                "Произошла какая-то ошибка, простите"
                            )

                        if response is not None:
                            print(response.display)
                            tts_task = asyncio.create_task(_say(ru_tts, en_tts, response.spoken))
                            state = State.Speaking
                            print(state)
                        else:
                            state = State.Idle
                            print(state)

                case State.Speaking:
                    assert tts_task is not None
                    if is_wakeword:
                        tts_task.cancel()
                        state = state.Idle
                        print(state)
                    elif tts_task.done():
                        state = State.Idle
                        print(state)

            vad_samples.clear()

            elapsed = time.monotonic() - start_time
            await asyncio.sleep(max(TICK_SECS - elapsed, 0))
