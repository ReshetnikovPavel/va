import abc
import asyncio
import ctypes
import queue
import time
from collections import deque

import numpy as np
import sounddevice as sd
from livekit.wakeword import WakeWordModel

import va.pipeline
from va.actions import AssistantResponse

from . import models, nlp

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


async def run() -> None:
    samples_queue: queue.Queue[np.ndarray] = queue.Queue()
    context = Context()

    def callback(
        indata: np.ndarray,
        frames: int,
        time: ctypes._CData,
        status: sd.CallbackFlags,
    ):
        samples_queue.put_nowait(indata.copy())

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        channels=CHANNELS,
        callback=callback,
    ):
        state = Idle()
        print(state)
        while True:
            start_time = time.monotonic()

            context.update_samples(samples_queue)
            old_state = state
            state = await old_state.next(context)
            if type(state) != type(old_state):
                print(state)

            elapsed = time.monotonic() - start_time
            await asyncio.sleep(max(TICK_SECS - elapsed, 0))


class Context:
    def __init__(self) -> None:
        self.wakeword_samples: deque[np.ndarray] = deque()
        self.samples: list[np.ndarray] = []

    def update_samples(self, samples: queue.Queue[np.ndarray]) -> None:
        self.samples.clear()
        while True:
            try:
                block = samples.get_nowait()
            except queue.Empty:
                break
            while len(self.wakeword_samples) * BLOCK_SIZE >= WAKEWORD_BLOCK_SIZE:
                self.wakeword_samples.popleft()
            self.wakeword_samples.append(block)
            self.samples.append(block)


class State(abc.ABC):
    @abc.abstractmethod
    async def next(self, context: Context) -> State: ...


def _predict_wakeword(model: WakeWordModel, samples: deque[np.ndarray]) -> bool:
    if len(samples) * BLOCK_SIZE >= WAKEWORD_BLOCK_SIZE:
        sample = np.concat(samples)
        result = model.predict(sample)
        is_wakeword = any(
            score > WAKEWORD_THRESHOLDS.get(name, 0.40)
            for name, score in result.items()
        )
        if is_wakeword:
            samples.clear()
        return is_wakeword
    return False


class Idle(State):
    async def next(self, context: Context) -> State:
        if _predict_wakeword(models.WAKEWORD, context.wakeword_samples):
            return Listening()
        return self


class Listening(State):
    def __init__(self) -> None:
        self.start = time.monotonic()
        self.recording = []
        self.is_recording = False

    async def next(self, context: Context) -> State:
        if (
            not self.is_recording
            and len(self.recording) == 0
            and time.monotonic() - self.start > LISTENING_WAIT_SECS
        ):
            return Idle()
        self._record_voice(context.samples)
        if not self.is_recording and len(self.recording) > 0:
            recording = np.concat(self.recording).flatten()
            return Processing(recording)
        return self

    def _record_voice(self, samples: list[np.ndarray]) -> None:
        for sample in samples:
            vad_result = models.VAD(sample.T)
            if vad_result:
                if "start" in vad_result:
                    print("start")
                    self.recording.append(sample)
                    self.is_recording = True
                elif "end" in vad_result:
                    print("end")
                    self.is_recording = False
            elif self.is_recording:
                self.recording.append(sample)

def _transcribe(audio: np.ndarray) -> str:
    segments, _info = models.STT.transcribe(
        audio,
        language="ru",
        beam_size=1,
        condition_on_previous_text=False,
        vad_filter=False,
    )
    return "".join(s.text for s in segments)


class Processing(State):
    def __init__(self, recording: np.ndarray) -> None:
        self.recording = recording

    async def next(self, context: Context) -> State:
        text = _transcribe(self.recording)
        if response := await va.pipeline.process(text):
            print(response.display)
            return Speaking(response)
        return Idle()


class Speaking(State):
    def __init__(self, response: AssistantResponse) -> None:
        self.tts_task = asyncio.create_task(_say(response.spoken))

    async def next(self, context: Context) -> State:
        if _predict_wakeword(models.WAKEWORD, context.wakeword_samples):
            self.tts_task.cancel()
            sd.stop()
            return Listening()
        if self.tts_task.done():
            return Idle()
        return self


async def _say(text: str) -> None:
    voices = {"ru": models.RU_TTS, "en": models.EN_TTS, "ja": models.JA_TTS}
    for language, part in nlp.split_by_script(text):
        tts = voices.get(language, models.EN_TTS)
        for chunk in await asyncio.to_thread(tts.synthesize, part):
            await asyncio.to_thread(
                sd.play,
                chunk.audio_int16_array,
                samplerate=chunk.sample_rate,
                blocking=True,
            )
