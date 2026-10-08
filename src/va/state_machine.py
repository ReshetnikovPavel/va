import abc
import asyncio
import ctypes
import queue
import sys
import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd

import va.actions.player
import va.pipeline
from va.actions import AssistantResponse

from . import models, nlp

TICK_SECS = 0.1
LISTENING_WAIT_SECS = 2
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SIZE = 512

WAKEWORD_BLOCK_SIZE = 32000
WAKEWORD_THRESHOLDS = {
    "ada_ru": 0.40,
    "ada_en": 0.40,
}


async def run() -> None:
    text_queue: queue.Queue[str] = queue.Queue()
    samples_queue: queue.Queue[np.ndarray] = queue.Queue()

    def _stdin_reader():
        for line in sys.stdin:
            if line := line.strip():
                text_queue.put(line)

    threading.Thread(target=_stdin_reader, daemon=True).start()

    def audio_callback(
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
        callback=audio_callback,
    ):
        ctx = Context()
        state = Idle()
        print(state)
        while True:
            start_time = time.monotonic()

            ctx.update(samples_queue, text_queue)
            old_state = state
            state = await old_state.next(ctx)
            if type(state) != type(old_state):
                print(state)

            elapsed = time.monotonic() - start_time
            await asyncio.sleep(max(TICK_SECS - elapsed, 0))


class Context:
    def __init__(self) -> None:
        self.wakeword_samples: deque[np.ndarray] = deque()
        self.samples: list[np.ndarray] = []
        self.text_input: list[str] = []

    def update(self, samples: queue.Queue[np.ndarray], text: queue.Queue[str]) -> None:
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
        while True:
            try:
                self.text_input.append(text.get_nowait())
            except queue.Empty:
                break


def _predict_wakeword(ctx: Context) -> bool:
    is_talking = False
    for sample in ctx.samples:
        models.VAD(sample.T)
        if models.VAD.triggered:
            is_talking = True
            break

    if is_talking and len(ctx.wakeword_samples) * BLOCK_SIZE >= WAKEWORD_BLOCK_SIZE:
        sample = np.concat(ctx.wakeword_samples)
        result = models.WAKEWORD.predict(sample)
        is_wakeword = any(
            score > WAKEWORD_THRESHOLDS.get(name, 0.40)
            for name, score in result.items()
        )
        if is_wakeword:
            ctx.wakeword_samples.clear()
        return is_wakeword

    return False


class State(abc.ABC):
    @abc.abstractmethod
    async def next(self, ctx: Context) -> State: ...


class Idle(State):
    async def next(self, ctx: Context) -> State:
        if ctx.text_input:
            text = "\n".join(ctx.text_input)
            ctx.text_input.clear()
            return Processing(text)
        if _predict_wakeword(ctx):
            return Listening()
        return self


class Listening(State):
    def __init__(self) -> None:
        self.start = time.monotonic()
        self.recording = []

    async def next(self, ctx: Context) -> State:
        if (
            not models.VAD.triggered
            and not self.recording
            and time.monotonic() - self.start > LISTENING_WAIT_SECS
        ):
            return Idle()

        for sample in ctx.samples:
            models.VAD(sample.T)
            if models.VAD.triggered:
                self.recording.append(sample)

        if not models.VAD.triggered and self.recording:
            recording = np.concat(self.recording).flatten()
            return Transcribing(recording)
        return self


def _transcribe(audio: np.ndarray) -> str:
    segments, _info = models.STT.transcribe(
        audio,
        language="ru",
        beam_size=1,
        condition_on_previous_text=False,
        vad_filter=False,
    )
    return "".join(s.text for s in segments)


class Transcribing(State):
    def __init__(self, recording: np.ndarray) -> None:
        self.recording = recording

    async def next(self, ctx: Context) -> State:
        text = _transcribe(self.recording)
        return Processing(text)


class Processing(State):
    def __init__(self, text: str) -> None:
        self.text = text

    async def next(self, ctx: Context) -> State:
        if response := await va.pipeline.process(self.text):
            print(response.display)
            return Speaking(response)
        return Idle()


class Speaking(State):
    def __init__(self, response: AssistantResponse) -> None:
        self.tts_task = asyncio.create_task(_say(response.spoken))

    async def next(self, ctx: Context) -> State:
        if _predict_wakeword(ctx):
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
