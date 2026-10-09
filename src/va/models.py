import laya
import importlib.metadata
import os
from pathlib import Path

import faster_whisper
import livekit.wakeword
import llama_cpp
import natasha
import piper
import pymorphy3
import silero_vad_notorch
from pymorphy2 import analyzer as _pymorphy2_analyzer

from va import decima

WAKEWORD = livekit.wakeword.WakeWordModel(
    models=[
        Path("models", "wakeword", "ada_ru.onnx"),
        Path("models", "wakeword", "ada_en.onnx"),
    ]
)
VAD = silero_vad_notorch.VADIterator(
    silero_vad_notorch.load_silero_vad(), min_silence_duration_ms=800, speech_pad_ms=300
)
STT = faster_whisper.WhisperModel(
    "medium",
    device="cpu",
    compute_type="int8",
    cpu_threads=8,
)
RU_TTS = piper.PiperVoice.load(Path("models", "piper", "ru_RU-irina-medium.onnx"))
EN_TTS = piper.PiperVoice.load(Path("models", "piper", "en_US-amy-medium.onnx"))
JA_TTS = piper.PiperVoice.load(
    Path("models", "piper", "ja_JP-hi_fi_captain-medium.onnx")
)

QWEN_2_5_1_5B = llama_cpp.Llama(
    model_path=os.path.join("models", "llm", "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
    n_ctx=512,
    n_gpu_layers=0,
    verbose=False,
)

MORPH_ANALYZER = pymorphy3.analyzer.MorphAnalyzer()

# natasha used pkg_resources which is deprecated. Monkey-patch
_pymorphy2_analyzer._iter_entry_points = lambda *groups, **_: (  # ty: ignore[invalid-assignment]
    importlib.metadata.entry_points(group=groups[0])
)

NATASHA_SEGMENTER = natasha.Segmenter()
NATASHA_MORPH_VOCAB = natasha.MorphVocab()
NATASHA_EMBEDDING = natasha.NewsEmbedding()
NATASHA_MORPH_TAGGER = natasha.NewsMorphTagger(NATASHA_EMBEDDING)
NATASHA_SYNTAX_PARSER = natasha.NewsSyntaxParser(NATASHA_EMBEDDING)
NATASHA_NER_TAGGER = natasha.NewsNERTagger(NATASHA_EMBEDDING)

LAYA_ROUTER = laya.Router(preload=True, lang_guess="ru")
DECIMA = decima.Decima(Path("models", "decima-base"))
