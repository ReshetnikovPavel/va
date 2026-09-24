import importlib.metadata
import re
import typing
from typing import Literal

import natasha
import num2words
import pymorphy3.analyzer
from pymorphy2 import analyzer as _pymorphy2_analyzer

# Падежи русского языка (граммемы pymorphy).
#   nomn — именительный: кто? что?            (стол, вода)
#   gent — родительный: кого? чего?           (стола, воды)
#   datv — дательный: кому? чему?             (столу, воде)
#   accs — винительный: кого? что?            (стол, воду)
#   ablt — творительный: кем? чем?            (столом, водой)
#   loct — предложный: о ком? о чём?          (о столе, о воде)
#   gen2 — второй родительный (партитив):     (чашку чаю, много снегу)
#   loc2 — второй предложный (местный):       (в лесу, на берегу)
#   voct — звательный: реликт, в русском почти не используется (Господи)
#   acc2 — второй винительный: в русском практически не встречается
Case = Literal[
    "nomn", "gent", "datv", "accs", "ablt", "loct", "gen2", "loc2", "voct", "acc2"
]

MORPH_ANALYZER = pymorphy3.analyzer.MorphAnalyzer()


def plural(n: int, one: str, few: str, many: str) -> str:
    n = abs(n)
    if 10 <= n % 100 <= 20:
        return many
    if n % 10 == 1:
        return one
    if 2 <= n % 10 <= 4:
        return few
    return many


def restore_case(word: str, inflected: str) -> str:
    out = list(inflected)
    for i, ch in enumerate(word):
        if i < len(out) and ch.isupper() and out[i].isalpha():
            out[i] = out[i].upper()
    return "".join(out)


def is_cyrillic(word: str) -> bool:
    return any("\u0400" <= ch <= "\u04ff" for ch in word)


def lemmatize(word: str) -> str:
    return MORPH_ANALYZER.parse(word)[0].normal_form


_PUNCTUATION_RE = re.compile(r"[^\w\s]")


def remove_punctuation(s: str) -> str:
    return _PUNCTUATION_RE.sub("", s)


_LATIN_PHRASE_RE = re.compile(
    r"[A-Za-z\u00C0-\u017F]+(?:[\s'\-/&.][A-Za-z\u00C0-\u017F]+)*"
)


def split_by_script(text: str) -> list[tuple[str, str]]:
    parts: list[tuple[str, str]] = []
    pos = 0
    for m in _LATIN_PHRASE_RE.finditer(text):
        if m.start() > pos:
            gap = text[pos : m.start()]
            parts.append(("ru" if is_cyrillic(gap) else "en", gap))
        parts.append(("en", m.group()))
        pos = m.end()
    if pos < len(text):
        tail = text[pos:]
        parts.append(("ru" if is_cyrillic(tail) else "en", tail))
    return parts


def decline(word: str, case: Case) -> str:
    if not word or not is_cyrillic(word):
        return word
    inflected = MORPH_ANALYZER.parse(word)[0].inflect({case})
    if inflected is None:
        return word
    return restore_case(word, inflected.word)


def number_to_words(n: int, case: Case) -> str:
    sign = "минус " if n < 0 else ""
    tokens = num2words.num2words(abs(n), lang="ru").split()
    return sign + " ".join(decline(token, case) for token in tokens)


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


def extract_locations(s: str) -> list[str]:
    doc = natasha.Doc(s)
    doc.segment(NATASHA_SEGMENTER)
    doc.tag_morph(NATASHA_MORPH_TAGGER)
    doc.parse_syntax(NATASHA_SYNTAX_PARSER)
    doc.tag_ner(NATASHA_NER_TAGGER)
    assert isinstance(doc.spans, typing.Iterable)
    for span in doc.spans:
        span.normalize(NATASHA_MORPH_VOCAB)
    return [span.normal for span in doc.spans if span.type == "LOC"]
