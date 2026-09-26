import re
from typing import Literal

import num2words
import pymorphy3.analyzer

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


_JAPANESE_RE = re.compile(
    r"[\u3040-\u309F\u30A0-\u30FF\u31F0-\u31FF\u4E00-\u9FFF]"
)
_LATIN_RE = re.compile(r"[A-Za-z\u00C0-\u017F]")
_WORD_RE = re.compile(r"[^\W\d_]+(?:['\-][^\W\d_]+)*")


def _word_lang(word: str) -> str:
    if _JAPANESE_RE.search(word):
        return "ja"
    if is_cyrillic(word):
        return "ru"
    return "en"


def split_by_script(text: str) -> list[tuple[str, str]]:
    parts: list[tuple[str, str]] = []
    pos = 0
    for m in _WORD_RE.finditer(text):
        lang = _word_lang(m.group())
        chunk = text[pos : m.end()]
        if parts and parts[-1][0] == lang:
            parts[-1] = (lang, parts[-1][1] + chunk)
        else:
            parts.append((lang, chunk))
        pos = m.end()
    if pos < len(text):
        tail = text[pos:]
        lang = parts[-1][0] if parts else "en"
        if parts and parts[-1][0] == lang:
            parts[-1] = (lang, parts[-1][1] + tail)
        else:
            parts.append((lang, tail))
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
