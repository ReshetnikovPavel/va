import re
import typing
from typing import Literal

from num2words import num2words

from va import models

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
    return models.MORPH_ANALYZER.parse(word)[0].normal_form


_PUNCTUATION_RE = re.compile(r"[^\w\s]")


def remove_punctuation(s: str) -> str:
    return _PUNCTUATION_RE.sub("", s)


_JAPANESE_RE = re.compile(r"[\u3040-\u309F\u30A0-\u30FF\u31F0-\u31FF\u4E00-\u9FFF]")
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


def decline(word: str, case: Case, plur: bool = False) -> str:
    if not word or not is_cyrillic(word):
        return word
    inflect = {case}
    if plur:
        inflect.add("plur")
    inflected = models.MORPH_ANALYZER.parse(word)[0].inflect(inflect)
    if inflected is None:
        return word
    return restore_case(word, inflected.word)


def number_to_words(n: int, case: Case) -> str:
    sign = "минус " if n < 0 else ""
    tokens = num2words(abs(n), lang="ru").split()
    return sign + " ".join(decline(token, case) for token in tokens)


def secs_to_words(secs: int, case: Case, digits: bool = False) -> str:
    assert secs >= 0

    hours = secs // 3600
    minutes = (secs - hours * 3600) // 60
    seconds = secs - hours * 3600 - minutes * 60
    return time_to_words(hours, minutes, seconds, case, digits=digits)


_NUM2WORDS_CASE = {
    "nomn": "n",
    "gent": "g",
    "datv": "d",
    "accs": "a",
    "ablt": "i",
    "loct": "p",
}


def time_to_words(
    hours: int, minutes: int, seconds: int, case: Case, digits: bool = False
) -> str:
    n2w_case = _NUM2WORDS_CASE[case]
    parts = [(hours, "час", "m"), (minutes, "минута", "f"), (seconds, "секунда", "f")]
    parts = [p for p in parts if p[0]] or [(0, "секунда", "f")]

    res = []
    for n, unit, gender in parts:
        number = (
            str(n)
            if digits
            else num2words(n, lang="ru", case=n2w_case, gender=gender, animate=False)
        )
        res.append(number)

        form = models.MORPH_ANALYZER.parse(unit)[0]
        last_digit = n % 10
        two_last_digits = n % 100
        if last_digit == 1 and two_last_digits != 11:
            grammemes = {case}
        elif case in ("nomn", "accs"):
            if 2 <= last_digit <= 4 and two_last_digits not in (12, 13, 14):
                grammemes = {"gent"}
            else:
                grammemes = {"gent", "plur"}
        else:
            grammemes = {case, "plur"}
        res.append((form.inflect(typing.cast(set[str], grammemes)) or form).word)
    return " ".join(res)
