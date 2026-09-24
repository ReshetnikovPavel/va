from typing import Literal

from num2words import num2words as _num2words
from pymorphy3.analyzer import MorphAnalyzer

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

MORPH_ANALYZER = MorphAnalyzer()


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


def decline(word: str, case: Case) -> str:
    if not word or not is_cyrillic(word):
        return word
    inflected = MORPH_ANALYZER.parse(word)[0].inflect({case})
    if inflected is None:
        return word
    return restore_case(word, inflected.word)


def number_to_words(n: int, case: Case) -> str:
    sign = "минус " if n < 0 else ""
    tokens = _num2words(abs(n), lang="ru").split()
    return sign + " ".join(decline(token, case) for token in tokens)
