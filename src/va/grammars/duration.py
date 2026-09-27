from yargy import or_, rule
from yargy.interpretation import fact
from yargy.predicates import (
    caseless,
    eq,
    gte,
    normalized,
)


class DurationValue:
    __slots__ = ("hours", "minutes", "seconds")

    def __init__(self, hours, minutes, seconds):
        self.hours = hours
        self.minutes = minutes
        self.seconds = seconds

    @property
    def total_seconds(self):
        return (self.hours or 0) * 3600 + (self.minutes or 0) * 60 + (self.seconds or 0)

    def __repr__(self):
        return "Duration(%ss)" % self.total_seconds


Duration = fact("Duration", ["hours", "minutes", "seconds"])


class Duration(Duration):
    @property
    def obj(self):
        return DurationValue(self.hours, self.minutes, self.seconds)


# Русские числительные 0-99 словами.
# Ключи — нормальные формы: yargy матчит токен по `normalized()` через pymorphy,
# поэтому падеж/род («трёх», «две», «одна») схлопывается в именительный мужской
# («три», «два», «один»).
_UNITS = {
    "один": 1,
    "два": 2,
    "три": 3,
    "четыре": 4,
    "пять": 5,
    "шесть": 6,
    "семь": 7,
    "восемь": 8,
    "девять": 9,
}

_TEENS = {
    "десять": 10,
    "одиннадцать": 11,
    "двенадцать": 12,
    "тринадцать": 13,
    "четырнадцать": 14,
    "пятнадцать": 15,
    "шестнадцать": 16,
    "семнадцать": 17,
    "восемнадцать": 18,
    "девятнадцать": 19,
}

_TENS = {
    "двадцать": 20,
    "тридцать": 30,
    "сорок": 40,
    "пятьдесят": 50,
    "шестьдесят": 60,
    "семьдесят": 70,
    "восемьдесят": 80,
    "девяносто": 90,
}

_NUMBERS: dict[str, int] = {
    "ноль": 0,
    **_UNITS,
    **_TEENS,
    **_TENS,
}
for tens_word, tens_value in _TENS.items():
    for unit_word, unit_value in _UNITS.items():
        if unit_value:
            _NUMBERS[f"{tens_word} {unit_word}"] = tens_value + unit_value


def _quantity(attribute):
    """Число цифрой (без ограничения) или словами (словарь 0–99)."""
    digits = gte(0).interpretation(attribute.custom(int))
    words = or_(
        *(
            rule(*(normalized(word) for word in phrase.split())).interpretation(
                attribute.custom(lambda _, value=value: value)
            )
            for phrase, value in _NUMBERS.items()
        )
    )
    return or_(digits, words)


HOUR = _quantity(Duration.hours)

MINUTE = _quantity(Duration.minutes)

SECOND = _quantity(Duration.seconds)

DOT = eq(".")

HOUR_WORD = or_(rule(normalized("час")), rule(caseless("ч"), DOT.optional()))

MINUTE_WORD = or_(rule(normalized("минута")), rule(caseless("мин"), DOT.optional()))

SECOND_WORD = or_(rule(normalized("секунда")), rule(caseless("сек"), DOT.optional()))

# "час" без числа — один час
AN_HOUR = rule(HOUR_WORD).interpretation(Duration.hours.custom(lambda _: 1))

# "полчаса" — 30 минут, "полминуты" — 30 секунд
HALF_HOUR = rule(normalized("полчаса")).interpretation(
    Duration.minutes.custom(lambda _: 30)
)

HALF_MINUTE = rule(normalized("полминуты")).interpretation(
    Duration.seconds.custom(lambda _: 30)
)

# "полтора часа" — 90 минут
ONE_AND_A_HALF_HOURS = rule(normalized("полтора"), normalized("час")).interpretation(
    Duration.hours.custom(lambda _: 1.5)
)

# Разделитель между компонентами: союз/запятая. Опционален, поэтому
# «5 минут 30 секунд» (без разделителя) матчится как раньше.
JOIN = or_(
    rule(eq(",")),
    rule(normalized("и")),
    rule(normalized("с")),
    rule(normalized("да")),
).optional()

HOUR_SLOT = or_(rule(HOUR, HOUR_WORD), AN_HOUR, ONE_AND_A_HALF_HOURS)

MINUTE_SLOT = rule(MINUTE, MINUTE_WORD)

SECOND_SLOT = rule(SECOND, SECOND_WORD)

DURATION = or_(
    HALF_HOUR,
    HALF_MINUTE,
    HOUR_SLOT,
    MINUTE_SLOT,
    SECOND_SLOT,
    rule(HOUR_SLOT, JOIN, MINUTE_SLOT),
    rule(HOUR_SLOT, JOIN, SECOND_SLOT),
    rule(MINUTE_SLOT, JOIN, SECOND_SLOT),
    rule(HOUR_SLOT, JOIN, MINUTE_SLOT, JOIN, SECOND_SLOT),
).interpretation(Duration)