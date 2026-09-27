import pymorphy3
import pytest
from yargy import Parser
from yargy.morph import MorphAnalyzer as YargyMorphAnalyzer
from yargy.tokenizer import MorphTokenizer

import va.grammars.duration


@pytest.fixture(scope="module")
def parser():
    morph = YargyMorphAnalyzer(pymorphy3.MorphAnalyzer())
    return Parser(
        va.grammars.duration.DURATION,
        tokenizer=MorphTokenizer(morph=morph),
    )


def secs(parser, text, index=0):
    match = list(parser.findall(text))
    if not match:
        return None
    return match[index].fact.obj.total_seconds


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2 часа", 7200),
        ("5 часов", 18000),
        ("1 час", 3600),
        ("24 часа", 86400),
        ("на 10 часов", 36000),
        ("5 минут", 300),
        ("1 минута", 60),
        ("на 7 минут", 420),
        ("без трёх минут", 180),
        ("30 секунд", 30),
        ("1 секунда", 1),
        ("на 2 секунды", 2),
        ("2 часа 30 минут", 9000),
        ("1 час 5 минут 10 секунд", 3910),
        ("полчаса", 1800),
        ("полминуты", 30),
        ("полтора часа", 5400),
        ("час", 3600),
        ("3 ч", 10800),
        ("2 мин", 120),
    ],
)
def test_duration_digits_and_forms(parser, text, expected):
    assert secs(parser, text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("пять минут", 300),
        ("семь часов", 25200),
        ("три часа", 10800),
        ("двадцать секунд", 20),
        ("двадцать пять минут", 1500),
        ("сорок один час", 147600),
        ("девяносто девять секунд", 99),
        ("шестьдесят секунд", 60),
        ("шестьдесят пять минут", 3900),
        ("ноль секунд", 0),
        ("одна минута", 60),
        ("две минуты", 120),
        ("пятнадцать минут", 900),
        ("семьдесят пять минут", 4500),
        ("один час тридцать минут", 5400),
        ("восемьдесят восемь минут", 5280),
        ("шестьдесят", None),
        ("девяносто девять", None),
        ("минута", None),
    ],
)
def test_duration_number_words(parser, text, expected):
    assert secs(parser, text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "расскажи анекдот",
        "включи музыку",
        "сколько времени",
        "привет",
        "два два сапога",
        "три в ряд",
        "минута молчания",
        "дай пять",
        "на секунду",
        "минутку",
        "",
    ],
)
def test_no_duration(parser, text):
    assert list(parser.findall(text)) == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("120 секунд", 120),
        ("1000 секунд", 1000),
        ("999999 секунд", 999999),
        ("120 минут", 7200),
        ("1000 минут", 60000),
        ("500 часов", 1800000),
        ("1 час 30 минут 30 секунд", 5430),
        ("2 часа 10000000 секунд", 10007200),
        ("на 120 секунд", 120),
        ("1000000000 секунд", 1000000000),
    ],
)
def test_duration_unlimited_digits(parser, text, expected):
    assert secs(parser, text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "сто секунд",
        "двести минут",
        "триста секунд",
    ],
)
def test_duration_words_still_capped(parser, text):
    assert list(parser.findall(text)) == []


@pytest.mark.parametrize(
    ("digits", "words"),
    [
        ("48 минут", "сорок восемь минут"),
        ("17 часов", "семнадцать часов"),
        ("3 часа 25 минут", "три часа двадцать пять минут"),
        ("5 минут 7 секунд", "пять минут семь секунд"),
        ("90 секунд", "девяносто секунд"),
        ("2 часа", "два часа"),
    ],
)
def test_digit_word_parity(parser, digits, words):
    assert secs(parser, digits) == secs(parser, words)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("5 минут и одну секунду", 301),
        ("5 минут и 30 секунд", 330),
        ("2 часа и 45 минут", 9900),
        ("2 часа, 30 минут", 9000),
        ("5 минут, 30 секунд", 330),
        ("5 минут с 30 секундами", 330),
        ("5 минут да 30 секунд", 330),
        ("час и 5 минут", 3900),
        ("полтора часа и 10 минут", 6000),
        ("1 час 5 минут и 10 секунд", 3910),
        ("2 часа 5 минут и 30 секунд", 7530),
        ("два часа, двадцать минут", 8400),
    ],
)
def test_duration_with_joins(parser, text, expected):
    assert secs(parser, text) == expected
