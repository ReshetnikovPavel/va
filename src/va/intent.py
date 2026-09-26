import difflib
import enum

from va import nlp, slots


class Intent(enum.Enum):
    Unknown = enum.auto()
    Weather = enum.auto()
    Time = enum.auto()
    PauseMusic = enum.auto()
    PlayMusic = enum.auto()
    NextTrack = enum.auto()
    PreviousTrack = enum.auto()
    NowPlaying = enum.auto()
    VolumeUp = enum.auto()
    VolumeDown = enum.auto()
    VolumeMuchUp = enum.auto()
    VolumeMuchDown = enum.auto()


# Интенты в порядке приоритета (точное совпадение по леммам).
KEYWORDS = {
    Intent.PauseMusic: ["выключить", "пауза", "хватить", "стоп", "pause"],
    Intent.NextTrack: ["следующий", "далёкий", "next"],
    Intent.PreviousTrack: ["предыдущий", "назад", "previous"],
    Intent.PlayMusic: [
        "включить",
        "включать",
        "поставить",
        "постав",
        "сыграть",
        "запустить",
        "завести",
        "врубить",
        "врубать",
        "давать",
        "дать",
        "послушать",
        "слушать",
        "play",
    ],
    Intent.Weather: ["погода", "weather"],
    Intent.Time: ["время", "time", "который"],
    Intent.NowPlaying: ["играть"],
    Intent.VolumeMuchUp: ["погромче", "навали"],
    Intent.VolumeMuchDown: ["потише", "приглуши"],
    Intent.VolumeUp: ["громкий", "громко", "louder"],
    Intent.VolumeDown: ["тихий", "тихо", "quieter"],
}
SIMILARITY_THRESHOLD = 0.7


def classify(s: str) -> Intent:
    words = nlp.remove_punctuation(s).lower().split()
    lemmas = [nlp.lemmatize(word) for word in words]
    for intent, keywords in KEYWORDS.items():
        for keyword in keywords:
            if keyword in lemmas or keyword in words:
                print(intent)
                return intent
    best_intent, best_ratio = Intent.Unknown, 0.0
    for intent, keywords in KEYWORDS.items():
        for keyword in keywords:
            if lemmas:
                ratio = max(
                    difflib.SequenceMatcher(None, lemma, keyword).ratio()
                    for lemma in lemmas
                )
                if ratio > best_ratio:
                    best_intent, best_ratio = intent, ratio
    if best_ratio > SIMILARITY_THRESHOLD:
        print(best_intent)
        return best_intent
    print(Intent.Unknown)
    return Intent.Unknown
