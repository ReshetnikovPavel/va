import difflib
import enum

from va import nlp


class Intent(enum.Enum):
    Unknown = enum.auto()
    Weather = enum.auto()
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


PLAY_TRIGGERS = KEYWORDS[Intent.PlayMusic]


def extract_data(s: str, intent: Intent) -> dict:
    match intent:
        case Intent.Weather:
            locations = nlp.extract_locations(s)
            location = locations[0] if locations else None
            return {"location": location}
        case Intent.PlayMusic:
            words = nlp.remove_punctuation(s).lower().split()
            lemmas = [nlp.lemmatize(word) for word in words]
            index = next(
                (
                    i
                    for i, (word, lemma) in enumerate(zip(words, lemmas))
                    if word in PLAY_TRIGGERS or lemma in PLAY_TRIGGERS
                ),
                None,
            )
            if index is not None:
                words = words[index + 1 :]
                lemmas = lemmas[index + 1 :]
            query = " ".join(
                word
                for word, lemma in zip(words, lemmas)
                if word not in PLAY_STOP_RAW
                and lemma not in PLAY_STOP_LEMMAS
                and word not in PLAY_STOP_LEMMAS
            )
            return {"song": query or None}
    return {}


PLAY_STOP_LEMMAS = {
    "музыка",
    "песня",
    "песенка",
    "трек",
    "альбом",
    "группа",
    "композиция",
    "пожалуйста",
    "спасибо",
    "пж",
    "плиз",
    "please",
    "я",
    "мы",
    "мой",
    "один",
    "такой",
    "какой",
    "любой",
    "какой-нибудь",
    "снова",
    "потом",
    "сейчас",
    "хотеть",
    "слушать",
    "послушать",
    "выключить",
    "стоп",
    "пауза",
    "хватить",
    "следующий",
    "предыдущий",
    "далёкий",
    "и",
}

# Слова с дефисами, которые remove_punctuation режет до неузнаваемости.
PLAY_STOP_RAW = {"чтонибудь", "чтото", "какоенибудь", "какиенибудь"}
