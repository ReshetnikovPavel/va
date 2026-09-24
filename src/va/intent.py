import difflib
import enum


class Intent(enum.Enum):
    Unknown = enum.auto()
    Weather = enum.auto()
    PauseMusic = enum.auto()
    PlayMusic = enum.auto()

KEYWORDS = {
    Intent.Weather: ["погода", "weather"],
    Intent.PauseMusic: ["пауза", "pause"],
    Intent.PlayMusic: ["включи", "play"],
}
SIMILARITY_THRESHOLD = 0.7


def classify(s: str) -> Intent:
    words = s.lower().split()
    for intent, keywords in KEYWORDS.items():
        for keyword in keywords:
            max_similarity = 0
            for word in words:
                similarity = difflib.SequenceMatcher(None, word, keyword).ratio()
                max_similarity = max(max_similarity, similarity)
            print(keyword, max_similarity)
            if max_similarity > SIMILARITY_THRESHOLD:
                print(intent)
                return intent
    print(Intent.Unknown)
    return Intent.Unknown
