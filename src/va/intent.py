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
    VolumeUp = enum.auto()
    VolumeDown = enum.auto()
    VolumeMuchUp = enum.auto()
    VolumeMuchDown = enum.auto()


KEYWORDS = {
    Intent.Weather: ["погода", "weather"],
    Intent.PauseMusic: ["пауза", "pause"],
    Intent.PlayMusic: ["включи", "play"],
    Intent.NextTrack: ["следующий", "дальше", "next"],
    Intent.PreviousTrack: ["предыдущий", "назад", "previous"],
    Intent.VolumeUp: ["громче", "громко", "louder"],
    Intent.VolumeDown: ["тише", "тихо", "quieter"],
    Intent.VolumeMuchUp: ["погромче", "навали"],
    Intent.VolumeMuchDown: ["потише", "приглуши"],
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


def extract_data(s: str, intent: Intent) -> dict:
    match intent:
        case Intent.Weather:
            locations = nlp.extract_locations(s)
            location = locations[0] if locations else None
            return {"location": location}
    return {}
