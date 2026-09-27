from yargy import Parser
from yargy.morph import MorphAnalyzer as YargyMorphAnalyzer
from yargy.tokenizer import MorphTokenizer

import va.grammars.duration
from va import models


def extract_durations(s: str) -> list[int]:
    morph = YargyMorphAnalyzer(models.MORPH_ANALYZER)
    duration_parser = Parser(
        va.grammars.duration.DURATION,
        tokenizer=MorphTokenizer(morph=morph),
    )
    matches = list(duration_parser.findall(s))
    res = [match.fact.obj.total_seconds for match in matches]
    return res
