import typing

import natasha

from va import models


def extract_locations(s: str) -> list[str]:
    doc = natasha.Doc(s)
    doc.segment(models.NATASHA_SEGMENTER)
    doc.tag_morph(models.NATASHA_MORPH_TAGGER)
    doc.parse_syntax(models.NATASHA_SYNTAX_PARSER)
    doc.tag_ner(models.NATASHA_NER_TAGGER)
    assert isinstance(doc.spans, typing.Iterable)
    for span in doc.spans:
        span.normalize(models.NATASHA_MORPH_VOCAB)
    return [span.normal for span in doc.spans if span.type == "LOC"]
