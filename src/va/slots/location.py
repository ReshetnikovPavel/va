import importlib.metadata
import typing

import natasha
from pymorphy2 import analyzer as _pymorphy2_analyzer

# natasha used pkg_resources which is deprecated. Monkey-patch
_pymorphy2_analyzer._iter_entry_points = lambda *groups, **_: (  # ty: ignore[invalid-assignment]
    importlib.metadata.entry_points(group=groups[0])
)


NATASHA_SEGMENTER = natasha.Segmenter()
NATASHA_MORPH_VOCAB = natasha.MorphVocab()
NATASHA_EMBEDDING = natasha.NewsEmbedding()
NATASHA_MORPH_TAGGER = natasha.NewsMorphTagger(NATASHA_EMBEDDING)
NATASHA_SYNTAX_PARSER = natasha.NewsSyntaxParser(NATASHA_EMBEDDING)
NATASHA_NER_TAGGER = natasha.NewsNERTagger(NATASHA_EMBEDDING)


def extract_locations(s: str) -> list[str]:
    doc = natasha.Doc(s)
    doc.segment(NATASHA_SEGMENTER)
    doc.tag_morph(NATASHA_MORPH_TAGGER)
    doc.parse_syntax(NATASHA_SYNTAX_PARSER)
    doc.tag_ner(NATASHA_NER_TAGGER)
    assert isinstance(doc.spans, typing.Iterable)
    for span in doc.spans:
        span.normalize(NATASHA_MORPH_VOCAB)
    return [span.normal for span in doc.spans if span.type == "LOC"]
