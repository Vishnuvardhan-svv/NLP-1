"""PhraseMatcher based extraction using a local curated skill vocabulary."""

from functools import lru_cache
import spacy
from spacy.matcher import PhraseMatcher

from ml.skill_analysis.skill_normalization import alias_map, load_taxonomy

@lru_cache(maxsize=1)
def _matcher():
    taxonomy = load_taxonomy()
    nlp = spacy.blank("en")
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    labels: dict[str, str] = {}
    for group in taxonomy.values():
        for canonical, aliases in group.items():
            for phrase in {canonical, *aliases}:
                if phrase.strip():
                    key = f"SKILL::{canonical}"
                    labels[key] = canonical
                    matcher.add(key, [nlp.make_doc(phrase)])
    return nlp, matcher, labels

def extract_skills(text: str) -> list[str]:
    """Return canonical skills found in text, in first-appearance order."""
    if not text or not text.strip():
        return []
    nlp, matcher, labels = _matcher()
    doc = nlp(text)
    matches = matcher(doc)
    found = sorted(((start, end, labels[nlp.vocab.strings[match_id]]) for match_id, start, end in matches), key=lambda item: (item[0], -(item[1] - item[0])))
    result: list[str] = []
    occupied_until = -1
    for start, end, canonical in found:
        if start < occupied_until:
            continue
        if canonical not in result:
            result.append(canonical)
        occupied_until = end
    return result
