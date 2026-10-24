"""
lexicon.py
----------
Word-level helpers for the Semantic-Anchor watermark.

  * normalise(word)         -> lowercase letters only
  * stem(word)              -> very light suffix stripping (running -> run)
  * canonical_label(word)   -> synonym-cluster label (e.g. "big", "large",
                               "huge" all share one label)
  * is_anchor(word)         -> True for meaning-bearing content words

IMPORTANT (honest-evaluation note): the defender's cluster table below is
deliberately *smaller* than the attacker's synonym table in attacks.py, so the
benchmark does not hand the watermark an unfair advantage. Words that the
defender does not know simply keep their own label.
"""

import re
from typing import Dict, List

STOPWORDS = set("""
a about above after again against all also am an and any are as at be because been
before being below between both but by can could did do does doing down during each
few for from further had has have having he her here hers him his how i if in into
is it its just me more most my no nor not now of off on once only or other our out
over own same she should so some such than that the their them then there these they
this those through to too under until up very was we were what when where which while
who whom why will with would you your yours
""".split())

# Curated synonym clusters used by the DEFENDER (detector + generator).
_CLUSTERS: List[List[str]] = [
    ["big", "large", "huge", "enormous", "massive"],
    ["small", "little", "tiny", "compact"],
    ["fast", "quick", "rapid", "speedy", "swift"],
    ["slow", "gradual", "sluggish"],
    ["good", "great", "excellent", "fine"],
    ["bad", "poor", "terrible", "awful"],
    ["important", "crucial", "vital", "essential", "key"],
    ["help", "assist", "aid", "support"],
    ["use", "utilize", "employ", "apply"],
    ["show", "display", "demonstrate", "reveal"],
    ["make", "create", "build", "produce", "generate"],
    ["begin", "start", "commence", "initiate"],
    ["end", "finish", "conclude", "complete"],
    ["buy", "purchase", "acquire"],
    ["tell", "inform", "notify"],
    ["think", "believe", "consider", "suppose"],
    ["need", "require"],
    ["get", "obtain", "receive", "gain"],
    ["keep", "retain", "preserve", "maintain"],
    ["change", "alter", "modify", "transform"],
    ["improve", "enhance", "boost", "upgrade"],
    ["increase", "grow", "rise", "expand"],
    ["decrease", "reduce", "lower", "shrink"],
    ["problem", "issue", "difficulty", "challenge"],
    ["answer", "response", "reply"],
    ["idea", "concept", "notion"],
    ["method", "technique", "approach", "strategy"],
    ["result", "outcome", "consequence"],
    ["people", "persons", "individuals", "humans"],
    ["computer", "machine", "device"],
    ["intelligence", "cognition"],
    ["smart", "clever", "intelligent", "bright"],
    ["learn", "study", "master"],
    ["teach", "instruct", "educate", "train"],
    ["understand", "grasp", "comprehend"],
    ["simple", "easy", "basic", "plain"],
    ["hard", "difficult", "tough", "complex"],
    ["many", "numerous", "several", "multiple"],
    ["begin", "launch"],
    ["quickly", "rapidly", "swiftly"],
    ["often", "frequently", "regularly"],
    ["main", "primary", "principal", "chief"],
    ["strong", "powerful", "robust", "sturdy"],
    ["weather", "climate"],
    ["city", "town", "metropolis"],
    ["house", "home", "residence", "dwelling"],
    ["road", "street", "route", "path"],
    ["food", "meal", "cuisine"],
    ["water", "liquid"],
    ["data", "information", "facts"],
    ["system", "framework", "platform"],
    ["text", "passage", "writing"],
    ["model", "network"],
    ["shows", "displays"],
    ["happy", "glad", "joyful", "cheerful"],
    ["sad", "unhappy", "gloomy"],
    ["beautiful", "pretty", "lovely", "attractive"],
    ["dangerous", "risky", "hazardous", "unsafe"],
    ["clear", "obvious", "evident", "apparent"],
    ["buy", "shop"],
    ["travel", "journey", "trip"],
    ["job", "career", "occupation", "profession"],
    ["money", "cash", "funds"],
    ["friend", "companion", "buddy"],
    ["child", "kid", "youngster"],
    ["doctor", "physician"],
    ["car", "automobile", "vehicle"],
    ["picture", "image", "photo"],
    ["start", "origin"],
]

_WORD_TO_LABEL: Dict[str, str] = {}
for _group in _CLUSTERS:
    _label = _group[0]
    for _w in _group:
        _WORD_TO_LABEL.setdefault(_w, _label)

_NON_ALPHA = re.compile(r"[^a-z]")


def normalise(word: str) -> str:
    return _NON_ALPHA.sub("", word.lower())


def stem(word: str) -> str:
    """Light, dependency-free stemmer: enough to merge run/runs/running."""
    w = word
    if len(w) > 5 and w.endswith("ing"):
        w = w[:-3]
        if len(w) > 2 and w[-1] == w[-2]:
            w = w[:-1]
    elif len(w) > 4 and w.endswith("ies"):
        w = w[:-3] + "y"
    elif len(w) > 4 and w.endswith("ed"):
        w = w[:-2]
        if len(w) > 2 and w[-1] == w[-2]:
            w = w[:-1]
    elif len(w) > 4 and w.endswith("es"):
        w = w[:-2]
    elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    elif len(w) > 5 and w.endswith("ly"):
        w = w[:-2]
    return w


def canonical_label(word: str) -> str:
    """Synonym-cluster label of a word ('' for non-words such as punctuation)."""
    w = normalise(word)
    if not w:
        return ""
    if w in _WORD_TO_LABEL:
        return _WORD_TO_LABEL[w]
    s = stem(w)
    if s in _WORD_TO_LABEL:
        return _WORD_TO_LABEL[s]
    return s


def is_anchor(word: str, min_len: int = 4) -> bool:
    """Anchors are meaning-bearing content words that paraphrasers usually keep."""
    w = normalise(word)
    return len(w) >= min_len and w not in STOPWORDS
