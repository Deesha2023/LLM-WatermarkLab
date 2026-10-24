"""
utils.py
--------
Text-quality scoring.

text_quality_score(original, attacked)
    How close the attacked text still is to the original. Word-overlap based by
    default (no downloads); uses sentence-transformers embeddings when
    use_embeddings=True and the package is installed. This guards against a
    watermark "surviving" only because the attack destroyed the text.

relative_perplexity(plain_text, watermarked_text)
    Generation-quality check: perplexity of the watermarked text under the base
    model, next to the perplexity of un-watermarked text.
"""

import re
from collections import Counter

import numpy as np

_WORD = re.compile(r"[A-Za-z0-9']+")


def _f1(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    overlap = sum((a & b).values())
    if overlap == 0:
        return 0.0
    p, r = overlap / sum(b.values()), overlap / sum(a.values())
    return 2 * p * r / (p + r)


def text_quality_score(original: str, attacked: str, use_embeddings: bool = False) -> float:
    """Similarity in [0, 1] between the original and attacked text."""
    if use_embeddings:
        try:
            from sentence_transformers import SentenceTransformer, util
            model = SentenceTransformer("all-MiniLM-L6-v2")
            e = model.encode([original, attacked], convert_to_tensor=True)
            return float(max(0.0, util.cos_sim(e[0], e[1]).item()))
        except Exception:
            pass
    wa = [w.lower() for w in _WORD.findall(original)]
    wb = [w.lower() for w in _WORD.findall(attacked)]
    uni = _f1(Counter(wa), Counter(wb))
    bi = _f1(Counter(zip(wa, wa[1:])), Counter(zip(wb, wb[1:])))
    return float(0.6 * uni + 0.4 * bi)


def perplexity(text: str) -> float:
    from .backend import get_backend
    return get_backend().perplexity(text)


def relative_perplexity(plain_text: str, watermarked_text: str) -> float:
    """> 1 means the watermarked text is less fluent than plain text."""
    p, w = perplexity(plain_text), perplexity(watermarked_text)
    return float(w / p) if p and not np.isnan(p) else float("nan")
