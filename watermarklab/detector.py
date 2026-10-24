"""
detector.py
-----------
Statistical + ML watermark detection.

statistical_detect(text, context_window=1)   classic green-list z-test (Methods A / B)
detect_method(text, method)                  detector for ANY method (incl. the 3 new ones)
train_ml_classifier / ml_detect              small ML classifier on detector features
hybrid_detect(text, method)                  statistical + ML combined

Every statistical result is a dict with at least:
    is_watermarked, confidence, z_score, num_tokens, green_fraction, method
"""

import math
import os
import re
from typing import Dict, List, Optional

import numpy as np
from scipy.special import log_ndtr, ndtri_exp

from . import config
from .backend import get_backend
from .methods import METHODS, MethodSpec, get_method
from .scheme import WatermarkScheme


# ---------------------------------------------------------------------------
# statistics helpers
# ---------------------------------------------------------------------------
def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def confidence_from_z(z: float, threshold: float) -> float:
    """
    Decision-centred confidence in [0, 1]: exactly 50% at the z threshold, ~73% one
    z-unit above it, ~27% one unit below. (The raw one-sided p-value saturates at
    100% for any z > 4, which makes before/after-attack comparisons uninformative.)
    """
    x = z - threshold
    return float(1.0 / (1.0 + math.exp(-max(min(x, 50.0), -50.0))))


def p_value_from_z(z: float) -> float:
    return float(0.5 * math.erfc(z / math.sqrt(2.0)))


def weighted_z(hits: np.ndarray, weights: np.ndarray, p0) -> float:
    """z-score of sum_t w_t (hit_t - p0_t); valid under the null 'hit_t ~ Bernoulli(p0_t)'."""
    p0 = np.broadcast_to(np.asarray(p0, dtype=np.float64), hits.shape)
    var = float((weights ** 2 * p0 * (1 - p0)).sum())
    if var <= 0:
        return 0.0
    return float((weights * (hits - p0)).sum() / math.sqrt(var))


def _empty_result(method: str, n: int) -> Dict:
    return {"is_watermarked": False, "confidence": 0.0, "z_score": 0.0, "p_value": 1.0, "num_tokens": n,
            "green_fraction": 0.0, "method": method, "note": "text too short to test"}


# ---------------------------------------------------------------------------
# universal detector
# ---------------------------------------------------------------------------
def detect_method(text: str, method="basic", threshold: Optional[float] = None,
                  key: Optional[str] = None) -> Dict:
    spec = method if isinstance(method, MethodSpec) else get_method(method)
    backend = get_backend()
    scheme = WatermarkScheme(spec, backend, key)
    thr = config.Z_SCORE_THRESHOLD if threshold is None else threshold

    ids = backend.encode(text)
    if len(ids) < 5:
        return _empty_result(spec.key, len(ids))

    rows = scheme.score_text_ids(ids)
    weights = scheme.detector_weights(ids) if spec.adaptive else np.ones(len(ids))

    if spec.ecc:
        res = _detect_ecc(scheme, rows, weights, thr)
    else:
        res = _detect_greenlist(spec, rows, weights, thr)
    res.update({"method": spec.key, "num_tokens": len(ids),
                "scored_tokens": int(sum(1 for r in rows if not r[2]))})
    return res


def _anchor_layer_z(spec: MethodSpec, rows, weights: np.ndarray):
    """z-score of the context-free anchor-word layer (first occurrence of each anchor cluster only)."""
    idx = [i for i, r in enumerate(rows) if r[3] is not None]
    if len(idx) < 3:
        return 0.0, len(idx)
    hits = np.array([1.0 if rows[i][3] < spec.anchor_gamma else 0.0 for i in idx])
    return weighted_z(hits, weights[idx], spec.anchor_gamma), len(idx)


def _combine_layers(z_context: float, z_anchor: float, n_anchor: int):
    """
    Recovery statistic. Both z-scores are ~N(0,1) under 'no watermark' and use
    independent hash domains, so Stouffer's combination (z1+z2)/sqrt(2) is valid.
    We report the best of {context layer, anchor layer, combined}; the decision
    threshold (z >= 4) is so strict that the 3-way max costs almost no false-positive budget.
    """
    if n_anchor < 3:
        return z_context, False
    z_comb = (z_context + z_anchor) / math.sqrt(2.0)
    z_best = max(z_context, z_anchor, z_comb)
    return z_best, z_best > z_context + 1e-9


def _detect_greenlist(spec: MethodSpec, rows, weights: np.ndarray, thr: float) -> Dict:
    gamma = spec.gamma
    keep = np.array([not r[2] for r in rows])
    u0 = np.array([r[0] for r in rows])[keep]
    hits0 = (u0 < gamma).astype(np.float64)
    z_context = weighted_z(hits0, weights[keep], gamma)
    out = {
        "green_fraction": float(hits0.mean()) if len(hits0) else 0.0,
        "z_primary": z_context,
    }
    z = z_context
    if spec.anchor_layer:
        z_anchor, n_anchor = _anchor_layer_z(spec, rows, weights)
        z, recovered = _combine_layers(z_context, z_anchor, n_anchor)
        out.update({"z_context_layer": z_context, "z_anchor_layer": z_anchor,
                    "anchors_scored": n_anchor, "recovery_used": bool(recovered),
                    "z_recovery": float(z)})
    out.update({"z_score": float(z), "confidence": confidence_from_z(z, thr),
                "p_value": p_value_from_z(z), "is_watermarked": bool(z >= thr)})
    return out


def _detect_ecc(scheme: WatermarkScheme, rows, weights: np.ndarray, thr: float) -> Dict:
    spec, code, n = scheme.spec, scheme.code, scheme.n_slots
    votes = np.zeros((n, 2))
    token_votes = []
    for (u, slot, dup, _), wt in zip(rows, weights):
        bit = 1 if u >= 0.5 else 0
        token_votes.append((slot, bit, wt, dup))
        if not dup:
            votes[slot, bit] += wt
    # soft-decision maximum-likelihood decode over every valid payload
    dec = code.decode_ml(votes[:, 1] - votes[:, 0])
    codeword = code.encode(dec["payload"])
    # second opinion: classic hard-decision Hamming decode + CRC on the same votes
    hard_in = [None if votes[i, 0] == votes[i, 1] else int(votes[i, 1] > votes[i, 0]) for i in range(n)]
    hard = code.decode(hard_in)

    agree = np.array([1.0 if bit == codeword[slot] else 0.0 for slot, bit, _, dup in token_votes if not dup])
    w = np.array([wt for slot, bit, wt, dup in token_votes if not dup])
    z_raw = weighted_z(agree, w, 0.5)
    # Multiple-testing (Bonferroni) correction: the decoder chose the best of 2^k payloads,
    # so the evidence has to beat what the best of 2^k random payloads would show.
    logp = float(log_ndtr(-z_raw)) + config.ECC_PAYLOAD_BITS * math.log(2)
    if logp >= 0:
        z_payload = 0.0                        # no evidence of any payload
    else:
        z_payload = -float(ndtri_exp(logp))
    z, extra = z_payload, {}
    if spec.anchor_layer:                       # anchors add independent evidence of the watermark
        z_anchor, n_anchor = _anchor_layer_z(spec, rows, weights)
        z, recovered = _combine_layers(z_payload, z_anchor, n_anchor)
        extra = {"z_payload_layer": z_payload, "z_anchor_layer": z_anchor,
                 "anchors_scored": n_anchor, "recovery_used": bool(recovered)}
    return {
        "z_score": float(z), "confidence": confidence_from_z(z, thr), "p_value": p_value_from_z(z),
        "is_watermarked": bool(z >= thr),
        "green_fraction": float(agree.mean()) if len(agree) else 0.0,
        "payload": dec["payload"],
        "crc_ok": bool(hard["crc_ok"] and hard["payload"] == dec["payload"]),
        "corrected_bits": dec["corrected"], "erased_bits": dec["erasures"],
        "bit_agreement": float(agree.mean()) if len(agree) else 0.0,
        "slot_votes": votes.tolist(), **extra,
    }


def statistical_detect(text: str, context_window: int = 1, threshold: Optional[float] = None) -> Dict:
    """Classic green-list z-test (the original Method A / B detector)."""
    spec = METHODS["basic"].with_(window=context_window, key=f"window{context_window}")
    return detect_method(text, spec, threshold)


def detect_all(text: str) -> Dict[str, Dict]:
    """Run every method's detector on a text (used by the dashboard's 'which watermark?' view)."""
    return {k: detect_method(text, k) for k in METHODS}


# ---------------------------------------------------------------------------
# ML detector
# ---------------------------------------------------------------------------
FEATURE_NAMES = ["z_window1", "z_window4", "z_anchor", "green_w1", "green_w4",
                 "type_token_ratio", "mean_word_len", "repeat_bigram_rate", "log_tokens"]
_ML_CACHE = {}


def ml_features(text: str) -> np.ndarray:
    r1 = detect_method(text, "basic")
    r4 = detect_method(text, "context")
    ra = detect_method(text, "anchor")
    words = re.findall(r"[A-Za-z']+", text.lower())
    n = max(len(words), 1)
    bigrams = list(zip(words, words[1:]))
    rep = 1.0 - len(set(bigrams)) / max(len(bigrams), 1)
    return np.array([
        r1["z_score"], r4["z_score"], ra.get("z_primary", ra["z_score"]),
        r1["green_fraction"], r4["green_fraction"],
        len(set(words)) / n, float(np.mean([len(w) for w in words])) if words else 0.0,
        rep, math.log(n + 1),
    ])


def train_ml_classifier(watermarked_texts: List[str], plain_texts: List[str], path: Optional[str] = None):
    from joblib import dump
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = np.array([ml_features(t) for t in list(watermarked_texts) + list(plain_texts)])
    y = np.array([1] * len(watermarked_texts) + [0] * len(plain_texts))
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    pipe.fit(X, y)
    path = path or config.ML_CLASSIFIER_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    dump({"pipeline": pipe, "features": FEATURE_NAMES}, path)
    _ML_CACHE.pop(path, None)
    return pipe


def _load_ml(path: str):
    if path not in _ML_CACHE:
        if not os.path.exists(path):
            return None
        from joblib import load
        _ML_CACHE[path] = load(path)["pipeline"]
    return _ML_CACHE[path]


def ml_detect(text: str, path: Optional[str] = None) -> Optional[Dict]:
    pipe = _load_ml(path or config.ML_CLASSIFIER_PATH)
    if pipe is None:
        return None
    p = float(pipe.predict_proba(ml_features(text).reshape(1, -1))[0, 1])
    return {"is_watermarked": p >= 0.5, "confidence": p, "method": "ml"}


def hybrid_detect(text: str, method="basic") -> Dict:
    """Statistical test + ML classifier. The statistical test stays the primary evidence."""
    stat = detect_method(text, method)
    ml = ml_detect(text)
    out = dict(stat)
    out["stat_confidence"] = stat["confidence"]
    if ml is not None:
        out["ml_probability"] = ml["confidence"]
        out["is_watermarked"] = bool(stat["is_watermarked"] or
                                     (ml["confidence"] >= 0.9 and stat["z_score"] >= config.Z_SCORE_THRESHOLD / 2))
        out["confidence"] = max(stat["confidence"], ml["confidence"])
    return out
