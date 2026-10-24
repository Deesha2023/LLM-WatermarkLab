"""
scheme.py
---------
The shared mathematics of every watermark method.

For each position t of the text, the scheme derives a *context key* c_t from
the already-written text. A candidate token v then gets a pseudo-random number
u = H(c_t, key(v)) in [0, 1):

    plain watermark :  v is GREEN  if  u < gamma
    payload (ECC)   :  slot i = H(c_t) mod n ;  payload bit b = codeword[i] ;
                       v is in the TARGET set if  floor(2u) == b

The generator adds a bonus `delta_t` to the logits of target tokens; the
detector recomputes the very same keys from the text alone and counts how many
tokens landed in their target set.

What the context key is built from depends on the method:
  * window mode : the previous `window` tokens (classic KGW-style)
  * anchor mode : the most recent *anchor* word (a meaning-bearing content word),
                  and tokens are scored by the synonym-cluster of the word.
                  A second, context-FREE layer marks some anchor-word clusters as
                  green; it keeps working when edits break the context seeds.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import config, lexicon
from .backend import LMBackend, entropy_nats, get_backend, softmax
from .ecc import ErrorCorrectingCode
from .hashing import (combine, key_to_int, splitmix64, string_to_key, unit_interval,
                      unit_interval_scalar)
from .methods import MethodSpec

_SENTINEL = 0x5EEDFACE
_SLOT_SALT = 0xECC0DE
_ANCHOR_LAYER_SALT = 0xA11C40

_VOCAB_CACHE: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}


class WatermarkScheme:
    def __init__(self, spec: MethodSpec, backend: Optional[LMBackend] = None,
                 key: Optional[str] = None):
        self.spec = spec
        self.backend = backend or get_backend()
        self.key_int = key_to_int(key or config.SECRET_KEY)
        self.code: Optional[ErrorCorrectingCode] = None
        if spec.ecc:
            self.code = ErrorCorrectingCode(config.ECC_PAYLOAD_BITS, config.ECC_CRC_BITS)
        self._info_cache: Dict[int, Tuple[int, bool]] = {}
        self._score_keys: Optional[np.ndarray] = None
        self._anchor_mask: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ keys
    @property
    def n_slots(self) -> int:
        return self.code.codeword_len if self.code else 0

    def vocab_score_keys(self) -> np.ndarray:
        """uint64 key per vocabulary entry that decides greenness (token key or synonym-cluster key)."""
        if self._score_keys is None:
            if self.spec.seed_mode == "window":
                self._score_keys = self.backend.token_keys()
            else:
                ck = (id(self.backend), self.backend.vocab_size)
                if ck not in _VOCAB_CACHE:
                    strings = self.backend.vocab_strings()
                    keys = np.empty(len(strings), dtype=np.uint64)
                    anchors = np.zeros(len(strings), dtype=bool)
                    for i, s in enumerate(strings):
                        k, a = self._info_from_string(s)
                        keys[i], anchors[i] = k, a
                    _VOCAB_CACHE[ck] = (keys, anchors)
                self._score_keys, self._anchor_mask = _VOCAB_CACHE[ck]
        return self._score_keys

    def vocab_anchor_mask(self) -> np.ndarray:
        self.vocab_score_keys()
        return self._anchor_mask

    # ---- context-free anchor layer ----------------------------------------
    @property
    def anchor_layer_key(self) -> int:
        return combine(self.key_int, _ANCHOR_LAYER_SALT)

    def anchor_layer_mask(self) -> np.ndarray:
        """Boolean mask over the vocabulary: anchor words whose synonym cluster is green (layer 2)."""
        if not hasattr(self, "_anchor_green"):
            u = unit_interval(self.anchor_layer_key, self.vocab_score_keys())
            self._anchor_green = self.vocab_anchor_mask() & (u < self.spec.anchor_gamma)
        return self._anchor_green

    def anchor_layer_u(self, skey: int) -> float:
        return unit_interval_scalar(self.anchor_layer_key, skey)

    @staticmethod
    def _info_from_string(s: str) -> Tuple[int, bool]:
        """(canonical score key, is_anchor) of a token's surface string."""
        label = lexicon.canonical_label(s)
        word_start = s.startswith(" ") or not s[:1].isalpha()
        anchor = word_start and lexicon.is_anchor(s, config.ANCHOR_MIN_LEN)
        if not label:
            label = "#" + s.strip()          # punctuation / odd tokens keep their own identity
        elif not word_start:
            label = "##" + label             # mid-word fragments are not merged with whole words
        return string_to_key(label), anchor

    def token_info(self, tid: int) -> Tuple[int, bool]:
        """(score key, is_anchor) of one token id, as seen by the detector."""
        if self.spec.seed_mode == "window":
            return self.backend.key_of(tid), False
        hit = self._info_cache.get(tid)
        if hit is None:
            hit = self._info_from_string(self.backend.token_str(tid))
            self._info_cache[tid] = hit
        return hit

    # --------------------------------------------------------- context keys
    def context_key(self, prev_ids: List[int], anchors: List[int]) -> int:
        """Context key for the NEXT token. `anchors` are the score keys of the anchor words written so far."""
        if self.spec.seed_mode == "window":
            w = self.spec.window
            last = [self.backend.key_of(t) for t in prev_ids[-w:]]
            last = [_SENTINEL] * (w - len(last)) + last
            return combine(self.key_int, w, *last)
        a = anchors[-1] if anchors else _SENTINEL
        return combine(self.key_int, 7, a)

    def slot_of(self, ctx_key: int) -> int:
        return splitmix64(ctx_key ^ _SLOT_SALT) % self.n_slots

    # --------------------------------------------------------- generation side
    def target_mask(self, ctx_key: int, codeword: Optional[List[int]] = None) -> np.ndarray:
        """Boolean mask over the vocabulary: which tokens get the bonus at this step."""
        u = unit_interval(ctx_key, self.vocab_score_keys())
        if self.spec.ecc:
            bit = codeword[self.slot_of(ctx_key)]
            return (u >= 0.5) if bit else (u < 0.5)
        return u < self.spec.gamma

    # --------------------------------------------------------- adaptive strength
    def adaptive_delta(self, entropy: float, hits: int, total: int) -> float:
        """Per-token bias: scaled by model uncertainty, nudged up if the watermark is falling behind."""
        lo, hi = config.ADAPTIVE_ENTROPY_LOW, config.ADAPTIVE_ENTROPY_HIGH
        scale = min(1.0, max(0.0, (entropy - lo) / (hi - lo)))
        delta = config.ADAPTIVE_DELTA_MIN + (config.ADAPTIVE_DELTA_MAX - config.ADAPTIVE_DELTA_MIN) * scale
        if total >= 8:
            deficit = config.ADAPTIVE_FEEDBACK_TARGET - hits / total
            if deficit > 0:
                delta *= 1.0 + config.ADAPTIVE_FEEDBACK_GAIN * deficit
        return float(min(delta, 1.5 * config.ADAPTIVE_DELTA_MAX))

    def detector_weights(self, ids: List[int]) -> np.ndarray:
        """Per-token weights for the adaptive detector (low-entropy tokens count less)."""
        try:
            ent = self.backend.entropies(ids)
        except Exception:
            return np.ones(len(ids))
        w = np.clip(ent / config.ADAPTIVE_ENTROPY_HIGH, config.ADAPTIVE_WEIGHT_FLOOR, 1.0)
        return w

    # --------------------------------------------------------- detection side
    def score_text_ids(self, ids: List[int]):
        """
        Walk the token ids exactly as the generator did and return one row per token:
          (u, slot, dup, anchor_u)
            u         : context-layer u-value of the token
            slot      : payload slot (ECC methods), else -1
            dup       : True if this (context, token) pair already occurred -> ignored by the tests
            anchor_u  : context-free layer u-value if the token is the first occurrence of an
                        anchor-word cluster (anchor methods), else None
        """
        anchors: List[int] = []
        rows = []
        seen, seen_anchor = set(), set()
        for t, tid in enumerate(ids):
            skey, is_anchor = self.token_info(tid)
            ck = self.context_key(ids[:t], anchors)
            u = unit_interval_scalar(ck, skey)
            pair = (ck, skey)
            dup = pair in seen
            seen.add(pair)
            slot = self.slot_of(ck) if self.spec.ecc else -1
            anchor_u = None
            if self.spec.anchor_layer and is_anchor and skey not in seen_anchor:
                anchor_u = self.anchor_layer_u(skey)
                seen_anchor.add(skey)
            rows.append((u, slot, dup, anchor_u))
            if is_anchor:
                anchors.append(skey)
        return rows
