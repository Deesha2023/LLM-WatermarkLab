"""
generator.py
------------
Plants the watermark while text is generated.

Public API
----------
generate_watermarked(prompt, method, max_new_tokens, payload=None, seed=None)
        -> GenerationResult(text, trace, payload, codeword, method)
generate_plain(prompt, max_new_tokens)                       -> str
generate_basic_watermarked(prompt, max_new_tokens)           -> str   (Method A)
generate_context_aware_watermarked(prompt, max_new_tokens)   -> str   (Method B)
generate_semantic_anchor_watermarked(...)                    -> str   (NEW 1)
generate_adaptive_watermarked(...)                           -> str   (NEW 2)
generate_ecc_watermarked(prompt, max_new_tokens, payload)    -> str   (NEW 3)
generate_full_watermarked(prompt, max_new_tokens, payload)   -> str   (all three)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from . import config
from .backend import entropy_nats, get_backend, softmax
from .methods import MethodSpec, get_method
from .scheme import WatermarkScheme


@dataclass
class GenerationResult:
    text: str
    method: str
    payload: Optional[int] = None
    codeword: Optional[List[int]] = None
    trace: List[Dict] = field(default_factory=list)   # per-token diagnostics

    def __str__(self):
        return self.text


def _sample(logits: np.ndarray, rng: np.random.Generator, temperature: float, top_k: int,
            bias: Optional[np.ndarray]):
    """Top-k candidates come from the ORIGINAL model; the logit bonus `bias` is added inside that set."""
    k = min(top_k, len(logits))
    cand = np.argpartition(-logits, k - 1)[:k]
    bonus = np.zeros(k) if bias is None else bias[cand]
    probs = softmax(logits[cand] + bonus, temperature)
    j = rng.choice(k, p=probs)
    return int(cand[j])


def generate_watermarked(prompt: str, method="basic", max_new_tokens: int = None,
                         payload: Optional[int] = None, seed: Optional[int] = None,
                         key: Optional[str] = None) -> GenerationResult:
    spec = method if isinstance(method, MethodSpec) else get_method(method)
    backend = get_backend()
    scheme = WatermarkScheme(spec, backend, key)
    max_new_tokens = max_new_tokens or config.MAX_NEW_TOKENS
    rng = np.random.default_rng(config.SEED if seed is None else seed)

    codeword = None
    if spec.ecc:
        payload = config.ECC_DEFAULT_PAYLOAD if payload is None else payload
        codeword = scheme.code.encode(payload)

    prompt_ids = backend.encode(prompt)
    gen: List[int] = []
    anchors: List[int] = []
    hits = total = 0
    trace: List[Dict] = []

    for _ in range(max_new_tokens):
        logits = backend.next_logits(prompt_ids + gen)
        ck = scheme.context_key(gen, anchors)
        target = scheme.target_mask(ck, codeword)

        probs_model = softmax(logits)
        ent = entropy_nats(probs_model)
        delta = scheme.adaptive_delta(ent, hits, total) if spec.adaptive else spec.delta

        bias = delta * target.astype(np.float64)
        if spec.anchor_layer:                       # context-free layer on the anchor words
            ratio = delta / spec.delta if spec.adaptive else 1.0
            bias = bias + spec.anchor_delta * ratio * scheme.anchor_layer_mask().astype(np.float64)

        tid = _sample(logits, rng, config.TEMPERATURE, config.TOP_K, bias)
        if tid == backend.eos_id:
            break
        in_target = bool(target[tid])
        hits += in_target
        total += 1
        skey, is_anchor = scheme.token_info(tid)
        trace.append({
            "token": backend.token_str(tid), "entropy": ent, "delta": delta,
            "in_target": in_target, "is_anchor": is_anchor,
            "anchor_green": bool(spec.anchor_layer and scheme.anchor_layer_mask()[tid]),
            "slot": scheme.slot_of(ck) if spec.ecc else -1,
        })
        gen.append(tid)
        if is_anchor:
            anchors.append(skey)

    return GenerationResult(text=backend.decode(gen), method=spec.key, payload=payload,
                            codeword=codeword, trace=trace)


def generate_plain(prompt: str, max_new_tokens: int = None, seed: Optional[int] = None) -> str:
    """Un-watermarked sampling from the same model (control / baseline)."""
    backend = get_backend()
    rng = np.random.default_rng(config.SEED + 1 if seed is None else seed)
    prompt_ids = backend.encode(prompt)
    gen: List[int] = []
    for _ in range(max_new_tokens or config.MAX_NEW_TOKENS):
        logits = backend.next_logits(prompt_ids + gen)
        tid = _sample(logits, rng, config.TEMPERATURE, config.TOP_K, None)
        if tid == backend.eos_id:
            break
        gen.append(tid)
    return backend.decode(gen)


# ---- convenience wrappers (names used by run_demo.py / app.py) -------------
def generate_basic_watermarked(prompt, max_new_tokens=None, seed=None):
    return generate_watermarked(prompt, "basic", max_new_tokens, seed=seed).text


def generate_context_aware_watermarked(prompt, max_new_tokens=None, seed=None):
    return generate_watermarked(prompt, "context", max_new_tokens, seed=seed).text


def generate_semantic_anchor_watermarked(prompt, max_new_tokens=None, seed=None):
    return generate_watermarked(prompt, "anchor", max_new_tokens, seed=seed).text


def generate_adaptive_watermarked(prompt, max_new_tokens=None, seed=None):
    return generate_watermarked(prompt, "adaptive", max_new_tokens, seed=seed).text


def generate_ecc_watermarked(prompt, max_new_tokens=None, payload=None, seed=None):
    return generate_watermarked(prompt, "ecc", max_new_tokens, payload=payload, seed=seed).text


def generate_full_watermarked(prompt, max_new_tokens=None, payload=None, seed=None):
    return generate_watermarked(prompt, "full", max_new_tokens, payload=payload, seed=seed).text
