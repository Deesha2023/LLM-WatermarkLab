"""
methods.py
----------
The registry of watermarking methods compared by the platform. A method is just
a MethodSpec; generator, detector, benchmark and dashboard are all driven by it.

  basic      Method A : green list seeded by the previous token
  context    Method B : green list seeded by the previous 4 tokens
  anchor     NEW (1)  : Semantic-Anchor watermark with recovery detection
  adaptive   NEW (2)  : Adaptive Watermark Strength (entropy-aware)
  ecc        NEW (3)  : Error-Correcting multi-bit watermark (hidden payload)
  full       all three new features combined
"""

from dataclasses import dataclass, replace
from typing import Dict

from . import config


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    seed_mode: str = "window"          # "window" (previous tokens) | "anchor" (previous content word)
    window: int = 1
    adaptive: bool = False
    ecc: bool = False
    gamma: float = config.GREEN_FRACTION
    delta: float = config.WATERMARK_STRENGTH
    anchor_layer: bool = False         # add the context-free anchor-word layer (Semantic-Anchor)
    anchor_delta: float = config.ANCHOR_LAYER_DELTA
    anchor_gamma: float = config.ANCHOR_LAYER_GAMMA
    description: str = ""

    @property
    def context_window(self) -> int:
        return self.window

    def with_(self, **kw) -> "MethodSpec":
        return replace(self, **kw)


METHODS: Dict[str, MethodSpec] = {
    "basic": MethodSpec(
        key="basic", label="Basic Statistical", window=config.BASIC_CONTEXT_WINDOW,
        description="Green list seeded by the previous token (Method A)."),
    "context": MethodSpec(
        key="context", label="Context-Aware (Proposed)", window=config.CONTEXT_AWARE_WINDOW,
        description="Green list seeded by the previous 4 tokens (Method B)."),
    "anchor": MethodSpec(
        key="anchor", label="Semantic-Anchor", seed_mode="anchor", anchor_layer=True,
        gamma=config.ANCHOR_GREEN_FRACTION, delta=config.ANCHOR_CONTEXT_DELTA,
        description="Two layers: (1) a green list seeded by the latest meaning-bearing word, "
                    "and (2) a context-free watermark on the meaning-bearing words themselves, "
                    "scored on synonym clusters. When paraphrasing destroys layer 1, the "
                    "detector recovers the watermark from the surviving anchors in layer 2."),
    "adaptive": MethodSpec(
        key="adaptive", label="Adaptive Strength", window=config.BASIC_CONTEXT_WINDOW,
        adaptive=True,
        description="Basic scheme, but the bias is scaled per token by the model's "
                    "uncertainty: strong where many words fit, gentle where one word is obvious."),
    "ecc": MethodSpec(
        key="ecc", label="Error-Correcting", window=config.BASIC_CONTEXT_WINDOW,
        ecc=True, gamma=0.5, delta=2.5,
        description="Hides a multi-bit payload (Hamming code + CRC) so it can be read back "
                    "even after some tokens are altered."),
    "full": MethodSpec(
        key="full", label="Full (Anchor+Adaptive+ECC)", seed_mode="anchor", anchor_layer=True,
        adaptive=True, ecc=True, gamma=0.5, delta=config.ANCHOR_CONTEXT_DELTA,
        description="Semantic-anchor seeding + adaptive strength + error-correcting payload."),
}

PLAIN_LABEL = "None (plain baseline)"

BY_LABEL: Dict[str, MethodSpec] = {m.label: m for m in METHODS.values()}


def get_method(name: str) -> MethodSpec:
    """Accept either a key ('anchor') or a label ('Semantic-Anchor')."""
    if name in METHODS:
        return METHODS[name]
    if name in BY_LABEL:
        return BY_LABEL[name]
    raise KeyError(f"Unknown watermark method: {name!r}. Options: {list(METHODS)}")
