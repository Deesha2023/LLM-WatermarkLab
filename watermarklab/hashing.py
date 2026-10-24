"""
hashing.py
----------
Small, fast, deterministic 64-bit hashing helpers (SplitMix64). Everything the
watermark does - choosing green lists, choosing payload-bit slots - is a pure
function of (secret key, context, token), so generator and detector agree
without sharing any state except the key.
"""

import hashlib

import numpy as np

MASK64 = (1 << 64) - 1
_GOLDEN = 0x9E3779B97F4A7C15
_M1 = 0xBF58476D1CE4E5B9
_M2 = 0x94D049BB133111EB


def splitmix64(x: int) -> int:
    x = (x + _GOLDEN) & MASK64
    z = x
    z = ((z ^ (z >> 30)) * _M1) & MASK64
    z = ((z ^ (z >> 27)) * _M2) & MASK64
    return z ^ (z >> 31)


def splitmix64_np(x: np.ndarray) -> np.ndarray:
    """Vectorised SplitMix64 over a uint64 array."""
    x = x.astype(np.uint64, copy=False)
    with np.errstate(over="ignore"):
        x = x + np.uint64(_GOLDEN)
        z = x
        z = (z ^ (z >> np.uint64(30))) * np.uint64(_M1)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(_M2)
        return z ^ (z >> np.uint64(31))


def key_to_int(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little")


def string_to_key(s: str) -> int:
    """Stable 64-bit key for a string (used for canonical word labels)."""
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little")


def combine(*values: int) -> int:
    """Order-sensitive hash of several ints into one 64-bit int."""
    h = 0x243F6A8885A308D3
    for v in values:
        h = splitmix64((h ^ (int(v) & MASK64)) & MASK64)
    return h


def unit_interval(context_key: int, token_keys: np.ndarray) -> np.ndarray:
    """
    Map (context, token) -> pseudo-random number in [0, 1).
    token_keys: uint64 array. Returns float64 array of the same shape.
    """
    with np.errstate(over="ignore"):
        mixed = splitmix64_np((token_keys * np.uint64(_M1)) ^ np.uint64(context_key))
    return (mixed >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def unit_interval_scalar(context_key: int, token_key: int) -> float:
    mixed = splitmix64(((int(token_key) * _M1) & MASK64) ^ int(context_key))
    return (mixed >> 11) / float(1 << 53)
