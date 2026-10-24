"""
ecc.py
------
Error-correcting codes for the multi-bit (Error-Correcting) watermark.

Pipeline for a payload of k bits:

    payload bits --append CRC--> data bits --Hamming(7,4) per nibble--> codeword
    codeword --(embedded 1 bit per "slot" in the text)--> watermark

Hamming(7,4) corrects any single flipped bit inside each 7-bit block. Slots
that received no votes at all are treated as ERASURES: up to 2 erasures per
block are recoverable (they are brute-forced), because the code has minimum
distance 3.
"""

import itertools
from typing import List, Optional, Sequence, Tuple

# generator rows for Hamming(7,4): codeword = [d1 d2 d3 d4 p1 p2 p3]
_PARITY = [
    (0, 1, 3),   # p1 = d1 ^ d2 ^ d4
    (0, 2, 3),   # p2 = d1 ^ d3 ^ d4
    (1, 2, 3),   # p3 = d2 ^ d3 ^ d4
]


def int_to_bits(value: int, n: int) -> List[int]:
    return [(value >> (n - 1 - i)) & 1 for i in range(n)]


def bits_to_int(bits: Sequence[int]) -> int:
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def crc_bits(bits: Sequence[int], n_crc: int) -> List[int]:
    """Tiny CRC: polynomial division over GF(2) (x^4 + x + 1 for 4 bits)."""
    if n_crc <= 0:
        return []
    poly = {4: 0b10011, 8: 0b100000111, 2: 0b111}.get(n_crc, (1 << n_crc) | 0b11)
    reg = 0
    for b in list(bits) + [0] * n_crc:
        reg = (reg << 1) | int(b)
        if reg >> n_crc:
            reg ^= poly
    return int_to_bits(reg & ((1 << n_crc) - 1), n_crc)


def hamming74_encode_nibble(d: Sequence[int]) -> List[int]:
    p = [d[a] ^ d[b] ^ d[c] for (a, b, c) in _PARITY]
    return list(d) + p


def _syndrome(cw: Sequence[int]) -> Tuple[int, int, int]:
    return tuple((cw[a] ^ cw[b] ^ cw[c] ^ cw[4 + i]) for i, (a, b, c) in enumerate(_PARITY))


def hamming74_decode_block(cw: Sequence[Optional[int]]) -> Tuple[List[int], int, bool]:
    """
    Decode one 7-bit block. `None` entries are erasures.
    Returns (data_nibble, n_corrections, success).
    """
    erased = [i for i, b in enumerate(cw) if b is None]
    if len(erased) > 2:
        return [0, 0, 0, 0], 0, False
    best = None
    for fill in itertools.product([0, 1], repeat=len(erased)):
        trial = list(cw)
        for idx, v in zip(erased, fill):
            trial[idx] = v
        syn = _syndrome(trial)
        flips = 0
        if syn != (0, 0, 0):
            # find the single bit whose flip clears the syndrome
            fixed = False
            for pos in range(7):
                t2 = list(trial)
                t2[pos] ^= 1
                if _syndrome(t2) == (0, 0, 0):
                    trial, flips, fixed = t2, 1, True
                    break
            if not fixed:
                continue
        # prefer the candidate that needs the fewest corrections
        if best is None or flips < best[1]:
            best = (trial[:4], flips)
    if best is None:
        return [0, 0, 0, 0], 0, False
    return best[0], best[1], True


class ErrorCorrectingCode:
    """Payload <-> codeword, with CRC integrity check."""

    def __init__(self, payload_bits: int = 8, crc_bits_n: int = 4):
        self.payload_bits = payload_bits
        self.crc_n = crc_bits_n
        data_len = payload_bits + crc_bits_n
        self.pad = (-data_len) % 4          # pad to whole nibbles
        self.data_len = data_len + self.pad
        self.n_blocks = self.data_len // 4
        self.codeword_len = self.n_blocks * 7

    # ---- encode ----------------------------------------------------------
    def encode(self, payload: int) -> List[int]:
        if payload < 0 or payload >= (1 << self.payload_bits):
            raise ValueError(f"payload must fit in {self.payload_bits} bits")
        data = int_to_bits(payload, self.payload_bits)
        data = data + crc_bits(data, self.crc_n) + [0] * self.pad
        cw: List[int] = []
        for i in range(self.n_blocks):
            cw += hamming74_encode_nibble(data[4 * i: 4 * i + 4])
        return cw

    # ---- soft-decision maximum-likelihood decode --------------------------
    def _all_codewords(self):
        if not hasattr(self, "_cw_matrix"):
            import numpy as np
            n = 1 << self.payload_bits
            self._cw_matrix = np.array([self.encode(p) for p in range(n)], dtype=np.int8)
        return self._cw_matrix

    def decode_ml(self, llr: Sequence[float]):
        """
        Soft decoding. llr[i] = (weighted votes for bit 1) - (weighted votes for bit 0)
        in codeword slot i. Picks the VALID payload whose codeword best matches the
        votes (this is exactly maximum-likelihood decoding), so it never needs
        any slot to be perfectly clean and handles empty slots naturally.
        Only practical for payload_bits <= 16 (brute force over all payloads).
        Returns dict(payload, margin, corrected, erasures).
        """
        import numpy as np
        if self.payload_bits > 16:
            raise ValueError("decode_ml supports payloads up to 16 bits")
        llr = np.asarray(llr, dtype=np.float64)
        cw = self._all_codewords()
        scores = (2 * cw - 1) @ llr
        order = np.argsort(-scores)
        best = int(order[0])
        margin = float(scores[best] - scores[order[1]]) if len(order) > 1 else float(scores[best])
        hard = (llr > 0).astype(np.int8)
        return {
            "payload": best,
            "margin": margin,
            "corrected": int(((hard != cw[best]) & (llr != 0)).sum()),
            "erasures": int((llr == 0).sum()),
        }

    # ---- hard-decision decode ---------------------------------------------
    def decode(self, received: Sequence[Optional[int]]):
        """
        received: list of length codeword_len with 0 / 1 / None (erasure).
        Returns dict(payload, crc_ok, corrected, erasures, ok).
        """
        if len(received) != self.codeword_len:
            raise ValueError("wrong codeword length")
        data: List[int] = []
        corrected, erasures, ok = 0, sum(b is None for b in received), True
        for i in range(self.n_blocks):
            nib, flips, good = hamming74_decode_block(received[7 * i: 7 * i + 7])
            data += nib
            corrected += flips
            ok = ok and good
        payload_part = data[: self.payload_bits]
        crc_part = data[self.payload_bits: self.payload_bits + self.crc_n]
        crc_ok = crc_part == crc_bits(payload_part, self.crc_n)
        return {
            "payload": bits_to_int(payload_part),
            "crc_ok": bool(crc_ok and ok),
            "corrected": corrected,
            "erasures": erasures,
            "ok": bool(ok),
        }
