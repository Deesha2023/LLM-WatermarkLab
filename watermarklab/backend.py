"""
backend.py
----------
Language-model backends. The watermark code only talks to the small interface
defined by `LMBackend`, so it works unchanged with:

  * HFBackend  - a real Hugging Face causal LM (default: distilgpt2)
  * ToyBackend - an offline word-level n-gram LM trained on a built-in corpus.
                 It exists so the project (and its test-suite) run with no
                 internet, no GPU and no model download. Its absolute numbers
                 are NOT representative of a real LLM.

`get_backend()` returns a process-wide singleton chosen by config.BACKEND.
"""

import math
import re
import warnings
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from typing import Dict, List, Optional

import numpy as np

from . import config
from .hashing import splitmix64, splitmix64_np


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits.astype(np.float64) / max(temperature, 1e-6)
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


def entropy_nats(probs: np.ndarray) -> float:
    p = probs[probs > 1e-12]
    return float(-(p * np.log(p)).sum())


class LMBackend(ABC):
    name = "abstract"
    is_fallback = False
    vocab_size = 0
    eos_id: Optional[int] = None

    @abstractmethod
    def encode(self, text: str) -> List[int]: ...

    @abstractmethod
    def decode(self, ids: List[int]) -> str: ...

    @abstractmethod
    def token_str(self, tid: int) -> str:
        """Surface string of a token; a leading space marks a word start."""

    @abstractmethod
    def next_logits(self, ids: List[int]) -> np.ndarray:
        """Logits over the vocabulary for the token that follows `ids`."""

    def vocab_strings(self) -> List[str]:
        return [self.token_str(i) for i in range(self.vocab_size)]

    def token_keys(self) -> np.ndarray:
        """uint64 hash key for every vocabulary entry."""
        if not hasattr(self, "_token_keys"):
            self._token_keys = splitmix64_np(np.arange(self.vocab_size, dtype=np.uint64))
        return self._token_keys

    def key_of(self, tid: int) -> int:
        return splitmix64(int(tid))

    def entropies(self, ids: List[int], prefix: Optional[List[int]] = None) -> np.ndarray:
        """Next-token entropy at every position of `ids` (default: one call per position)."""
        prefix = list(prefix or [])
        out = []
        for i in range(len(ids)):
            logits = self.next_logits(prefix + list(ids[:i]))
            out.append(entropy_nats(softmax(logits)))
        return np.array(out)

    def perplexity(self, text: str) -> float:
        ids = self.encode(text)
        if len(ids) < 2:
            return float("nan")
        nll = 0.0
        for i in range(1, len(ids)):
            logits = self.next_logits(ids[:i])
            logp = logits - logits.max()
            logp = logp - np.log(np.exp(logp).sum())
            nll -= logp[ids[i]]
        return float(math.exp(nll / (len(ids) - 1)))


# ---------------------------------------------------------------------------
# Toy offline LM
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[a-z]+)?|\d+|[^\w\s]")


class ToyBackend(LMBackend):
    name = "toy-ngram"

    def __init__(self):
        from .corpus import CORPUS
        words = [w.lower() for w in _TOKEN_RE.findall(CORPUS)]
        counts = Counter(words)
        self._vocab: List[str] = ["<eos>"] + sorted(counts)
        self._index: Dict[str, int] = {w: i for i, w in enumerate(self._vocab)}
        self.vocab_size = len(self._vocab)
        self.eos_id = 0
        self._extra: Dict[int, str] = {}     # out-of-vocabulary words seen by encode()

        ids = [self._index[w] for w in words]
        self._uni = np.full(self.vocab_size, 0.05)
        for i in ids:
            self._uni[i] += 1.0
        self._uni /= self._uni.sum()
        self._bi: Dict[int, Counter] = defaultdict(Counter)
        self._tri: Dict[tuple, Counter] = defaultdict(Counter)
        for a, b in zip(ids, ids[1:]):
            self._bi[a][b] += 1
        for a, b, c in zip(ids, ids[1:], ids[2:]):
            self._tri[(a, b)][c] += 1
        # interpolation weights (a lot of unigram mass keeps entropy usable)
        self._lam = (0.40, 0.35, 0.25)       # trigram, bigram, unigram

    # --- tokenisation ------------------------------------------------------
    def encode(self, text: str) -> List[int]:
        ids = []
        for w in _TOKEN_RE.findall(text):
            w = w.lower()
            if w in self._index:
                ids.append(self._index[w])
            else:
                tid = self.vocab_size + (splitmix64(hash_str(w)) % (1 << 31))
                self._extra[tid] = w
                ids.append(tid)
        return ids

    def token_str(self, tid: int) -> str:
        w = self._vocab[tid] if tid < self.vocab_size else self._extra.get(tid, "<unk>")
        if re.match(r"[A-Za-z0-9]", w):
            return " " + w
        return w

    def decode(self, ids: List[int]) -> str:
        text = "".join(self.token_str(t) for t in ids if t != self.eos_id).strip()
        # capitalise sentence starts for readability
        text = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)
        return text

    # --- model ------------------------------------------------------------
    def next_logits(self, ids: List[int]) -> np.ndarray:
        V = self.vocab_size
        p = self._lam[2] * self._uni            # always keep some unigram mass
        # bigram / trigram evidence only when the preceding tokens are in-vocabulary
        if ids and ids[-1] < V:
            bi = self._bi.get(ids[-1])
            if bi:
                tot = sum(bi.values())
                for k, v in bi.items():
                    p[k] += self._lam[1] * v / tot
            if len(ids) >= 2 and ids[-2] < V:
                tri = self._tri.get((ids[-2], ids[-1]))
                if tri:
                    tot = sum(tri.values())
                    for k, v in tri.items():
                        p[k] += self._lam[0] * v / tot
        p = p / p.sum()
        p[self.eos_id] = 1e-9
        p = p / p.sum()
        return np.log(p)


def hash_str(s: str) -> int:
    from .hashing import string_to_key
    return string_to_key(s)


# ---------------------------------------------------------------------------
# Hugging Face backend
# ---------------------------------------------------------------------------
class HFBackend(LMBackend):
    def __init__(self, model_name: str = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.model_name = model_name or config.BASE_MODEL_NAME
        self.name = self.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.vocab_size = self.model.config.vocab_size
        self.eos_id = self.tokenizer.eos_token_id
        self._tok_strings: Optional[List[str]] = None
        self._past = None
        self._past_ids: List[int] = []
        cfg = self.model.config
        self._max_len = int(getattr(cfg, "n_positions", None)
                            or getattr(cfg, "max_position_embeddings", None) or 1024) - 1

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True).strip()

    def token_str(self, tid: int) -> str:
        if self._tok_strings is None:
            raw = self.tokenizer.convert_ids_to_tokens(list(range(self.tokenizer.vocab_size)))
            self._tok_strings = [
                (t or "").replace("Ġ", " ").replace("▁", " ").replace("Ċ", "\n")
                for t in raw
            ]
        if tid < len(self._tok_strings):
            return self._tok_strings[tid]
        return ""

    def next_logits(self, ids: List[int]) -> np.ndarray:
        torch = self._torch
        ids = (list(ids) or [self.eos_id])[-self._max_len:]      # stay inside the context window
        with torch.no_grad():
            # reuse the key/value cache when `ids` extends the previous call by one token
            if self._past is not None and len(ids) == len(self._past_ids) + 1 and ids[:-1] == self._past_ids:
                inp = torch.tensor([[ids[-1]]], device=self.device)
                out = self.model(inp, past_key_values=self._past, use_cache=True)
            else:
                inp = torch.tensor([ids], device=self.device)
                out = self.model(inp, use_cache=True)
            self._past = out.past_key_values
            self._past_ids = ids
        return out.logits[0, -1].float().cpu().numpy()[: self.vocab_size]

    def entropies(self, ids: List[int], prefix: Optional[List[int]] = None) -> np.ndarray:
        """Entropy of the next-token distribution at every position, in ONE forward pass."""
        torch = self._torch
        if not ids:
            return np.array([])
        n = len(ids)
        tail = list(ids)[-self._max_len:]                          # long texts: score the last window
        inp = torch.tensor([[self.eos_id] + tail[:-1]], device=self.device)
        with torch.no_grad():
            logits = self.model(inp).logits[0].float()
        probs = torch.softmax(logits, dim=-1)
        ent = (-(probs * torch.log(probs.clamp_min(1e-12))).sum(-1)).cpu().numpy()
        if n > len(tail):                                          # earlier tokens: use the mean entropy
            ent = np.concatenate([np.full(n - len(tail), ent.mean()), ent])
        return ent

    def perplexity(self, text: str) -> float:
        torch = self._torch
        ids = ([self.eos_id] + self.encode(text))[: self._max_len]
        if len(ids) < 3:
            return float("nan")
        inp = torch.tensor([ids], device=self.device)
        with torch.no_grad():
            loss = self.model(inp, labels=inp).loss
        return float(torch.exp(loss).item())


# ---------------------------------------------------------------------------
# Singleton access
# ---------------------------------------------------------------------------
_BACKEND: Optional[LMBackend] = None


def get_backend() -> LMBackend:
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    mode = config.BACKEND
    if mode == "toy":
        _BACKEND = ToyBackend()
    elif mode == "hf":
        _BACKEND = HFBackend()
    else:
        try:
            _BACKEND = HFBackend()
        except Exception as exc:     # no torch / no internet / model missing
            warnings.warn(
                f"Could not load Hugging Face model '{config.BASE_MODEL_NAME}' ({exc!r}). "
                "Falling back to the OFFLINE TOY language model; results are for "
                "testing only. Install torch+transformers and connect to the "
                "internet once to use the real model.",
                RuntimeWarning,
            )
            _BACKEND = ToyBackend()
            _BACKEND.is_fallback = True
    return _BACKEND


def set_backend(backend: Optional[LMBackend]) -> None:
    """Force a specific backend (used by tests)."""
    global _BACKEND
    _BACKEND = backend
