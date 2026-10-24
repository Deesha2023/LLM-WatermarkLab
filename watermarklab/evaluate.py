"""
evaluate.py
-----------
The benchmark: every watermark method x every attack x every sample prompt.

For each (method, prompt, sample) the benchmark
  1. generates a watermarked text (and a plain control text from the same model),
  2. detects the watermark before any attack      -> row "No attack"
  3. applies every attack and detects again       -> one row per attack
  4. runs the detector on the plain text          -> row "Control: no watermark"
     (its "detection accuracy" is the FALSE-POSITIVE RATE: lower is better)

Columns of the raw results table:
  method, attack, prompt_id, sample, detected, confidence, z_score,
  text_quality, payload_ok (error-correcting methods only), gen_ppl_ratio
"""

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import config
from .attacks import ATTACKS
from .detector import detect_method
from .generator import generate_plain, generate_watermarked
from .methods import METHODS
from .utils import text_quality_score

NO_ATTACK = "No attack"
CONTROL = "Control: no watermark"

DEFAULT_PROMPTS = [
    "Explain what artificial intelligence is to a beginner.",
    "Describe a sunny day.",
    "Explain what a computer is.",
]


def load_prompts(path: Optional[str] = None) -> List[str]:
    path = path or config.PROMPTS_FILE
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            prompts = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
        if prompts:
            return prompts
    return DEFAULT_PROMPTS


def run_full_benchmark(use_model_attacks: bool = False,
                       methods: Optional[List[str]] = None,
                       attacks: Optional[List[str]] = None,
                       prompts: Optional[List[str]] = None,
                       n_samples: int = 2,
                       max_new_tokens: Optional[int] = None,
                       save: bool = True,
                       progress=None) -> pd.DataFrame:
    methods = methods or list(METHODS)
    attack_names = attacks or list(ATTACKS)
    prompts = prompts or load_prompts()
    max_new_tokens = max_new_tokens or config.BENCHMARK_TOKENS

    rows = []
    total = len(methods) * len(prompts) * n_samples
    done = 0
    for mkey in methods:
        spec = METHODS[mkey]
        for pid, prompt in enumerate(prompts):
            for s in range(n_samples):
                seed = 10_000 + 97 * pid + s
                payload = (53 * pid + 29 * s + 7) % (1 << config.ECC_PAYLOAD_BITS) if spec.ecc else None
                gen = generate_watermarked(prompt, spec, max_new_tokens, payload=payload, seed=seed)
                plain = generate_plain(prompt, max_new_tokens, seed=seed + 5_000)

                def record(attack, text, source):
                    d = detect_method(text, spec)
                    rows.append({
                        "method": spec.label, "attack": attack, "prompt_id": pid, "sample": s,
                        "detected": bool(d["is_watermarked"]), "confidence": d["confidence"],
                        "z_score": d["z_score"],
                        "text_quality": text_quality_score(source, text) if attack not in (CONTROL,) else np.nan,
                        "payload_ok": (d.get("payload") == payload) if spec.ecc and attack != CONTROL else np.nan,
                    })

                record(NO_ATTACK, gen.text, gen.text)
                for name in attack_names:
                    fn = ATTACKS[name]
                    try:
                        attacked = fn(gen.text, use_model=use_model_attacks)
                    except TypeError:
                        attacked = fn(gen.text)
                    record(name, attacked, gen.text)
                record(CONTROL, plain, plain)
                done += 1
                if progress:
                    progress(done / total)

    df = pd.DataFrame(rows)
    if save:
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        df.to_csv(os.path.join(config.RESULTS_DIR, "benchmark.csv"), index=False)
    return df


def summarize_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (method, attack): detection accuracy %, mean z, text quality %, payload recovery %."""
    g = df.groupby(["method", "attack"], sort=False)
    out = g.agg(
        detection_accuracy=("detected", lambda x: 100.0 * x.mean()),
        avg_z_score=("z_score", "mean"),
        avg_text_quality=("text_quality", lambda x: 100.0 * x.mean()),
        payload_recovery=("payload_ok", lambda x: 100.0 * x.dropna().astype(float).mean() if x.notna().any() else np.nan),
    ).reset_index()
    order = {m.label: i for i, m in enumerate(METHODS.values())}
    att = {a: i for i, a in enumerate([NO_ATTACK] + list(ATTACKS) + [CONTROL])}
    out["_m"] = out["method"].map(lambda m: order.get(m, 99))
    out["_a"] = out["attack"].map(lambda a: att.get(a, 99))
    out = out.sort_values(["_m", "_a"]).drop(columns=["_m", "_a"]).reset_index(drop=True)
    return out.round(2)


def robustness_table(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot: detection accuracy % with methods as rows and attacks as columns (the headline table)."""
    s = summarize_benchmark(df)
    t = s.pivot(index="method", columns="attack", values="detection_accuracy")
    order = [m.label for m in METHODS.values() if m.label in t.index]
    cols = [c for c in [NO_ATTACK] + list(ATTACKS) + [CONTROL] if c in t.columns]
    t = t.loc[order, cols]
    attacked = [c for c in cols if c not in (NO_ATTACK, CONTROL)]
    if attacked:
        t["Mean under attack"] = t[attacked].mean(axis=1)
    return t.round(1)


def generation_quality_summary(methods: Optional[List[str]] = None,
                               prompts: Optional[List[str]] = None,
                               n_samples: int = 2,
                               max_new_tokens: Optional[int] = None) -> pd.DataFrame:
    """Fluency cost of watermarking: perplexity of watermarked text relative to plain text."""
    from .utils import perplexity
    methods = methods or list(METHODS)
    prompts = prompts or load_prompts()
    max_new_tokens = max_new_tokens or config.BENCHMARK_TOKENS
    rows = []
    for mkey in methods:
        spec = METHODS[mkey]
        ratios, ppls = [], []
        for pid, prompt in enumerate(prompts):
            for s in range(n_samples):
                seed = 10_000 + 97 * pid + s
                payload = (53 * pid + 29 * s + 7) % (1 << config.ECC_PAYLOAD_BITS) if spec.ecc else None
                wm = generate_watermarked(prompt, spec, max_new_tokens, payload=payload, seed=seed).text
                pl = generate_plain(prompt, max_new_tokens, seed=seed + 5_000)
                pw, pp = perplexity(wm), perplexity(pl)
                ppls.append(pw)
                if pp and not np.isnan(pp):
                    ratios.append(pw / pp)
        rows.append({"method": spec.label, "avg_perplexity": float(np.nanmean(ppls)),
                     "perplexity_vs_plain": float(np.nanmean(ratios))})
    return pd.DataFrame(rows).round(2)


def tradeoff_table(df: pd.DataFrame, quality: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Robustness vs fluency cost, side by side. A method only counts as "more robust"
    if it is not simply buying robustness with a stronger (more fluency-damaging) bias.
    """
    t = robustness_table(df)
    out = pd.DataFrame({
        "no_attack_detection": t["No attack"],
        "mean_detection_under_attack": t["Mean under attack"],
        "false_positive_rate": t["Control: no watermark"],
    })
    if quality is not None:
        q = quality.set_index("method")
        out["perplexity_vs_plain"] = q["perplexity_vs_plain"]
    return out.reset_index()
