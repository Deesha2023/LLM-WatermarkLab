"""
run_demo.py
-----------
A single command that runs the ENTIRE project end to end, from the
command line, no dashboard needed:

    python run_demo.py            # full run
    python run_demo.py --quick    # smaller benchmark (a few minutes on CPU)

What it does, step by step:
  1. Loads the language model (falls back to the offline toy LM if it can't).
  2. Generates a watermarked text (basic method) and detects it.
  3. Attacks it with synonym replacement and detects again.
  4. Trains the small ML classifier.
  5. Demonstrates the three new features:
       - Semantic-Anchor watermark recovery
       - Adaptive Watermark Strength
       - Error-Correcting (multi-bit payload) watermark
  6. Runs the full benchmark (all methods x all attacks x all prompts) and
     prints + saves the result tables.

First run downloads the base model (distilgpt2, ~330 MB) if you have
internet; otherwise it uses the offline toy model automatically.
"""

import argparse
import os
import sys
import time


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller benchmark (4 prompts, 1 sample each)")
    ap.add_argument("--real-attacks", action="store_true",
                    help="use real NLP-model attacks (T5 / MarianMT / flan-t5; ~600 MB extra downloads)")
    args = ap.parse_args()

    start = time.time()
    os.makedirs("results", exist_ok=True)

    section("STEP 1: Loading the language model (first run may download ~330 MB)")
    from watermarklab.backend import get_backend
    backend = get_backend()
    print(f"Backend: {backend.name}" + ("   <-- OFFLINE TOY FALLBACK (results are for testing only)"
                                         if backend.is_fallback else ""))

    from watermarklab.generator import (generate_basic_watermarked, generate_plain,
                                        generate_watermarked)
    from watermarklab.detector import detect_method, statistical_detect

    section("STEP 2: Generating a watermarked sample (basic method)")
    prompt = "Explain what artificial intelligence is to a beginner."
    watermarked_text = generate_basic_watermarked(prompt, max_new_tokens=80)
    print(f"Prompt: {prompt}\n")
    print("Watermarked output:")
    print(watermarked_text)

    section("STEP 3: Detecting the watermark (before any attack)")
    before = statistical_detect(watermarked_text)
    print(f"Detected as watermarked : {before['is_watermarked']}")
    print(f"Confidence              : {before['confidence']*100:.1f}%")
    print(f"z-score                 : {before['z_score']:.2f}")

    section("STEP 4: Attacking the text (synonym replacement) and re-detecting")
    from watermarklab.attacks import synonym_replace_attack
    attacked_text = synonym_replace_attack(watermarked_text, fraction=0.3)
    print("Attacked output:")
    print(attacked_text)
    after = statistical_detect(attacked_text)
    print(f"\nDetected as watermarked : {after['is_watermarked']}")
    print(f"Confidence              : {after['confidence']*100:.1f}%")
    print(f"z-score                 : {after['z_score']:.2f}")
    print(f"--> Confidence went from {before['confidence']*100:.1f}% to {after['confidence']*100:.1f}% after the attack.")

    section("STEP 5: Training the small ML classifier")
    from watermarklab.detector import train_ml_classifier, hybrid_detect
    train_prompts = ["Describe a sunny day.", "Explain what a computer is.",
                     "Why is education important for students?", "Describe life in a big city.",
                     "Talk about the benefits of regular exercise.", "Explain how the weather changes."]
    print("Generating labelled examples (watermarked vs plain)...")
    wm_examples = [generate_basic_watermarked(p, max_new_tokens=80, seed=300 + i) for i, p in enumerate(train_prompts)]
    pl_examples = [generate_plain(p, max_new_tokens=80, seed=700 + i) for i, p in enumerate(train_prompts)]
    train_ml_classifier(wm_examples, pl_examples)
    print("Classifier trained and saved to results/ml_classifier.joblib")
    h = hybrid_detect(watermarked_text, "basic")
    print(f"Hybrid verdict on the sample: watermarked={h['is_watermarked']}, "
          f"stat confidence={h['stat_confidence']*100:.1f}%, ML probability={h.get('ml_probability', float('nan'))*100:.1f}%")

    # ---------------------------------------------------------------- NEW FEATURES
    from watermarklab.attacks import paraphrase_attack

    section("STEP 6a: NEW - Semantic-Anchor Watermark Recovery")
    r = generate_watermarked(prompt, "anchor", 120)
    d0 = detect_method(r.text, "anchor")
    attacked = paraphrase_attack(r.text)
    d1 = detect_method(attacked, "anchor")
    anchors = [t["token"].strip() for t in r.trace if t["is_anchor"]][:12]
    print(f"Anchor words (meaning-bearing words used as seeds): {', '.join(anchors)} ...")
    print(f"Before attack: z={d0['z_score']:.2f}  (context layer {d0['z_context_layer']:.2f}, "
          f"anchor layer {d0['z_anchor_layer']:.2f})")
    print(f"After paraphrase: z={d1['z_score']:.2f}  (context layer {d1['z_context_layer']:.2f}, "
          f"anchor layer {d1['z_anchor_layer']:.2f})  recovery used: {d1['recovery_used']}")

    section("STEP 6b: NEW - Adaptive Watermark Strength")
    import numpy as np
    ra = generate_watermarked(prompt, "adaptive", 120)
    deltas = np.array([t["delta"] for t in ra.trace])
    ents = np.array([t["entropy"] for t in ra.trace])
    print(f"Bias applied per token ranged from {deltas.min():.2f} to {deltas.max():.2f} "
          f"(mean {deltas.mean():.2f}).")
    lo, hi = np.quantile(ents, [1 / 3, 2 / 3])
    print(f"Most predictable third of the tokens got mean bias {deltas[ents <= lo].mean():.2f}; "
          f"the most open-ended third got {deltas[ents >= hi].mean():.2f}.")
    da = detect_method(ra.text, "adaptive")
    print(f"Detection: watermarked={da['is_watermarked']}, z={da['z_score']:.2f}")

    section("STEP 6c: NEW - Error-Correcting Watermark (hidden payload)")
    from watermarklab.attacks import ATTACKS
    payload = 0b10110010
    re_ = generate_watermarked(prompt, "ecc", 150, payload=payload)
    print(f"Hidden payload: {payload:08b}   (codeword: {''.join(map(str, re_.codeword))})")
    for name in ["(no attack)", "Synonym Replacement", "Random Word Deletion", "Paraphrasing"]:
        txt = re_.text if name == "(no attack)" else ATTACKS[name](re_.text)
        d = detect_method(txt, "ecc")
        status = "RECOVERED" if d["payload"] == payload else "wrong"
        print(f"  {name:22s} -> decoded {d['payload']:08b}  [{status}]  "
              f"z={d['z_score']:.2f}, bit errors corrected={d['corrected_bits']}")

    section("STEP 7: Running the full benchmark (this is the slowest step)")
    from watermarklab.evaluate import (generation_quality_summary, load_prompts, robustness_table,
                                       run_full_benchmark, summarize_benchmark, tradeoff_table)
    prompts = load_prompts()
    kw = {}
    if args.quick:
        prompts, kw = prompts[:4], {"n_samples": 1}
    if args.quick:
        print("(--quick uses very few samples: percentages move in big steps and are NOISY; "
              "run without --quick for stable numbers)")
    print(f"{len(prompts)} prompts x 6 methods x 7 attacks "
          f"({'real NLP-model attacks' if args.real_attacks else 'fast offline attacks'}) ...")
    df = run_full_benchmark(use_model_attacks=args.real_attacks, prompts=prompts, **kw)
    summary = summarize_benchmark(df)
    quality = generation_quality_summary(prompts=prompts, n_samples=kw.get("n_samples", 2))

    section("RESULTS 1: detection accuracy (%) by method and attack  (control column = false positives)")
    import pandas as pd
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(robustness_table(df).to_string())

    section("RESULTS 2: robustness vs fluency cost (perplexity relative to plain text; lower = less damage)")
    print(tradeoff_table(df, quality).to_string(index=False))

    section("RESULTS 3: hidden-payload recovery (%) for the error-correcting methods")
    pr = summary[summary["payload_recovery"].notna()][["method", "attack", "payload_recovery"]]
    print(pr.to_string(index=False))

    summary.to_csv("results/benchmark_summary.csv", index=False)
    robustness_table(df).to_csv("results/robustness_table.csv")
    tradeoff_table(df, quality).to_csv("results/tradeoff_table.csv", index=False)
    print("\nSaved: results/benchmark.csv, benchmark_summary.csv, robustness_table.csv, tradeoff_table.csv")

    elapsed = time.time() - start
    section(f"DONE in {elapsed/60:.1f} minutes")
    print("Next: run 'streamlit run app.py' to explore results interactively.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(1)
