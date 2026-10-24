# LLM-WatermarkLab

**A Robustness-Aware Framework for Watermarking and Detection of AI-Generated Text**

A working implementation of the B.Tech major-project synopsis: it generates AI text with a hidden
watermark, detects the watermark, attacks the text (paraphrasing, translation, synonym swaps,
AI rewriting, deletion, restructuring) and measures how well the watermark survives.

**Version 2 adds three new features**, all wired into the generator, detector, benchmark, tests
and dashboard:

| # | Feature | One-line idea |
|---|---|---|
| 1 | **Semantic-Anchor Watermark Recovery** | Key the watermark on meaning-bearing "anchor" words and their synonym clusters; if paraphrasing destroys the context layer, recover the watermark from the surviving anchors. |
| 2 | **Adaptive Watermark Strength** | Bias strongly only where the model is uncertain, gently where one word is obvious, and weight the detector the same way. |
| 3 | **Error-Correcting Watermark** | Hide a multi-bit payload (author / model ID) with a Hamming code + CRC and soft maximum-likelihood decoding, so it can be read back even after tokens are altered. |

Everything runs on a normal laptop CPU. No paid API keys, no GPU required.

---

## 1. What's in this folder

```
LLM-WatermarkLab/
├── app.py                      <- the dashboard (4 tabs)
├── run_demo.py                 <- command-line end-to-end demo (python run_demo.py [--quick])
├── requirements.txt
├── watermarklab/
│   ├── config.py               <- ALL settings (model, strengths, thresholds, feature parameters)
│   ├── methods.py              <- registry of the 6 watermark methods (MethodSpec)
│   ├── scheme.py               <- shared math: context keys, green lists, anchors, adaptive strength, payload slots
│   ├── generator.py            <- plants the watermark while text is generated
│   ├── detector.py             <- statistical detector (all methods) + ML classifier + hybrid
│   ├── ecc.py                  <- Hamming(7,4) + CRC + soft ML decoder   (feature 3)
│   ├── lexicon.py              <- anchor words + synonym clusters         (feature 1)
│   ├── attacks.py              <- 7 attacks (synonym, paraphrase, back-translation, LLM rewrite, ...)
│   ├── evaluate.py             <- benchmark, robustness table, fluency/robustness trade-off
│   ├── backend.py              <- Hugging Face backend + offline toy-LM backend
│   ├── corpus.py               <- tiny corpus for the offline toy LM
│   ├── hashing.py              <- deterministic 64-bit hashing
│   └── utils.py                <- text-quality and perplexity helpers
├── data/prompts.txt            <- 12 sample prompts
├── tests/                      <- 57 automated tests (no model download needed)
└── results/                    <- benchmark output is saved here
```

---

## 2. Installation (step by step)

1. **Python 3.9+** (`python3 --version`).
2. Unzip and enter the folder: `unzip LLM-WatermarkLab.zip && cd LLM-WatermarkLab`
3. (Recommended) virtual environment: `python3 -m venv venv`, then
   `source venv/bin/activate` (Mac/Linux) or `venv\Scripts\activate` (Windows).
4. `pip install -r requirements.txt` (2-5 minutes; PyTorch is the big one).

The AI model itself (`distilgpt2`, ~330 MB) downloads automatically the first time you generate text.

> **No internet / no PyTorch?** The project still runs. It automatically falls back to a small
> **offline toy language model** and prints a clear warning. Everything (generation, detection,
> attacks, benchmark, dashboard, tests) works, but the *numbers are for testing only* - they are not
> representative of a real LLM. Force it with `WATERMARKLAB_BACKEND=toy`, or force the real model with
> `WATERMARKLAB_BACKEND=hf`.

---

## 3. Testing the installation

**Option A - fast offline test (~10 seconds, no download):**
```bash
pytest tests/ -v
```
Expected: `57 passed`. This checks the hashing, green-list statistics, false-positive rate, every
attack, the ECC code, adaptive strength, anchor recovery and the benchmark plumbing.

**Option B - full end-to-end demo:**
```bash
python run_demo.py --quick      # a few minutes on CPU with the real model
python run_demo.py              # full benchmark (12 prompts x 2 samples x 6 methods x 7 attacks)
python run_demo.py --real-attacks   # use real NLP-model attacks (~600 MB extra downloads)
```
It walks through generation, detection, attack, the ML classifier, each new feature, and the full
benchmark, then saves tables to `results/`. `--quick` uses very few samples, so its percentages are
noisy - use the full run for numbers you report.

---

## 4. The dashboard

```bash
streamlit run app.py
```

| Tab | What it does |
|---|---|
| **Generate & Detect** | Pick any of the 6 methods (or plain text), generate, see detected? / confidence / z-score / p-value. For error-correcting methods you also type the hidden payload and see the decoded one. |
| **Attack Lab** | Attack any text with any of 7 attacks, choose which detector to use, compare before vs after (and whether the payload survived). |
| **Benchmark Results** | Run the whole benchmark: headline detection table, **robustness-vs-fluency trade-off** chart, text-quality chart, payload-recovery chart, CSV download. |
| **New Features Lab** | One interactive demo per new feature: anchor words highlighted with per-layer z-scores before/after attack; bias-vs-uncertainty chart; per-slot payload vote chart. |

---

## 5. How the three new features work

All methods share one idea: for each position the scheme derives a secret **context key** from the
text so far, splits candidate words into a *target set* and the rest, and the generator adds a small
logit bonus δ to target words. The detector recomputes the same keys from the text alone and counts how
many words landed in their target set (a z-test).

### 5.1 Semantic-Anchor Watermark Recovery  (`scheme.py`, `lexicon.py`, `detector.py`)
Standard watermarks seed on the previous 1-4 *tokens*, so changing one word breaks the next several
checks. This method has two layers:

1. **Context layer** - seeded on the latest **anchor** (a meaning-bearing content word: >= 4 letters, not a
   stopword). Replacing a function word does not disturb it. Words are judged by their **synonym-cluster
   label** (`big/large/huge` and `run/runs/running` share one label), so swaps *inside* a cluster change nothing
   (this is unit-tested to give an identical z-score).
2. **Anchor layer** - a context-free watermark on the anchor words themselves (green clusters get a bonus δ_a).
   It does not depend on neighbours, so it keeps working when edits scramble the context.

**Recovery:** the detector computes `z_context`, `z_anchor` and the Stouffer combination
`(z_context + z_anchor)/sqrt(2)` and reports the best. All three are ~N(0,1) without a watermark, and the
z >= 4 threshold is strict enough that taking the best of three costs almost no false-positive budget
(measured FPR is 0). Repeated (context, word) pairs and repeated anchors are counted once.

### 5.2 Adaptive Watermark Strength  (`scheme.py`, `generator.py`, `detector.py`)
At each step the generator measures the model's **entropy** H:

```
delta_t = DELTA_MIN + (DELTA_MAX - DELTA_MIN) * clip((H - H_low) / (H_high - H_low), 0, 1)
```
plus a small feedback boost if the running share of target words falls behind. Predictable steps
(one obvious word) get almost no bias, so fluency is protected; open-ended steps get a strong bias, so
the signal is concentrated where it is cheap. The detector uses per-token weights from the same
entropy, via a weighted z-test that stays valid under the "no watermark" null.

### 5.3 Error-Correcting Watermark  (`ecc.py`, `scheme.py`, `detector.py`)
```
payload (8 bits) -> + CRC-4 -> Hamming(7,4) per nibble -> 21-bit codeword
each token -> slot = hash(context) mod 21 -> votes for that codeword bit (green list = "bit 0" or "bit 1" half of the vocabulary)
```
Because the slot comes from a *hash of the context*, not from the token position, insertions and
deletions do not shift every later vote out of sync. Decoding is **soft maximum-likelihood over all 256
valid payloads**, which also handles empty slots and is the optimal use of the votes. Detection evidence is
the agreement between the votes and the decoded codeword, **Bonferroni-corrected for the 2^8 payloads the
decoder could have picked** (otherwise a random text would always "decode" to something). The result also
reports the classic hard-decision Hamming + CRC check as a second opinion, and how many bit errors were
corrected. Payloads need roughly 100+ tokens (`ECC_MIN_TOKENS`) to decode reliably.

---

## 6. Understanding the results

- **Detection accuracy** - % of texts still flagged as AI-generated. Higher = survives better.
- **Control column / false-positive rate** - % of *plain* (un-watermarked) texts wrongly flagged. Must stay ~0.
- **z-score / p-value** - statistical evidence. z >= 4.0 (`Z_SCORE_THRESHOLD`) means "watermarked".
- **Confidence** - a logistic of `(z - threshold)`: exactly **50% at the decision boundary**, ~73% one z-unit
  above it. (The raw one-sided p-value saturates at 100% for any z > 4, which hides before/after-attack
  differences, so it is reported separately as `p_value`.)
- **Text quality** - word/bigram overlap between attacked and original text (embeddings if installed).
- **Perplexity vs plain** - the fluency cost of watermarking: perplexity of watermarked text divided by
  that of plain text from the same model (lower = less damage).
- **Payload recovery** - % of runs where the decoded payload exactly equals the embedded one.

**Read robustness together with fluency cost.** A bigger bias δ makes *any* method more robust but
hurts text; the Benchmark tab therefore plots detection-under-attack against perplexity. Adaptive Strength
is compared with Basic **at matched fluency** (Basic at δ = 2.2), not at matched δ.

### Example results (offline toy model, 24 samples per cell - NOT representative of a real LLM)

| Method | No attack | Mean under 7 attacks | False positives | Perplexity vs plain |
|---|---|---|---|---|
| Basic Statistical | 87.5% | 44.6% | 0% | 1.60x |
| Context-Aware (Proposed) | 100% | 26.2% | 0% | 1.53x |
| **Semantic-Anchor** | 95.8% | 53.0% | 0% | 1.63x |
| **Adaptive Strength** | 100% | **80.4%** | 0% | 1.66x |
| **Error-Correcting** | 91.7% | 45.8% | 0% | 2.03x |
| **Full (Anchor+Adaptive+ECC)** | 100% | 58.3% | 0% | 1.71x |

Hidden-payload recovery: Error-Correcting 95.8% with no attack and 58-100% under attacks (hardest:
LLM rewriting); Full 95.8% with no attack and 46-92% under attacks.

What to take from this (on this toy model): Adaptive Strength gives a large robustness gain at about the same
fluency cost as Basic; Semantic-Anchor gives a modest gain at about the same cost; the error-correcting payload
costs more fluency (its per-token bias is stronger and it needs the multiple-testing correction) but adds a
capability the others lack - a message that can be read back. The window-4 "Context-Aware" method is *less*
robust than Basic under edits in this benchmark (a longer seed window breaks on more edits). **Re-run the
benchmark with the real model before drawing conclusions for your report** - effects can differ on a real LLM.

---

## 7. Adjusting the project scope (`watermarklab/config.py`)

| Setting | Meaning |
|---|---|
| `BASE_MODEL_NAME` | Swap `distilgpt2` for a larger Hugging Face causal LM (Qwen / Llama / Mistral) if you have a GPU. |
| `BACKEND` | `"auto"` (HF, else toy), `"hf"`, `"toy"`; or env var `WATERMARKLAB_BACKEND`. |
| `WATERMARK_STRENGTH`, `GREEN_FRACTION` | Basic bias δ and green share γ. |
| `Z_SCORE_THRESHOLD` | How strict the detector is. |
| `BASIC_CONTEXT_WINDOW`, `CONTEXT_AWARE_WINDOW` | Seed window of Method A / B (1 and 4). |
| `ANCHOR_*` | Anchor length, context-layer δ, anchor-layer δ and γ. |
| `ADAPTIVE_*` | δ range, entropy range, feedback target/gain, detector weight floor. |
| `ECC_PAYLOAD_BITS`, `ECC_CRC_BITS`, `ECC_DEFAULT_PAYLOAD` | Payload size (up to 16 bits), checksum size, demo payload. |
| `BENCHMARK_TOKENS` | Text length used in the benchmark (120 so payloads decode fairly). |

To add your own method, add a `MethodSpec` to `watermarklab/methods.py`; the generator, detector,
benchmark and dashboard pick it up automatically.

---

## 8. Honest notes and limitations

- **Offline attacks are simulations.** By default the Paraphrasing, Back-Translation and LLM-Rewriting attacks
  use rule-based fallbacks (synonym swaps, phrase rewrites, word-order changes, deletions). Use
  `use_model=True` / `--real-attacks` for real T5 / MarianMT / flan-t5 attacks (needs internet + PyTorch).
  Real model attacks are usually stronger; expect lower detection rates.
- **Fair synonym evaluation.** The attacker's synonym table (`attacks.py`) is written independently of the
  defender's synonym clusters (`lexicon.py`). Words the defender's clusters do not cover are *not* protected,
  so the Semantic-Anchor gain depends on lexicon coverage. Extending `_CLUSTERS` (or plugging in WordNet /
  embeddings) is the natural improvement.
- **The Hugging Face backend** (`HFBackend`) follows the standard `transformers` API, but this project was
  built and tested in an environment without internet or PyTorch, so it was exercised on the toy backend only.
  Run `pytest tests/` and `python run_demo.py --quick` on your machine to confirm.
- **Re-tokenisation.** The detector re-tokenises text, so on a real LLM a small share of tokens can merge
  differently than they were generated; expect slightly lower no-attack detection than on the toy model.
- **Detection vs AI-text classification.** The statistical detector tests for *this key's watermark*, and the
  ML classifier is trained on your own labelled examples. They are not general "is this AI-written" detectors.
- **Security.** Watermarks keyed by a single secret are not proof against an attacker who can query the
  detector many times; keep `SECRET_KEY` private and change it for real use.

---

## 9. Common problems

| Problem | Fix |
|---|---|
| `pip install` fails on `sentencepiece` | Install build tools: Windows -> "Visual C++ Build Tools"; Mac -> `xcode-select --install`; Linux -> `sudo apt install build-essential` |
| Warning "Falling back to the OFFLINE TOY language model" | PyTorch/transformers not installed, or no internet for the first model download. Install them and re-run, or ignore it for testing. |
| Model download slow / times out | Re-run; Hugging Face resumes partial downloads. |
| `streamlit: command not found` | Activate your virtual environment, then `pip install -r requirements.txt`. |
| Payload is not recovered | Use >= 100 tokens (ideally 120-200); very heavy rewriting can destroy too many votes. |
| Out of memory during generation | Lower `max_new_tokens` in the dashboard or `MAX_NEW_TOKENS` / `BENCHMARK_TOKENS` in `config.py`. |
| Want to start fresh | Delete the contents of `results/` and re-run. |

---

## 10. Project mapping (for your report)

| Synopsis component | Code location |
|---|---|
| Watermark Generator (Methods A-D) | `watermarklab/generator.py`, `scheme.py`, `methods.py` |
| Hybrid Statistical / ML Detector | `watermarklab/detector.py` |
| Attack / Robustness Engine | `watermarklab/attacks.py` |
| Analytics Dashboard | `app.py` |
| Benchmark / evaluation pipeline | `watermarklab/evaluate.py` |
| Basic statistical watermark (Method A) | `methods.py` -> `basic` (window = 1) |
| Context-aware watermark (Method B) | `methods.py` -> `context` (window = 4) |
| **New: Semantic-Anchor Watermark Recovery** | `methods.py` -> `anchor`; `lexicon.py`; `scheme.py`; `detector.py::_detect_greenlist` |
| **New: Adaptive Watermark Strength** | `methods.py` -> `adaptive`; `scheme.py::adaptive_delta`, `detector_weights` |
| **New: Error-Correcting Watermark** | `methods.py` -> `ecc`; `ecc.py`; `detector.py::_detect_ecc` |
| All three combined | `methods.py` -> `full` |
