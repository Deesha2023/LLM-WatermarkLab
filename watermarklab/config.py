"""
config.py
---------
Every tunable setting for LLM-WatermarkLab lives here.
"""

# --------------------------------------------------------------------------
# Model / backend
# --------------------------------------------------------------------------
BASE_MODEL_NAME = "distilgpt2"      # swap for a bigger HF causal LM if you have a GPU
import os as _os
BACKEND = _os.environ.get("WATERMARKLAB_BACKEND", "auto")   # can also be set from the shell
                                    # "auto" = try Hugging Face, fall back to the offline toy LM
                                    # "hf"   = Hugging Face only (raise if unavailable)
                                    # "toy"  = offline toy language model only (tests / no internet)
MAX_NEW_TOKENS = 80
TEMPERATURE = 0.9
TOP_K = 50
SEED = 1234                         # makes generation reproducible

# --------------------------------------------------------------------------
# Core watermark (green-list) parameters
# --------------------------------------------------------------------------
SECRET_KEY = "llm-watermarklab-secret-key"
WATERMARK_STRENGTH = 2.0            # delta: logit bonus given to green tokens
GREEN_FRACTION = 0.5                # gamma: share of vocabulary that is "green"
Z_SCORE_THRESHOLD = 4.0             # z above this => "watermarked"

BASIC_CONTEXT_WINDOW = 1            # Method A: seed on the previous 1 token
CONTEXT_AWARE_WINDOW = 4            # Method B (proposed): seed on previous 4 tokens

# --------------------------------------------------------------------------
# Feature 1: Semantic-Anchor Watermark Recovery
# --------------------------------------------------------------------------
ANCHOR_MIN_LEN = 4                  # a word needs >= this many letters to be an anchor
ANCHOR_GREEN_FRACTION = 0.5         # gamma of the context-seeded layer
ANCHOR_LAYER_DELTA = 1.0            # extra bias on "green" anchor words (context-free layer)
ANCHOR_LAYER_GAMMA = 0.5            # share of anchor-word clusters that are green in that layer
ANCHOR_CONTEXT_DELTA = 2.0          # bias of the context-seeded layer in the anchor method

# --------------------------------------------------------------------------
# Feature 2: Adaptive Watermark Strength
# --------------------------------------------------------------------------
ADAPTIVE_DELTA_MIN = 0.2            # bias on very predictable (low-entropy) steps
ADAPTIVE_DELTA_MAX = 3.0            # bias on open-ended (high-entropy) steps
ADAPTIVE_ENTROPY_LOW = 0.6          # entropy (nats) at/below which we use DELTA_MIN
ADAPTIVE_ENTROPY_HIGH = 3.0         # entropy (nats) at/above which we use DELTA_MAX
ADAPTIVE_FEEDBACK_TARGET = 0.70     # desired running share of tokens that landed in the target set
ADAPTIVE_FEEDBACK_GAIN = 1.0        # how hard to push when the running share falls behind
ADAPTIVE_WEIGHT_FLOOR = 0.15        # detector: minimum weight of a low-entropy token

# --------------------------------------------------------------------------
# Feature 3: Error-Correcting Watermark (multi-bit payload)
# --------------------------------------------------------------------------
ECC_PAYLOAD_BITS = 8                # bits of hidden message (e.g. an author / model ID)
ECC_DEFAULT_PAYLOAD = 0b10110010    # payload used by the demo and benchmark
ECC_CRC_BITS = 4                    # integrity checksum appended before encoding
ECC_MIN_TOKENS = 100                # recommended minimum text length for reliable decoding

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
RESULTS_DIR = "results"
PROMPTS_FILE = "data/prompts.txt"
ML_CLASSIFIER_PATH = "results/ml_classifier.joblib"

# --------------------------------------------------------------------------
# Benchmark
# --------------------------------------------------------------------------
BENCHMARK_TOKENS = 120              # long enough for the payload (ECC) methods to be fair
