"""Tests for the three new features: Semantic-Anchor, Adaptive Strength, Error-Correcting watermark."""

import itertools

import numpy as np
import pytest

from watermarklab import config, lexicon
from watermarklab.attacks import ATTACKS
from watermarklab.detector import _combine_layers, detect_method
from watermarklab.ecc import (ErrorCorrectingCode, bits_to_int, crc_bits, hamming74_decode_block,
                              hamming74_encode_nibble, int_to_bits)
from watermarklab.generator import generate_plain, generate_watermarked
from watermarklab.methods import METHODS
from watermarklab.scheme import WatermarkScheme


# =============================================================== Error-Correcting
def test_int_bit_roundtrip():
    for v in (0, 1, 178, 255):
        assert bits_to_int(int_to_bits(v, 8)) == v


def test_hamming_corrects_every_single_bit_error():
    for nib in itertools.product([0, 1], repeat=4):
        cw = hamming74_encode_nibble(nib)
        for pos in range(7):
            bad = list(cw)
            bad[pos] ^= 1
            data, flips, ok = hamming74_decode_block(bad)
            assert ok and data == list(nib) and flips == 1


def test_hamming_recovers_up_to_two_erasures():
    cw = hamming74_encode_nibble([1, 0, 1, 1])
    for e in itertools.combinations(range(7), 2):
        rec = [None if i in e else b for i, b in enumerate(cw)]
        data, _, ok = hamming74_decode_block(rec)
        assert ok and data == [1, 0, 1, 1]


def test_ecc_roundtrip_all_payloads():
    code = ErrorCorrectingCode(8, 4)
    for p in range(256):
        out = code.decode(code.encode(p))
        assert out["payload"] == p and out["crc_ok"]


def test_crc_flags_corruption():
    code = ErrorCorrectingCode(8, 4)
    cw = code.encode(77)
    # flip two bits in the same block: Hamming cannot fix it, so CRC must object (or decode differs)
    cw[0] ^= 1
    cw[1] ^= 1
    out = code.decode(cw)
    assert not (out["crc_ok"] and out["payload"] == 77)


def test_soft_decoder_survives_noisy_votes():
    code = ErrorCorrectingCode(8, 4)
    rng = np.random.default_rng(0)
    ok = 0
    for p in range(0, 256, 5):
        cw = np.array(code.encode(p))
        llr = (2 * cw - 1) * rng.uniform(1, 4, len(cw))
        flip = rng.choice(len(cw), 3, replace=False)           # 3 of 21 slots vote the wrong way
        llr[flip] *= -0.4
        ok += code.decode_ml(llr)["payload"] == p
    assert ok >= 0.9 * len(range(0, 256, 5))


def test_ecc_watermark_recovers_payload(prompt):
    ok = 0
    for i in range(10):
        payload = (37 * i + 11) % 256
        g = generate_watermarked(prompt, "ecc", 140, payload=payload, seed=70 + i)
        d = detect_method(g.text, "ecc")
        ok += d["payload"] == payload
    assert ok >= 9


@pytest.mark.parametrize("attack", ["Random Word Deletion", "Synonym Replacement", "Sentence Restructuring"])
def test_ecc_payload_survives_attacks(prompt, attack):
    ok = 0
    for i in range(10):
        payload = (53 * i + 5) % 256
        g = generate_watermarked(prompt, "ecc", 160, payload=payload, seed=120 + i)
        ok += detect_method(ATTACKS[attack](g.text), "ecc")["payload"] == payload
    assert ok >= 6, f"{attack}: only {ok}/10 payloads survived"


def test_ecc_does_not_hallucinate_payload_on_plain_text(prompt):
    flagged = 0
    for i in range(20):
        d = detect_method(generate_plain(prompt, 140, seed=2000 + i), "ecc")
        flagged += d["is_watermarked"]
    assert flagged == 0


def test_ecc_corrected_bits_reported(prompt):
    g = generate_watermarked(prompt, "ecc", 140, payload=200, seed=5)
    text = ATTACKS["Paraphrasing"](g.text)
    d = detect_method(text, "ecc")
    assert "corrected_bits" in d and "slot_votes" in d and len(d["slot_votes"]) == 21


# =============================================================== Adaptive strength
def test_adaptive_delta_grows_with_entropy():
    s = WatermarkScheme(METHODS["adaptive"])
    ds = [s.adaptive_delta(h, 0, 0) for h in np.linspace(0, 5, 30)]
    assert all(b >= a - 1e-12 for a, b in zip(ds, ds[1:]))
    assert ds[0] == pytest.approx(config.ADAPTIVE_DELTA_MIN)
    assert ds[-1] == pytest.approx(config.ADAPTIVE_DELTA_MAX)


def test_adaptive_feedback_boosts_when_behind():
    s = WatermarkScheme(METHODS["adaptive"])
    normal = s.adaptive_delta(2.0, hits=20, total=20)
    behind = s.adaptive_delta(2.0, hits=5, total=20)
    assert behind > normal
    assert behind <= 1.5 * config.ADAPTIVE_DELTA_MAX + 1e-9


def test_adaptive_generation_bias_tracks_uncertainty(prompt):
    g = generate_watermarked(prompt, "adaptive", 200, seed=11)
    ent = np.array([t["entropy"] for t in g.trace])
    delta = np.array([t["delta"] for t in g.trace])
    assert np.corrcoef(ent, delta)[0, 1] > 0.8
    assert delta.min() >= config.ADAPTIVE_DELTA_MIN - 1e-9


def test_adaptive_beats_basic_at_matched_fluency():
    """
    Compare at equal FLUENCY COST (perplexity), not equal bias: adaptive spends bias where the model
    is uncertain, so it reaches a stronger z-score for the same damage to the text.
    (Basic with delta=2.2 was calibrated to roughly the same perplexity as the adaptive defaults.)
    """
    from watermarklab.evaluate import load_prompts
    from watermarklab.utils import perplexity
    prompts = load_prompts()[:8]
    z_a, z_b, p_a, p_b = [], [], [], []
    for i, pr in enumerate(prompts):
        for s in range(2):
            seed = 400 + 7 * i + s
            a = generate_watermarked(pr, "adaptive", 140, seed=seed)
            b = generate_watermarked(pr, METHODS["basic"].with_(delta=2.2), 140, seed=seed)
            z_a.append(detect_method(a.text, "adaptive")["z_score"])
            z_b.append(detect_method(b.text, "basic")["z_score"])
            p_a.append(perplexity(a.text))
            p_b.append(perplexity(b.text))
    assert np.mean(p_a) <= 1.10 * np.mean(p_b)           # not buying strength with extra distortion
    assert np.mean(z_a) > np.mean(z_b) + 0.3             # yet clearly stronger evidence


def test_detector_weights_are_bounded(prompt):
    s = WatermarkScheme(METHODS["adaptive"])
    ids = s.backend.encode(generate_watermarked(prompt, "adaptive", 80, seed=2).text)
    w = s.detector_weights(ids)
    assert len(w) == len(ids)
    assert w.min() >= config.ADAPTIVE_WEIGHT_FLOOR - 1e-9 and w.max() <= 1.0 + 1e-9


# =============================================================== Semantic anchors
def test_synonyms_share_a_canonical_label():
    assert lexicon.canonical_label("large") == lexicon.canonical_label("huge") == lexicon.canonical_label("big")
    assert lexicon.canonical_label("running") == lexicon.canonical_label("runs")
    assert lexicon.canonical_label("computer") != lexicon.canonical_label("river")
    assert lexicon.canonical_label("!") == ""


def test_anchor_definition():
    assert lexicon.is_anchor("computer") and lexicon.is_anchor("Learning")
    assert not lexicon.is_anchor("the") and not lexicon.is_anchor("with") and not lexicon.is_anchor("is")


def _swap_in_cluster(text: str) -> str:
    """Replace words by a different member of their defender synonym cluster."""
    import re
    mates = {}
    for group in lexicon._CLUSTERS:
        for w in group:
            others = [o for o in group if o != w and len(o) >= 4]
            if others and len(w) >= 4:
                mates.setdefault(w, others[0])
    return re.sub(r"[A-Za-z]+", lambda m: mates.get(m.group(0).lower(), m.group(0)), text)


def test_in_cluster_synonym_swaps_leave_anchor_scores_unchanged(prompt):
    """Synonym clusters make the anchor watermark immune to swaps INSIDE a cluster."""
    found = 0
    for i in range(10):
        text = generate_watermarked(prompt, "anchor", 140, seed=600 + i).text
        swapped = _swap_in_cluster(text)
        if swapped.lower() == text.lower():
            continue
        found += 1
        a, b = detect_method(text, "anchor"), detect_method(swapped, "anchor")
        assert b["z_score"] == pytest.approx(a["z_score"], abs=1e-9)
    assert found >= 3


def test_basic_watermark_is_not_immune_to_the_same_swaps(prompt):
    drops = []
    for i in range(10):
        text = generate_watermarked(prompt, "basic", 140, seed=600 + i).text
        swapped = _swap_in_cluster(text)
        if swapped.lower() != text.lower():
            drops.append(detect_method(text, "basic")["z_score"] - detect_method(swapped, "basic")["z_score"])
    assert drops and np.mean(drops) > 0.5


def test_recovery_statistic_kicks_in_when_context_layer_is_destroyed():
    z, used = _combine_layers(z_context=0.3, z_anchor=4.8, n_anchor=25)
    assert used and z >= 4.8
    z2, used2 = _combine_layers(z_context=5.0, z_anchor=0.1, n_anchor=25)
    assert z2 == pytest.approx(5.0) and not used2
    z3, used3 = _combine_layers(z_context=1.0, z_anchor=9.0, n_anchor=2)     # too few anchors: no recovery
    assert z3 == 1.0 and not used3


def test_anchor_layer_survives_when_context_is_scrambled(prompt):
    """Shuffling every word destroys all context seeds; the context-free anchor layer must still show signal."""
    import random
    anchor_zs, ctx_zs = [], []
    for i in range(10):
        text = generate_watermarked(prompt, "anchor", 160, seed=700 + i).text
        words = text.split()
        random.Random(i).shuffle(words)
        d = detect_method(" ".join(words), "anchor")
        anchor_zs.append(d["z_anchor_layer"])
        ctx_zs.append(d["z_context_layer"])
    assert np.mean(anchor_zs) > 1.5          # still clearly above chance
    assert np.mean(anchor_zs) > np.mean(ctx_zs)


def test_anchor_method_reports_layers(prompt):
    d = detect_method(generate_watermarked(prompt, "anchor", 140, seed=1).text, "anchor")
    assert {"z_context_layer", "z_anchor_layer", "anchors_scored", "recovery_used"} <= set(d)


# =============================================================== Combined method
def test_full_method_recovers_payload_and_detects(prompt):
    ok = det = 0
    for i in range(8):
        payload = (91 * i + 3) % 256
        g = generate_watermarked(prompt, "full", 140, payload=payload, seed=800 + i)
        d = detect_method(g.text, "full")
        ok += d["payload"] == payload
        det += d["is_watermarked"]
    assert ok >= 7 and det >= 7
