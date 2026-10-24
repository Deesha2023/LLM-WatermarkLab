"""Core tests: hashing, green-list statistics, detection, attacks, benchmark plumbing. No model download needed."""

import numpy as np
import pytest

from watermarklab import config
from watermarklab.attacks import ATTACKS, synonym_replace_attack
from watermarklab.backend import get_backend
from watermarklab.detector import (detect_method, hybrid_detect, ml_detect, statistical_detect,
                                   train_ml_classifier, weighted_z)
from watermarklab.evaluate import run_full_benchmark, summarize_benchmark
from watermarklab.generator import (generate_basic_watermarked, generate_context_aware_watermarked,
                                    generate_plain, generate_watermarked)
from watermarklab.hashing import combine, key_to_int, unit_interval
from watermarklab.methods import METHODS, get_method
from watermarklab.scheme import WatermarkScheme
from watermarklab.utils import text_quality_score


# ---------------------------------------------------------------- hashing / scheme
def test_hash_is_deterministic():
    assert combine(1, 2, 3) == combine(1, 2, 3)
    assert combine(1, 2, 3) != combine(3, 2, 1)
    assert key_to_int("a") != key_to_int("b")


def test_green_list_is_deterministic():
    keys = get_backend().token_keys()
    a = unit_interval(12345, keys)
    b = unit_interval(12345, keys)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, unit_interval(54321, keys))


def test_green_fraction_matches_gamma():
    keys = get_backend().token_keys()
    for ctx in (1, 99, 2 ** 40 + 7):
        frac = (unit_interval(ctx, keys) < 0.5).mean()
        assert 0.4 < frac < 0.6


def test_z_score_positive_when_above_expected():
    hits = np.ones(100)
    assert weighted_z(hits, np.ones(100), 0.5) > 4
    assert weighted_z(np.zeros(100), np.ones(100), 0.5) < -4
    assert abs(weighted_z(np.array([1, 0] * 50, dtype=float), np.ones(100), 0.5)) < 1e-9


def test_different_keys_do_not_cross_detect(prompt):
    text = generate_watermarked(prompt, "basic", 120, seed=5).text
    assert detect_method(text, "basic")["is_watermarked"]
    assert not detect_method(text, "basic", key="some-other-secret")["is_watermarked"]


# ---------------------------------------------------------------- generation / detection
def test_registry_lookup():
    assert get_method("basic") is METHODS["basic"]
    assert get_method("Semantic-Anchor") is METHODS["anchor"]
    with pytest.raises(KeyError):
        get_method("nope")


def test_generation_is_reproducible(prompt):
    a = generate_watermarked(prompt, "basic", 60, seed=3).text
    b = generate_watermarked(prompt, "basic", 60, seed=3).text
    c = generate_watermarked(prompt, "basic", 60, seed=4).text
    assert a == b and a != c


@pytest.mark.parametrize("key", list(METHODS))
def test_every_method_detects_its_own_watermark(prompt, key):
    hits = sum(
        detect_method(generate_watermarked(prompt, key, 140, seed=200 + i).text, key)["is_watermarked"]
        for i in range(6))
    assert hits >= 5, f"{key} detected only {hits}/6"


@pytest.mark.parametrize("key", list(METHODS))
def test_no_false_positives_on_plain_text(prompt, key):
    fps = sum(detect_method(generate_plain(prompt, 140, seed=900 + i), key)["is_watermarked"]
              for i in range(12))
    assert fps == 0


def test_plain_text_z_scores_look_standard_normal(prompt):
    zs = [detect_method(generate_plain(prompt, 140, seed=1500 + i), "basic")["z_score"] for i in range(40)]
    assert abs(np.mean(zs)) < 0.6
    assert np.std(zs) < 1.6


def test_repeated_text_is_not_flagged():
    """Repeating the same sentence must not inflate the z-score (repeated pairs are ignored)."""
    text = ("Computers store data in memory. " * 12)
    assert not detect_method(text, "basic")["is_watermarked"]


def test_legacy_api_matches_original_signature(prompt):
    text = generate_basic_watermarked(prompt, max_new_tokens=100)
    r = statistical_detect(text)
    assert set(["is_watermarked", "confidence", "z_score"]) <= set(r)
    r4 = statistical_detect(generate_context_aware_watermarked(prompt, 100), context_window=4)
    assert r4["is_watermarked"]


def test_short_text_is_handled():
    r = detect_method("hi", "basic")
    assert r["is_watermarked"] is False


def test_confidence_is_centred_on_threshold():
    from watermarklab.detector import confidence_from_z
    thr = config.Z_SCORE_THRESHOLD
    assert abs(confidence_from_z(thr, thr) - 0.5) < 1e-12
    assert confidence_from_z(thr + 2, thr) > 0.8 > 0.2 > confidence_from_z(thr - 2, thr)


# ---------------------------------------------------------------- attacks
def test_every_attack_returns_text(prompt):
    text = generate_watermarked(prompt, "basic", 100, seed=8).text
    for name, fn in ATTACKS.items():
        out = fn(text)
        assert isinstance(out, str) and out.strip(), name


def test_attacks_are_deterministic(prompt):
    text = generate_watermarked(prompt, "basic", 100, seed=8).text
    for name, fn in ATTACKS.items():
        assert fn(text) == fn(text), name


def test_synonym_attack_changes_text(prompt):
    text = generate_watermarked(prompt, "basic", 100, seed=8).text
    assert synonym_replace_attack(text, fraction=0.9) != text
    assert synonym_replace_attack(text, fraction=0.0) == text


def test_attacks_handle_degenerate_input():
    for name, fn in ATTACKS.items():
        for t in ["", "Hello.", "one two three"]:
            assert isinstance(fn(t), str), (name, t)


def test_attacks_weaken_the_basic_watermark(prompt):
    drops = []
    for i in range(6):
        text = generate_watermarked(prompt, "basic", 140, seed=300 + i).text
        z0 = detect_method(text, "basic")["z_score"]
        z1 = detect_method(ATTACKS["LLM Rewriting"](text), "basic")["z_score"]
        drops.append(z0 - z1)
    assert np.mean(drops) > 1.0


def test_text_quality_score_bounds():
    assert text_quality_score("a b c d e", "a b c d e") == pytest.approx(1.0)
    assert text_quality_score("alpha beta gamma", "x y z") == 0.0
    assert 0 < text_quality_score("the cat sat on the mat", "the cat sat on a rug") < 1


# ---------------------------------------------------------------- ML detector + benchmark
def test_ml_classifier_trains_and_predicts(tmp_path, prompt):
    path = str(tmp_path / "clf.joblib")
    wm = [generate_watermarked(prompt, "basic", 100, seed=40 + i).text for i in range(6)]
    pl = [generate_plain(prompt, 100, seed=80 + i) for i in range(6)]
    train_ml_classifier(wm, pl, path=path)
    config_path = config.ML_CLASSIFIER_PATH
    config.ML_CLASSIFIER_PATH = path
    try:
        p_wm = ml_detect(wm[0])["confidence"]
        p_pl = ml_detect(pl[0])["confidence"]
        assert p_wm > p_pl
        h = hybrid_detect(wm[0], "basic")
        assert "ml_probability" in h and h["is_watermarked"]
    finally:
        config.ML_CLASSIFIER_PATH = config_path


def test_benchmark_shape_and_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", str(tmp_path))
    df = run_full_benchmark(methods=["basic", "ecc"], prompts=["Describe a sunny day."],
                            n_samples=2, max_new_tokens=100)
    expected = len(ATTACKS) + 2          # every attack + "No attack" + control
    assert len(df) == 2 * 2 * expected
    assert {"method", "attack", "detected", "z_score", "text_quality", "payload_ok"} <= set(df.columns)
    s = summarize_benchmark(df)
    assert {"detection_accuracy", "avg_text_quality", "payload_recovery"} <= set(s.columns)
    ctrl = s[s.attack == "Control: no watermark"]
    assert (ctrl.detection_accuracy == 0).all()           # no false positives
    assert (tmp_path / "benchmark.csv").exists()
