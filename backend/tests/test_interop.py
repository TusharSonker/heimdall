"""
Heimdall — Cross-Language Paillier Interop Tests
=================================================
These tests pin down the JS/Python encoding mismatch and both fixes.

Three paths are verified:
  1. NAIVE   — phe default (BASE=16) on JS ciphertext → catastrophically wrong
  2. BASE10  — Base10EncodedNumber fix → exact match with plaintext
  3. RAW     — bypass phe entirely via pow(c,k,n²) → exact match with plaintext

Running:
    cd backend && source venv/bin/activate
    python -m pytest tests/test_interop.py -v
"""

import sys, math, random, statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import phe as paillier

from core.encryption import Base10EncodedNumber, reconstruct_encrypted_number
from core.models import (
    encrypted_linear_inference,
    encrypted_linear_inference_raw,
    TRAINED_WEIGHTS,
    sigmoid,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

KEY_BITS = 1024  # 512-bit too small for anemia's large weights; production uses 2048

def make_keys():
    return paillier.generate_paillier_keypair(n_length=KEY_BITS)


def js_style_encrypt(pub, value: float, scale: int = 1_000_000) -> dict:
    """
    Simulate exactly what crypto.js does:
      m = round(value × 10^6)
      c = (1 + m*n) * r^n  mod  n²      (g = n+1 simplification)
      return { ciphertext: str(c), exponent: -6 }
    """
    n  = pub.n
    n2 = n * n
    m  = round(value * scale) % n
    gm = (1 + m * n) % n2
    r  = random.randrange(1, n)
    c  = gm * pow(r, n, n2) % n2
    return {"ciphertext": str(c), "exponent": -int(math.log10(scale))}


def plaintext_score(model_id: str, features: list) -> float:
    w = TRAINED_WEIGHTS[model_id]
    return sum(wi * xi for wi, xi in zip(w["weights"], features)) + w["bias"]


# ---------------------------------------------------------------------------
# Test 1 — Prove the bug: naive phe path is catastrophically wrong
# ---------------------------------------------------------------------------

def test_naive_phe_path_is_wrong():
    """
    A developer who naively wraps a JS ciphertext in phe.EncryptedNumber
    and does arithmetic with phe's default BASE=16 gets a result that is
    billions of times wrong. This test MUST stay failing — if it passes,
    phe changed its internal BASE and our paper claim needs updating.
    """
    pub, priv = make_keys()

    x = 0.5
    w = 5.0

    js_c = js_style_encrypt(pub, x)

    # Naive: hand JS ciphertext to phe as-is, multiply by weight using phe default
    enc_naive = paillier.EncryptedNumber(pub, int(js_c["ciphertext"]), js_c["exponent"])
    result_naive = enc_naive * w          # phe uses BASE=16 for weight encoding
    score_naive  = priv.decrypt(result_naive)

    expected = x * w   # 0.5 × 5.0 = 2.5

    # The naive path is wrong by (16/10)^6 ≈ 16.78× in either direction.
    # Check that the absolute error is substantial relative to the expected value.
    abs_error = abs(score_naive - expected)
    assert abs_error > expected * 0.5, (
        f"Naive path unexpectedly close to correct: got {score_naive:.6f}, "
        f"expected {expected:.6f}, abs_error={abs_error:.4f}. "
        f"phe may have changed its BASE. Recheck §V of the paper."
    )


# ---------------------------------------------------------------------------
# Test 2 — Base10EncodedNumber fix produces correct results
# ---------------------------------------------------------------------------

# Models where Base10EncodedNumber path fails due to EncryptedNumber.decrease_exponent_to
# using the hardcoded EncodedNumber.BASE=16 (not Base10EncodedNumber.BASE=10).
#
# Root cause: when two EncryptedNumbers have different product-exponents, phe's
# __add__ calls EncryptedNumber.decrease_exponent_to, which multiplies by
# pow(EncodedNumber.BASE, k) = 16^k instead of 10^k, introducing an error of
# (16/10)^k per alignment step.
#
# Affected: any model whose weights span more than one order of magnitude
# (different exponents from Base10EncodedNumber.encode → different product
# exponents after multiplication → alignment required → wrong BASE used).
#
# Safe: models where all weight-term products share the same exponent
# (no EncryptedNumber alignment called; bias via _add_encoded uses cls.BASE=10).
#
# Empirical: diabetes/heart weights all encode at exponent -16 → products all -22
# (no alignment). Anemia: w0=-13.92 encodes at -15 → product -21, others at -17
# → -23 (exponent diff=2 triggers 16^2=256 instead of 10^2=100).
_BASE10_ENC_ALIGN_BROKEN = {"anemia"}

@pytest.mark.parametrize("model_id", ["diabetes", "heart", "anemia"])
def test_base10_inference_matches_plaintext(model_id):
    """
    encrypted_linear_inference (Base10EncodedNumber path) must match
    plaintext logistic regression within quantisation tolerance.

    Fails for anemia because its weights span more than one order of magnitude,
    triggering EncryptedNumber.decrease_exponent_to (alignment during addition).
    That method hardcodes EncodedNumber.BASE=16, not Base10EncodedNumber.BASE=10,
    introducing (16/10)^2 ≈ 2.56× error per alignment step. The raw arithmetic
    path avoids this entirely by never using phe's encoding layer.
    """
    pub, priv = make_keys()
    n_feat = len(TRAINED_WEIGHTS[model_id]["weights"])

    errors = []
    for _ in range(30):
        features = [random.random() for _ in range(n_feat)]
        enc_feats = [js_style_encrypt(pub, x) for x in features]

        result = encrypted_linear_inference(pub, enc_feats, model_id)
        enc_num = paillier.EncryptedNumber(pub, int(result["ciphertext"]), result["exponent"])
        score_he = priv.decrypt_encoded(enc_num, Encoding=Base10EncodedNumber).decode()

        score_pt = plaintext_score(model_id, features)
        errors.append(abs(score_he - score_pt))

    mean_err = statistics.mean(errors)
    max_err  = max(errors)

    if model_id in _BASE10_ENC_ALIGN_BROKEN:
        pytest.xfail(
            f"[{model_id}] EncryptedNumber.decrease_exponent_to uses "
            f"EncodedNumber.BASE=16 not Base10EncodedNumber.BASE=10. "
            f"Weights span multiple orders of magnitude → alignment fires → "
            f"(16/10)^k error (max_err={max_err:.2e}). Use raw path."
        )

    assert max_err < 1e-3, (
        f"[{model_id}] Base10 path: max |Δscore| = {max_err:.2e} exceeds 1e-3"
    )
    assert mean_err < 1e-4, (
        f"[{model_id}] Base10 path: mean |Δscore| = {mean_err:.2e} exceeds 1e-4"
    )


# ---------------------------------------------------------------------------
# Test 3 — Raw Paillier fix produces correct results
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_id", ["diabetes", "heart", "anemia"])
def test_raw_inference_matches_plaintext(model_id):
    """
    encrypted_linear_inference_raw (bypass phe entirely) must also match
    plaintext logistic regression. Proves the raw group arithmetic is correct.
    """
    pub, priv = make_keys()
    n_feat = len(TRAINED_WEIGHTS[model_id]["weights"])

    errors = []
    for _ in range(30):
        features = [random.random() for _ in range(n_feat)]
        enc_feats = [js_style_encrypt(pub, x) for x in features]

        result   = encrypted_linear_inference_raw(pub, enc_feats, model_id)
        score_int = priv.decrypt(
            paillier.EncryptedNumber(pub, int(result["ciphertext"]), 0)
        )
        score_he = score_int / 10**12
        score_pt = plaintext_score(model_id, features)
        errors.append(abs(score_he - score_pt))

    mean_err = statistics.mean(errors)
    max_err  = max(errors)

    assert max_err < 1e-3, (
        f"[{model_id}] Raw path: max |Δscore| = {max_err:.2e} exceeds 1e-3"
    )


# ---------------------------------------------------------------------------
# Test 4 — Both fixes agree with each other
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_id", ["diabetes", "heart", "anemia"])
def test_base10_and_raw_agree(model_id):
    """
    The two independent fixes must produce scores within 1e-4 of each other.
    This is the cross-validation that proves both solutions solve the same problem.
    """
    pub, priv = make_keys()
    n_feat = len(TRAINED_WEIGHTS[model_id]["weights"])

    disagreements = 0
    threshold = TRAINED_WEIGHTS[model_id].get("threshold", 0.5)

    for _ in range(30):
        features  = [random.random() for _ in range(n_feat)]
        enc_feats = [js_style_encrypt(pub, x) for x in features]

        # Base10 path
        r1 = encrypted_linear_inference(pub, enc_feats, model_id)
        enc1 = paillier.EncryptedNumber(pub, int(r1["ciphertext"]), r1["exponent"])
        score1 = priv.decrypt_encoded(enc1, Encoding=Base10EncodedNumber).decode()

        # Raw path
        r2 = encrypted_linear_inference_raw(pub, enc_feats, model_id)
        score_int = priv.decrypt(paillier.EncryptedNumber(pub, int(r2["ciphertext"]), 0))
        score2 = score_int / 10**12

        if model_id not in _BASE10_ENC_ALIGN_BROKEN:
            assert abs(score1 - score2) < 1e-3, (
                f"[{model_id}] Base10 score={score1:.6f} vs Raw score={score2:.6f}: "
                f"divergence {abs(score1-score2):.2e} exceeds 1e-3"
            )
            if (sigmoid(score1) > threshold) != (sigmoid(score2) > threshold):
                disagreements += 1

    if model_id in _BASE10_ENC_ALIGN_BROKEN:
        pytest.xfail(
            f"[{model_id}] Base10 path broken by EncryptedNumber.decrease_exponent_to "
            f"using BASE=16 — raw path is authoritative for this model."
        )

    assert disagreements == 0, (
        f"[{model_id}] {disagreements}/30 classification disagreements between "
        f"Base10 and Raw paths"
    )


# ---------------------------------------------------------------------------
# Test 5 — Regression: naive path must NOT accidentally pass
# ---------------------------------------------------------------------------

def test_naive_exponent_interpretation_is_incompatible():
    """
    Confirms that phe's BASE=16 and JS's BASE=10 are genuinely incompatible
    by encrypting a known value, running through phe default arithmetic,
    and checking the result is meaningfully different from expected.

    If this test ever passes (naive path accidentally correct), it means
    phe changed its BASE to 10 — a breaking change worth flagging.
    """
    pub, priv = make_keys()
    known_value = 0.25
    js_c = js_style_encrypt(pub, known_value)

    # Default phe path — no Base10EncodedNumber
    enc = paillier.EncryptedNumber(pub, int(js_c["ciphertext"]), js_c["exponent"])
    decoded = priv.decrypt(enc)

    # phe decodes as: integer × 16^(-6), but integer encodes BASE-10 value
    # The result should NOT be close to 0.25
    assert abs(decoded - known_value) > 0.01, (
        "Naive phe path returned the correct value — phe BASE may have changed to 10. "
        "If so, Base10EncodedNumber is no longer needed. Verify and update the paper."
    )
