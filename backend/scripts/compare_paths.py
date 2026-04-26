"""
Heimdall — Head-to-Head Comparison: Raw Path vs Base10 Path
=============================================================

Old path:  encrypted_linear_inference_raw()
           Bypasses phe entirely. Works in raw Z_{n²} modular arithmetic.
           Integer weights, integer bias, no phe encoding at all.

New path:  encrypted_linear_inference()  [with Base10EncodedNumber]
           Uses phe fully but with BASE=10 subclass.
           Float weights, float bias, all encoded/decoded via phe.

Both receive the same JS-style ciphertexts (simulated via js_style_encrypt).
Both return an encrypted result that the client decrypts.

We compare:
  - Decrypted scores  (should be numerically identical or within float precision)
  - Risk decisions    (HIGH/LOW) — must be 100% identical
  - Max deviation     (should be < 1e-4, ideally < 1e-5)
"""

import sys
import math
import random
import secrets
import statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import phe as paillier
from core.encryption import Base10EncodedNumber, decrypt_value
from core.models import (
    encrypted_linear_inference,       # NEW: phe + Base10
    encrypted_linear_inference_raw,   # OLD: bypass phe
    TRAINED_WEIGHTS,
    MODEL_SPECS,
    sigmoid,
)

# ── Replicate crypto.js encryptValue() exactly ─────────────────────────────
FEATURE_SCALE = 1_000_000

def js_style_encrypt(pub, value):
    n  = pub.n
    n2 = pub.nsquare
    m  = round(value * FEATURE_SCALE) % n
    gm = (1 + m * n) % n2
    r  = secrets.randbelow(n) or 1
    c  = gm * pow(r, n, n2) % n2
    return {"ciphertext": str(c), "exponent": -6}

# ── Decrypt for old raw path (divides by 10^12 hardcoded) ──────────────────
def decrypt_raw_result(priv, enc_result):
    """Old path returns exponent=-12 always (hardcoded TOTAL_SCALE=1e12)."""
    pub   = priv.public_key
    n     = pub.n
    raw   = priv.raw_decrypt(int(enc_result["ciphertext"]))
    if raw > n // 2:
        raw -= n
    exponent = enc_result["exponent"]   # always -12 for raw path
    return raw / (10 ** (-exponent))

# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_paths(n_samples=500, key_bits=1024):
    print("=" * 65)
    print("  Heimdall — Raw Path vs Base10 Path Comparison")
    print(f"  Samples per model: {n_samples}  |  Key bits: {key_bits}")
    print("=" * 65)

    pub, priv = paillier.generate_paillier_keypair(n_length=key_bits)
    print(f"\n  Keypair generated ({key_bits}-bit).\n")

    total_mismatches = 0
    grand_deltas     = []

    for mid in MODEL_SPECS:
        w       = TRAINED_WEIGHTS[mid]
        n_feat  = len(w["weights"])
        thresh  = w.get("threshold", 0.5)

        score_deltas    = []
        prob_deltas     = []
        decision_mismatches = 0

        for i in range(n_samples):
            x = [random.random() for _ in range(n_feat)]

            # Same ciphertexts sent to both paths
            js_ciphers = [js_style_encrypt(pub, xi) for xi in x]

            # ── OLD path (raw modular arithmetic) ──────────────────────────
            enc_raw    = encrypted_linear_inference_raw(pub, js_ciphers, mid)
            score_old  = decrypt_raw_result(priv, enc_raw)

            # ── NEW path (phe + Base10EncodedNumber) ───────────────────────
            enc_new    = encrypted_linear_inference(pub, js_ciphers, mid)
            score_new  = decrypt_value(priv, enc_new)

            # ── Plaintext reference ────────────────────────────────────────
            plain = sum(wi * xi for wi, xi in zip(w["weights"], x)) + w["bias"]

            # Compute deviations
            score_delta = abs(score_old - score_new)
            score_deltas.append(score_delta)
            grand_deltas.append(score_delta)

            prob_old = sigmoid(score_old)
            prob_new = sigmoid(score_new)
            prob_deltas.append(abs(prob_old - prob_new))

            risk_old = prob_old > thresh
            risk_new = prob_new > thresh
            if risk_old != risk_new:
                decision_mismatches += 1
                total_mismatches    += 1

            # Spot-check first sample in detail
            if i == 0:
                print(f"  [{mid}] Sample #1 detail:")
                print(f"    Input x           = {[round(xi, 4) for xi in x]}")
                print(f"    Plaintext score   = {plain:.8f}")
                print(f"    OLD (raw) score   = {score_old:.8f}  |Δ_plain| = {abs(score_old - plain):.2e}")
                print(f"    NEW (b10) score   = {score_new:.8f}  |Δ_plain| = {abs(score_new - plain):.2e}")
                print(f"    |OLD - NEW|       = {score_delta:.2e}")
                print(f"    OLD prob          = {prob_old*100:.4f}%")
                print(f"    NEW prob          = {prob_new*100:.4f}%")
                print(f"    OLD risk          = {'HIGH' if risk_old else 'LOW'}")
                print(f"    NEW risk          = {'HIGH' if risk_new else 'LOW'}")
                print()

        mean_sd = statistics.mean(score_deltas)
        max_sd  = max(score_deltas)
        mean_pd = statistics.mean(prob_deltas)
        max_pd  = max(prob_deltas)

        print(f"  [{mid}] Summary over {n_samples} samples:")
        print(f"    Score deviation   mean = {mean_sd:.2e}   max = {max_sd:.2e}")
        print(f"    Prob  deviation   mean = {mean_pd:.2e}   max = {max_pd:.2e}")
        print(f"    Decision mismatches = {decision_mismatches}")
        print()

    print("─" * 65)
    print(f"  Grand total across all models ({n_samples * len(MODEL_SPECS)} samples):")
    print(f"    Overall mean |OLD-NEW| = {statistics.mean(grand_deltas):.2e}")
    print(f"    Overall max  |OLD-NEW| = {max(grand_deltas):.2e}")
    print(f"    Total decision mismatches = {total_mismatches}")
    if total_mismatches == 0:
        print("\n  VERDICT: Both paths produce IDENTICAL risk decisions.")
        print("  Score deviations are at floating-point quantisation level only.")
    else:
        print(f"\n  VERDICT: {total_mismatches} decision mismatches found — investigate!")
    print("=" * 65)


if __name__ == "__main__":
    compare_paths(n_samples=500, key_bits=1024)
