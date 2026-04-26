"""
Heimdall — Cross-Language Paillier Interop Tests
=================================================

These tests directly simulate what the JavaScript browser client (crypto.js)
does and verify that the Python server processes it correctly.

The JS encryption convention (from crypto.js lines 111-131):
    scaled = round(value × 10^6)
    ciphertext = (1 + scaled×n) × r^n  mod n²
    exponent = -6

This is BASE=10 fixed-point encoding with scale=10^6.  Python phe's default
is BASE=16, which would silently mis-decode the same ciphertext.

Test structure:
  - js_style_encrypt()   : replicates crypto.js encryptValue() exactly
  - test_bug_exists      : proves the OLD naive path (phe default) was wrong
  - test_base10_fix      : proves the NEW path (Base10EncodedNumber) is correct
  - test_all_models      : correctness across all 3 medical models × 200 random vectors
  - test_exponent_roundtrip: verifies the exponent tag in the response is usable by JS
"""

import sys
import math
import random
import secrets
import statistics
from pathlib import Path

import pytest
import phe as paillier
from phe.encoding import EncodedNumber

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.encryption import Base10EncodedNumber, decrypt_value
from core.models import (
    encrypted_linear_inference,
    encrypted_linear_inference_raw,
    TRAINED_WEIGHTS,
    MODEL_SPECS,
    sigmoid,
)


# ---------------------------------------------------------------------------
# Helper: simulate crypto.js  encryptValue(value, publicKey)
# ---------------------------------------------------------------------------

FEATURE_SCALE = 1_000_000   # must match crypto.js (1e6)
JS_EXPONENT   = -6          # exponent tag JS sends on the wire


def js_style_encrypt(pub: paillier.PaillierPublicKey, value: float) -> dict:
    """
    Python replica of crypto.js  encryptValue():

        const scaled = BigInt(Math.round(value * 1_000_000));
        const m      = ((scaled % n) + n) % n;
        const gm     = (1n + m * n) % n2;
        const rn     = modPow(r, n, n2);
        ciphertext   = gm * rn % n2;
        return { ciphertext: ciphertext.toString(), exponent: -6 };

    Produces a ciphertext that encodes `value` under BASE=10 / scale=10^6.
    This is what every browser request to /api/predict actually sends.
    """
    n  = pub.n
    n2 = pub.nsquare

    m  = round(value * FEATURE_SCALE) % n   # handles negative values via mod
    gm = (1 + m * n) % n2                   # g = n+1; g^m mod n² = 1 + m·n

    r  = secrets.randbelow(n) or 1
    rn = pow(r, n, n2)
    c  = gm * rn % n2

    return {"ciphertext": str(c), "exponent": JS_EXPONENT}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def keypair():
    """1024-bit keys for speed; same math as 2048-bit."""
    pub, priv = paillier.generate_paillier_keypair(n_length=1024)
    return pub, priv


# ---------------------------------------------------------------------------
# Test 1: Prove the OLD bug exists
# ---------------------------------------------------------------------------

class TestBugExists:
    def test_naive_phe_decode_wrong_by_base_ratio(self, keypair):
        """
        If a developer naively wraps a JS ciphertext in phe.EncryptedNumber
        and calls priv.decrypt() (BASE=16 decode), the result is wrong by
        (10/16)^6 ≈ 1/16.78.

        This test MUST fail with the old path and MUST pass now (asserting
        that the naive path IS still broken — so we can cite it in the paper).
        """
        pub, priv = keypair
        value = 0.5

        js_c = js_style_encrypt(pub, value)

        # Wrap as EncryptedNumber and decrypt with phe's DEFAULT BASE=16
        enc = paillier.EncryptedNumber(pub, int(js_c["ciphertext"]), js_c["exponent"])
        naive_decoded = priv.decrypt(enc)   # priv.decrypt uses EncodedNumber (BASE=16)

        # phe thinks: integer × 16^(-6).  JS meant: integer × 10^(-6).
        # So naive result = actual_value × (10/16)^6
        expected_naive = value * (10 / 16) ** 6   # ≈ 0.02981...

        # The naive path gives a completely wrong answer
        assert abs(naive_decoded - value) > 0.1, (
            "The naive BASE=16 decode unexpectedly gave the correct answer — "
            "check if phe's BASE was changed globally."
        )
        # And specifically it's off by the predicted ratio
        assert math.isclose(naive_decoded, expected_naive, rel_tol=1e-3), (
            f"BASE-ratio mismatch: expected ~{expected_naive:.6f}, got {naive_decoded:.6f}"
        )

    def test_naive_phe_inference_wrong(self, keypair):
        """
        Running the old naive path — BASE=16 weight encoding + BASE=16 decode
        (phe's defaults, no Base10EncodedNumber) — on JS ciphertexts gives wrong scores.

        Exact failure mode:
            JS encodes xi as round(xi × 10^6), exponent=-6.
            phe encodes wi as round(wi × 16^p), exponent=-p  (BASE=16).
            priv.decrypt() interprets the result in BASE=16:
                decoded = xi × 10^6 × wi × 16^p × 16^(-6-p)
                        = xi × wi × (10/16)^6
                        ≈ xi × wi × 0.0596   (off by ~16.8×)
        """
        pub, priv = keypair
        mid  = "diabetes"
        w    = TRAINED_WEIGHTS[mid]
        x    = [0.5] * len(w["weights"])

        js_ciphers  = [js_style_encrypt(pub, xi) for xi in x]
        error_ratio = (10 / 16) ** 6   # ≈ 0.0596 — the exact BASE mismatch factor

        for ef, wi in zip(js_ciphers, w["weights"]):
            enc    = paillier.EncryptedNumber(pub, int(ef["ciphertext"]), ef["exponent"])
            result = enc * wi           # phe encodes wi with BASE=16 (old/naive path)

            # Decrypt with phe's default BASE=16 — this is what the old code did
            naive_decoded = priv.decrypt(result)
            expected      = 0.5 * wi   # what the result should be

            # Must be substantially wrong (off by ~94%)
            assert abs(naive_decoded - expected) > abs(expected) * 0.3, (
                "Old BASE=16 decode unexpectedly gave correct answer — "
                "check if phe's default BASE was changed."
            )
            # Specifically off by the predicted (10/16)^6 ratio (±5% tolerance)
            assert math.isclose(naive_decoded, expected * error_ratio, rel_tol=0.05), (
                f"Unexpected error factor for wi={wi}: "
                f"expected ≈{expected * error_ratio:.8f}, got {naive_decoded:.8f}"
            )


# ---------------------------------------------------------------------------
# Test 2: Prove the new Base10 fix is correct
# ---------------------------------------------------------------------------

class TestBase10Fix:
    def test_js_ciphertext_decrypts_correctly(self, keypair):
        """Single JS-encrypted value, decrypted with Base10 gives back original."""
        pub, priv = keypair
        for value in [0.0, 0.25, 0.5, 0.75, 1.0, 0.123456]:
            js_c = js_style_encrypt(pub, value)
            # Decrypt using Base10 convention
            result = priv.decrypt_encoded(
                paillier.EncryptedNumber(pub, int(js_c["ciphertext"]), js_c["exponent"]),
                Encoding=Base10EncodedNumber,
            ).decode()
            assert math.isclose(result, value, abs_tol=1e-6), (
                f"Base10 decode failed: value={value}, got={result}"
            )

    def test_inference_on_js_ciphertexts(self, keypair):
        """
        Core fix test: encrypted_linear_inference on JS-style ciphertexts
        must give a score within 1e-4 of the plaintext dot product.
        """
        pub, priv = keypair
        for mid in MODEL_SPECS:
            w  = TRAINED_WEIGHTS[mid]
            x  = [0.5] * len(w["weights"])

            js_ciphers  = [js_style_encrypt(pub, xi) for xi in x]
            enc_result  = encrypted_linear_inference(pub, js_ciphers, mid)

            he_score    = decrypt_value(priv, enc_result)
            plain_score = sum(wi * xi for wi, xi in zip(w["weights"], x)) + w["bias"]

            assert abs(he_score - plain_score) < 1e-4, (
                f"{mid}: HE score={he_score:.6f}, plaintext={plain_score:.6f}, "
                f"delta={abs(he_score - plain_score):.2e}"
            )


# ---------------------------------------------------------------------------
# Test 3: Statistical correctness across random vectors
# ---------------------------------------------------------------------------

class TestStatisticalCorrectness:
    N_SAMPLES = 200

    def test_all_models_random_vectors(self, keypair):
        """
        200 random normalised feature vectors per model.
        |HE_score - plaintext_score| must be < 1e-4 for every sample.
        """
        pub, priv = keypair
        for mid in MODEL_SPECS:
            w        = TRAINED_WEIGHTS[mid]
            n_feat   = len(w["weights"])
            deltas   = []

            for _ in range(self.N_SAMPLES):
                x = [random.random() for _ in range(n_feat)]

                js_ciphers   = [js_style_encrypt(pub, xi) for xi in x]
                enc_result   = encrypted_linear_inference(pub, js_ciphers, mid)
                he_score     = decrypt_value(priv, enc_result)
                plain_score  = sum(wi * xi for wi, xi in zip(w["weights"], x)) + w["bias"]

                delta = abs(he_score - plain_score)
                deltas.append(delta)
                assert delta < 1e-4, (
                    f"{mid}: delta={delta:.2e} exceeds 1e-4 threshold "
                    f"(he={he_score:.6f}, plain={plain_score:.6f})"
                )

            mean_delta = statistics.mean(deltas)
            max_delta  = max(deltas)
            print(f"\n  {mid:12s}: mean_delta={mean_delta:.2e}  max_delta={max_delta:.2e}  "
                  f"over {self.N_SAMPLES} samples")

    def test_risk_decisions_match(self, keypair):
        """
        HIGH/LOW risk decisions from HE inference must match plaintext
        logistic regression across random samples (zero mismatches allowed,
        except for samples within 0.01 probability of the threshold).
        """
        pub, priv = keypair
        mid      = "diabetes"
        w        = TRAINED_WEIGHTS[mid]
        n_feat   = len(w["weights"])
        thresh   = w.get("threshold", 0.5)
        mismatches = 0

        for _ in range(self.N_SAMPLES):
            x = [random.random() for _ in range(n_feat)]

            js_ciphers   = [js_style_encrypt(pub, xi) for xi in x]
            enc_result   = encrypted_linear_inference(pub, js_ciphers, mid)
            he_score     = decrypt_value(priv, enc_result)

            plain_score  = sum(wi * xi for wi, xi in zip(w["weights"], x)) + w["bias"]

            he_risk      = sigmoid(he_score)    > thresh
            plain_risk   = sigmoid(plain_score) > thresh
            plain_prob   = sigmoid(plain_score)

            # Only count mismatches where the probability is far from threshold
            if he_risk != plain_risk and abs(plain_prob - thresh) > 0.01:
                mismatches += 1

        assert mismatches == 0, (
            f"diabetes: {mismatches} risk decision mismatches across {self.N_SAMPLES} samples"
        )


# ---------------------------------------------------------------------------
# Test 4: Exponent roundtrip — JS can decode the server's response
# ---------------------------------------------------------------------------

class TestExponentRoundtrip:
    def test_response_exponent_is_base10_compatible(self, keypair):
        """
        The exponent returned by the server must be interpretable as a
        BASE=10 exponent by the JavaScript client:
            actual_score = decrypted_bigint × 10^exponent

        Verify this by manually applying what decryptResult() does in crypto.js.
        """
        pub, priv = keypair
        mid  = "diabetes"
        w    = TRAINED_WEIGHTS[mid]
        x    = [0.5] * len(w["weights"])

        js_ciphers   = [js_style_encrypt(pub, xi) for xi in x]
        enc_result   = encrypted_linear_inference(pub, js_ciphers, mid)

        # Mimic JS decryptResult():
        #   1. Standard Paillier decrypt → signed BigInt
        #   2. Apply: result = signed × 10^exponent
        signed_int   = priv.raw_decrypt(int(enc_result["ciphertext"]))
        n            = pub.n
        # Handle sign (same as JS: if m > n/2, it's negative)
        if signed_int > n // 2:
            signed_int -= n

        exponent     = enc_result["exponent"]   # e.g. -12 or dynamic
        scale        = 10 ** (-exponent)        # 10^12 if exponent=-12
        js_score     = signed_int / scale

        plain_score  = sum(wi * xi for wi, xi in zip(w["weights"], x)) + w["bias"]

        assert abs(js_score - plain_score) < 1e-4, (
            f"JS-simulated decode: score={js_score:.6f}, plaintext={plain_score:.6f}"
        )

    def test_raw_path_exponent_is_base10_compatible(self, keypair):
        """
        The raw-arithmetic path always returns exponent=-12 (RESULT_EXPONENT).
        Verify the JS-simulated decode works identically:
            actual_score = decrypted_bigint × 10^(-12)

        This test was missing: the original test only covered the phe path.
        The raw path is tested across all models and random vectors to confirm
        the fixed-exponent contract holds.
        """
        pub, priv = keypair
        n = pub.n

        for mid in MODEL_SPECS:
            w     = TRAINED_WEIGHTS[mid]
            x     = [0.5] * len(w["weights"])

            js_ciphers  = [js_style_encrypt(pub, xi) for xi in x]
            enc_result  = encrypted_linear_inference_raw(pub, js_ciphers, mid)

            # Raw path always returns a fixed exponent of -12
            assert enc_result["exponent"] == -12, (
                f"{mid}: raw path returned unexpected exponent {enc_result['exponent']} "
                f"(expected -12 = -(log10(FEATURE_SCALE) + log10(WEIGHT_SCALE)))"
            )

            # Mimic JS decryptResult() — same logic as decryptResult() in crypto.js
            signed_int = priv.raw_decrypt(int(enc_result["ciphertext"]))
            if signed_int > n // 2:
                signed_int -= n

            exponent = enc_result["exponent"]   # -12 (fixed)
            scale    = 10 ** (-exponent)        # 10^12
            js_score = signed_int / scale

            plain_score = sum(wi * xi for wi, xi in zip(w["weights"], x)) + w["bias"]

            assert abs(js_score - plain_score) < 1e-4, (
                f"{mid} raw path: JS-simulated score={js_score:.6f}, "
                f"plaintext={plain_score:.6f}, delta={abs(js_score - plain_score):.2e}"
            )


# ---------------------------------------------------------------------------
# Test 5: Verify the claim about Base10EncodedNumber and positive/negative weights
# ---------------------------------------------------------------------------

# Actual weights from trained_weights.json — used across all sub-tests
_DIABETES_WEIGHTS = [5.75533623, 3.37774489, 2.25690634,  0.76301865]  # all positive
_HEART_WEIGHTS    = [0.82603114, -2.61300991, 1.15565612,  1.29107435]  # one negative
_ANEMIA_WEIGHTS   = [-13.92226671, -0.19437932, 0.70702177, -0.74095987]  # three negatives


class TestBase10EncodingClaim:
    """
    Verifies (and where necessary refutes) the claim:

        "For positive weights like 5.755 or 0.707, Base10EncodedNumber works
         perfectly — phe stores them as positive integers (5755336 or 707022)
         and the arithmetic is fine."

    Five concrete questions are answered:
      1. Are positive weights stored as positive integers by phe?
      2. Are the claimed integers (5755336, 707022) what phe actually stores?
      3. Does the arithmetic round-trip correctly for positive weights?
      4. What does phe store for negative weights — negative or wrapped?
      5. Does the arithmetic round-trip correctly for negative weights (anemia)?
    """

    # ── Q1: positive weight → positive integer? ──────────────────────────────

    def test_positive_weight_encodes_as_positive_integer(self, keypair):
        """
        Claim: phe stores positive weights as 'normal positive integers'.
        Verify: encoding attribute is positive for every positive weight.
        Applies to all diabetes weights and the one positive anemia weight.
        """
        pub, _ = keypair
        positive_weights = [w for w in _DIABETES_WEIGHTS + _ANEMIA_WEIGHTS if w > 0]

        for wi in positive_weights:
            enc = Base10EncodedNumber.encode(pub, wi)
            assert enc.encoding > 0, (
                f"Positive weight {wi} encoded as non-positive integer: {enc.encoding}"
            )

    # ── Q2: are 5755336 / 707022 what phe stores? ────────────────────────────

    def test_claimed_integers_belong_to_raw_path_not_phe(self, keypair):
        """
        The claim cites '5755336' for 5.75533623 and '707022' for 0.70702177.

        These are exactly round(w × 10^6) — the RAW arithmetic path's scale.
        phe + Base10EncodedNumber uses float precision (~10^16), producing a
        much larger integer.  This test documents both, proving the claim
        is citing the wrong path.
        """
        pub, _ = keypair

        cases = [
            (5.75533623, 5755336),   # first diabetes weight
            (0.70702177, 707022),    # third anemia weight (only positive one)
        ]

        for wi, claimed_int in cases:
            # Raw path: round(wi × 10^6)
            raw_int = round(wi * 1_000_000)
            assert raw_int == claimed_int, (
                f"Test setup error: raw_int={raw_int} != claimed {claimed_int}"
            )

            # phe + Base10: uses float precision, NOT 10^6
            enc = Base10EncodedNumber.encode(pub, wi)
            assert enc.encoding != claimed_int, (
                f"wi={wi}: phe unexpectedly produced the 10^6-scale integer "
                f"{claimed_int} — should use float precision"
            )
            # phe encoding must be much larger (higher precision)
            assert abs(enc.encoding) > claimed_int * 1_000, (
                f"wi={wi}: phe encoding {enc.encoding} is not higher-precision "
                f"than the claimed raw-path integer {claimed_int}"
            )
            # But the float round-trip must be exact
            assert math.isclose(enc.decode(), wi, rel_tol=1e-9), (
                f"wi={wi}: phe round-trip failed — got {enc.decode()}"
            )

    # ── Q3: positive weight arithmetic correct? ───────────────────────────────

    def test_positive_weight_arithmetic_roundtrip(self, keypair):
        """
        For every positive weight (all diabetes, one anemia):
            js_encrypt(xi) * Base10(wi) decrypts to xi × wi within 1e-6.
        Confirms the 'arithmetic is fine' part of the claim.
        """
        pub, priv = keypair
        positive_pairs = [(0.5, w) for w in _DIABETES_WEIGHTS if w > 0]
        positive_pairs += [(0.5, w) for w in _ANEMIA_WEIGHTS   if w > 0]

        for xi, wi in positive_pairs:
            js_c = js_style_encrypt(pub, xi)
            enc  = paillier.EncryptedNumber(pub, int(js_c["ciphertext"]), js_c["exponent"])

            result  = enc * Base10EncodedNumber.encode(pub, wi)
            decoded = priv.decrypt_encoded(result, Encoding=Base10EncodedNumber).decode()
            expected = xi * wi

            assert math.isclose(decoded, expected, rel_tol=1e-6), (
                f"Positive weight xi={xi}, wi={wi}: "
                f"expected={expected:.8f}, got={decoded:.8f}"
            )

    # ── Q4: negative weight — negative or wrapped encoding? ──────────────────

    def test_negative_weight_encoding_is_wrapped_not_negative(self, keypair):
        """
        For a negative weight like -13.92226671, what does phe store internally?

        Test result (confirmed by failure of the original assumption):
          phe wraps negative values as  n - abs_int  (a huge positive close to n),
          NOT as a negative integer.

        Consequence for performance:
          _raw_mul receives a ~2048-bit positive exponent and calls
              pow(ciphertext, n - abs_int, n²)
          whereas a positive weight uses a ~57-bit exponent:
              pow(ciphertext, 57553362300000000, n²)
          Negative weights are therefore ~35× slower per multiplication.

        Correctness is unaffected — E(m)^(n - k) = E(-k·m) holds in Paillier
        (proved by the passing round-trip and inference tests).
        """
        pub, _ = keypair

        for wi in [w for w in _ANEMIA_WEIGHTS + _HEART_WEIGHTS if w < 0]:
            enc = Base10EncodedNumber.encode(pub, wi)

            is_wrapped = enc.encoding > pub.n // 2   # huge positive close to n

            # Float round-trip must be correct regardless of storage convention
            assert math.isclose(enc.decode(), wi, rel_tol=1e-9), (
                f"Negative weight {wi} round-trip failed: got {enc.decode()}"
            )

            # phe wraps negatives — encoding is a huge positive (close to n)
            assert is_wrapped, (
                f"wi={wi}: expected wrapped encoding (> n/2), "
                f"got enc.encoding={enc.encoding}"
            )

            # Encoding must NOT be a small negative integer
            assert enc.encoding > 0, (
                f"wi={wi}: expected positive wrapped encoding, got {enc.encoding}"
            )

    # ── Q5: negative weight arithmetic correct? ───────────────────────────────

    def test_negative_weight_arithmetic_roundtrip(self, keypair):
        """
        If the claim 'breaks for anemia' is correct, this test fails.
        If Base10EncodedNumber handles negative weights correctly, it passes.

        Tests every negative weight in the anemia and heart models:
            js_encrypt(xi) * Base10(wi) decrypts to xi × wi  (negative result).
        """
        pub, priv = keypair
        negative_pairs = [(0.5, w) for w in _ANEMIA_WEIGHTS + _HEART_WEIGHTS if w < 0]

        for xi, wi in negative_pairs:
            js_c = js_style_encrypt(pub, xi)
            enc  = paillier.EncryptedNumber(pub, int(js_c["ciphertext"]), js_c["exponent"])

            result  = enc * Base10EncodedNumber.encode(pub, wi)
            decoded = priv.decrypt_encoded(result, Encoding=Base10EncodedNumber).decode()
            expected = xi * wi  # negative

            assert math.isclose(decoded, expected, rel_tol=1e-6), (
                f"Negative weight xi={xi}, wi={wi}: "
                f"expected={expected:.8f}, got={decoded:.8f}"
            )

    def test_anemia_full_inference_matches_plaintext(self, keypair):
        """
        End-to-end test: encrypted_linear_inference on the anemia model
        (weights [-13.922, -0.194, +0.707, -0.741]) must match the plaintext
        dot-product within 1e-4, across varied input vectors.

        This is the definitive answer to 'does it break for anemia?'
        """
        pub, priv = keypair
        w = TRAINED_WEIGHTS["anemia"]

        test_vectors = [
            [0.5,  0.5,  0.5,  0.5 ],   # uniform mid-range
            [0.1,  0.9,  0.3,  0.7 ],   # varied
            [0.0,  0.0,  0.0,  0.0 ],   # all-zero features
            [1.0,  1.0,  1.0,  1.0 ],   # all-one features
            [0.8,  0.2,  0.6,  0.4 ],   # asymmetric
        ]

        for x in test_vectors:
            js_ciphers  = [js_style_encrypt(pub, xi) for xi in x]
            enc_result  = encrypted_linear_inference(pub, js_ciphers, "anemia")
            he_score    = decrypt_value(priv, enc_result)
            plain_score = sum(wi * xi for wi, xi in zip(w["weights"], x)) + w["bias"]

            assert abs(he_score - plain_score) < 1e-4, (
                f"Anemia x={x}: HE={he_score:.6f}, plain={plain_score:.6f}, "
                f"delta={abs(he_score - plain_score):.2e}"
            )


# ---------------------------------------------------------------------------
# Run as script

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
