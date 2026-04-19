"""
Heimdall - Test Suite
Tests encryption correctness, model inference, and API endpoints.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import phe as paillier
from core.encryption import (
    generate_keys,
    serialize_public_key,
    deserialize_public_key,
    encrypt_vector,
    decrypt_value,
    reconstruct_encrypted_number,
)
from core.models import (
    normalize_features,
    encrypted_linear_inference,
    sigmoid,
    MODEL_SPECS,
    TRAINED_WEIGHTS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def keypair():
    pub, priv = generate_keys(key_size=1024)  # 1024 for test speed
    return pub, priv


# ---------------------------------------------------------------------------
# Encryption tests
# ---------------------------------------------------------------------------

class TestEncryption:
    def test_key_generation(self, keypair):
        pub, priv = keypair
        assert pub is not None
        assert priv is not None
        assert pub.n > 0

    def test_serialize_deserialize_public_key(self, keypair):
        pub, _ = keypair
        serialized = serialize_public_key(pub)
        assert "n" in serialized
        restored = deserialize_public_key(serialized)
        assert restored.n == pub.n

    def test_encrypt_decrypt_roundtrip(self, keypair):
        pub, priv = keypair
        values = [0.5, 1.23, -0.7, 100.0]
        enc_list = encrypt_vector(pub, values)
        for orig, enc_dict in zip(values, enc_list):
            decrypted = decrypt_value(priv, enc_dict)
            assert abs(decrypted - orig) < 1e-6, f"Mismatch: {orig} != {decrypted}"

    def test_additive_homomorphism(self, keypair):
        """E(a) * E(b) = E(a + b)"""
        pub, priv = keypair
        a, b = 3.5, 2.1
        enc_a = pub.encrypt(a)
        enc_b = pub.encrypt(b)
        enc_sum = enc_a + enc_b
        result = priv.decrypt(enc_sum)
        assert abs(result - (a + b)) < 1e-6

    def test_scalar_multiplication(self, keypair):
        """E(a)^k = E(k * a)"""
        pub, priv = keypair
        a, k = 2.0, 5.0
        enc_a = pub.encrypt(a)
        enc_ka = enc_a * k
        result = priv.decrypt(enc_ka)
        assert abs(result - (a * k)) < 1e-6

    def test_semantic_security(self, keypair):
        """Same plaintext encrypts to different ciphertexts."""
        pub, _ = keypair
        enc1 = pub.encrypt(42.0)
        enc2 = pub.encrypt(42.0)
        assert enc1.ciphertext() != enc2.ciphertext()


# ---------------------------------------------------------------------------
# Normalization tests
# ---------------------------------------------------------------------------

class TestNormalization:
    def test_normalization_bounds(self):
        for model_id, spec in MODEL_SPECS.items():
            mins = [f["min"] for f in spec["features"]]
            maxs = [f["max"] for f in spec["features"]]
            norm_min = normalize_features(model_id, mins)
            norm_max = normalize_features(model_id, maxs)
            assert all(abs(v) < 1e-9 for v in norm_min), f"{model_id}: min not 0"
            assert all(abs(v - 1.0) < 1e-9 for v in norm_max), f"{model_id}: max not 1"

    def test_normalization_midpoint(self):
        spec = MODEL_SPECS["diabetes"]["features"]
        mids = [(f["min"] + f["max"]) / 2 for f in spec]
        norm = normalize_features("diabetes", mids)
        assert all(abs(v - 0.5) < 1e-9 for v in norm)


# ---------------------------------------------------------------------------
# Encrypted inference tests
# ---------------------------------------------------------------------------

class TestEncryptedInference:
    def test_encrypted_matches_plaintext(self, keypair):
        """
        Core correctness test: encrypted inference must produce the
        same result as plaintext linear evaluation.
        """
        pub, priv = keypair

        for model_id in MODEL_SPECS:
            spec = MODEL_SPECS[model_id]
            n_features = len(spec["features"])
            # Use midpoint values (all 0.5 after normalization)
            norm_vals = [0.5] * n_features

            # Plaintext result
            w = TRAINED_WEIGHTS[model_id]
            plain_score = sum(w["weights"][i] * norm_vals[i] for i in range(n_features)) + w["bias"]

            # Encrypted result
            enc_list = encrypt_vector(pub, norm_vals)
            enc_result = encrypted_linear_inference(pub, enc_list, model_id)
            decrypted_score = decrypt_value(priv, enc_result)

            assert abs(decrypted_score - plain_score) < 1e-4, (
                f"{model_id}: plaintext={plain_score:.6f}, encrypted={decrypted_score:.6f}"
            )

    def test_all_models_run(self, keypair):
        pub, priv = keypair
        for model_id in MODEL_SPECS:
            n_features = len(MODEL_SPECS[model_id]["features"])
            norm_vals = [0.3] * n_features
            enc_list = encrypt_vector(pub, norm_vals)
            enc_result = encrypted_linear_inference(pub, enc_list, model_id)
            score = decrypt_value(priv, enc_result)
            assert isinstance(score, float)


# ---------------------------------------------------------------------------
# Sigmoid / utility tests
# ---------------------------------------------------------------------------

class TestUtils:
    def test_sigmoid_bounds(self):
        assert 0 < sigmoid(0) < 1
        assert sigmoid(100) > 0.99
        assert sigmoid(-100) < 0.01

    def test_sigmoid_symmetry(self):
        assert abs(sigmoid(0) - 0.5) < 1e-9
        assert abs(sigmoid(x := 2.5) + sigmoid(-x) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
