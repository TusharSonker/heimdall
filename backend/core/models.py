"""
Heimdall - Machine Learning Inference Module
=============================================
Loads trained weights from core/trained_weights.json if it exists
(produced by scripts/train_models.py), otherwise falls back to the
placeholder weights from the project report.

The encrypted linear inference formula:
    E(y) = w1*E(x1) + w2*E(x2) + ... + wn*E(xn) + b

This works because Paillier supports:
    E(a) * scalar  =  E(scalar * a)
    E(a) + E(b)    =  E(a + b)

encrypted_linear_inference_raw bypasses phe's base-16 encoding so that
JS-generated ciphertexts (which use base-10 scaling of 1e6) produce the
correct score when the client decrypts with base-10 exponent arithmetic.
"""

import json
import math
import secrets
from pathlib import Path
import phe as paillier
from .encryption import reconstruct_encrypted_number, Base10EncodedNumber

# ── Paths ──────────────────────────────────────────────────────────────────
_WEIGHTS_FILE = Path(__file__).parent / "trained_weights.json"


# ── Fallback weights (from the project report) ─────────────────────────────
_FALLBACK_WEIGHTS = {
    "diabetes": {
        "weights":       [0.035, 0.028, 0.015, 0.012],
        "bias":          -4.5,
        "feature_names": ["glucose", "bmi", "age", "bp"],
        "feature_mins":  [0,   10,  1,  40],
        "feature_maxs":  [300, 60, 120, 200],
    },
    "heart": {
        "weights":       [0.02, 0.003, 0.015, 0.4],
        "bias":          -3.2,
        "feature_names": ["age", "chol", "sbp", "smoking"],
        "feature_mins":  [1,   100, 80, 0],
        "feature_maxs":  [120, 600, 250, 1],
    },
    "anemia": {
        "weights":      [0.5, 0.02, 0.01, 0.01],
        "bias":         -8.0,
        "feature_names":["hemoglobin", "mch", "mchc", "mcv"],
        "feature_mins": [4,  15, 20,  50],
        "feature_maxs": [20, 40, 40, 120],
    },
}


# ── Model UI metadata ──────────────────────────────────────────────────────
MODEL_SPECS = {
    "diabetes": {
        "label":    "Diabetes",
        "accuracy": 77.3,
        "features": [
            {"id": "glucose", "label": "Glucose (mg/dL)",      "min": 0,   "max": 300, "hint": "Normal: 70-100"},
            {"id": "bmi",     "label": "BMI",                   "min": 10,  "max": 60,  "hint": "Normal: 18.5-24.9"},
            {"id": "age",     "label": "Age (years)",            "min": 1,   "max": 120, "hint": ""},
            {"id": "bp",      "label": "Blood Pressure (mmHg)", "min": 40,  "max": 200, "hint": "Normal diastolic: 60-80"},
        ],
    },
    "heart": {
        "label":    "Heart Disease",
        "accuracy": 80.0,
        "features": [
            {"id": "age",      "label": "Age (years)",               "min": 1,   "max": 120, "hint": ""},
            {"id": "thalach",  "label": "Max Heart Rate (bpm)",      "min": 60,  "max": 220, "hint": "Normal: 100-170 during exercise"},
            {"id": "trestbps", "label": "Resting Systolic BP (mmHg)","min": 80,  "max": 250, "hint": "Normal: <120"},
            {"id": "cp",       "label": "Chest Pain Type (0-3)",     "min": 0,   "max": 3,   "hint": "0=typical angina, 3=asymptomatic"},
        ],
    },
    "anemia": {
        "label":    "Anemia",
        "accuracy": 89.4,
        "features": [
            {"id": "hgb",  "label": "Hemoglobin (g/dL)", "min": 4,  "max": 20,  "hint": "Normal: 12-17.5"},
            {"id": "mch",  "label": "MCH (pg)",           "min": 15, "max": 40,  "hint": "Normal: 27-33"},
            {"id": "mchc", "label": "MCHC (g/dL)",        "min": 20, "max": 40,  "hint": "Normal: 31.5-35.7"},
            {"id": "mcv",  "label": "MCV (fL)",            "min": 50, "max": 120, "hint": "Normal: 80-100"},
        ],
    },
}


# ── Load weights ───────────────────────────────────────────────────────────
def _load_weights() -> dict:
    if _WEIGHTS_FILE.exists():
        with open(_WEIGHTS_FILE) as f:
            data = json.load(f)
        print(f"[Heimdall] Loaded TRAINED weights from {_WEIGHTS_FILE}")
        for model_id, w in data.items():
            if model_id in MODEL_SPECS:
                if "metrics" in w:
                    MODEL_SPECS[model_id]["accuracy"] = round(w["metrics"]["accuracy"] * 100, 1)
                if "threshold" in w:
                    MODEL_SPECS[model_id]["threshold"] = w["threshold"]
        return data
    else:
        print("[Heimdall] WARNING: trained_weights.json not found. Using placeholder weights.")
        print("           Run: python scripts/train_models.py")
        return _FALLBACK_WEIGHTS


TRAINED_WEIGHTS = _load_weights()


# ── Normalization ──────────────────────────────────────────────────────────
def normalize_features(model_id: str, raw_values: list) -> list:
    w    = TRAINED_WEIGHTS[model_id]
    mins = w["feature_mins"]
    maxs = w["feature_maxs"]
    return [max(0.0, min(1.0, (v - mins[i]) / (maxs[i] - mins[i])))
            for i, v in enumerate(raw_values)]


# ── Encrypted inference ────────────────────────────────────────────────────
def encrypted_linear_inference(
    public_key: paillier.PaillierPublicKey,
    encrypted_features: list,
    model_id: str
) -> dict:
    """
    Compute E(y) = Σ wᵢ·E(xᵢ) + b using homomorphic operations.
    Server NEVER decrypts — only sees ciphertexts.

    Limitation: fails for models whose weights span more than one order of
    magnitude (e.g. anemia: w0=−13.92 vs w1=−0.19). When two EncryptedNumbers
    with different product-exponents are added, phe calls
    EncryptedNumber.decrease_exponent_to, which hardcodes EncodedNumber.BASE=16
    instead of Base10EncodedNumber.BASE=10. Each alignment step introduces a
    (16/10)^k error. Diabetes and heart work because their weights all encode at
    the same exponent, so no EncryptedNumber alignment is ever triggered.
    Use encrypted_linear_inference_raw for guaranteed correctness on all models.
    """
    w       = TRAINED_WEIGHTS[model_id]
    weights = w["weights"]
    bias    = w["bias"]

    if len(encrypted_features) != len(weights):
        raise ValueError(f"Expected {len(weights)} features, got {len(encrypted_features)}")

    enc_nums = [reconstruct_encrypted_number(public_key, ef) for ef in encrypted_features]

    # Use Base10EncodedNumber for all scalars so phe tracks exponents in BASE=10,
    # matching the JS wire convention (exponent = -6 means × 10^-6, not × 16^-6).
    result = enc_nums[0] * Base10EncodedNumber.encode(public_key, weights[0])
    for i in range(1, len(enc_nums)):
        result = result + enc_nums[i] * Base10EncodedNumber.encode(public_key, weights[i])
    result = result + Base10EncodedNumber.encode(public_key, bias)

    return {"ciphertext": str(result.ciphertext()), "exponent": result.exponent}


# ── Raw Paillier inference (JS-compatible) ─────────────────────────────────

# JS scales each normalised feature by FEATURE_SCALE before encrypting.
# The server scales each weight by WEIGHT_SCALE (integer approximation).
# Final ciphertext decrypts to  (Σ w_i*x_i + b) * TOTAL_SCALE  in BigInt,
# and the returned exponent tells the JS client to divide by TOTAL_SCALE.
_FEATURE_SCALE = 1_000_000   # must match crypto.js encryptValue (1e6)
_WEIGHT_SCALE  = 1_000_000   # precision for weight quantisation
_TOTAL_SCALE   = _FEATURE_SCALE * _WEIGHT_SCALE          # 1e12
_RESULT_EXPONENT = -(int(math.log10(_FEATURE_SCALE))
                     + int(math.log10(_WEIGHT_SCALE)))   # -12


def encrypted_linear_inference_raw(
    public_key: paillier.PaillierPublicKey,
    encrypted_features: list,
    model_id: str,
) -> dict:
    """
    Encrypted dot-product for JS-generated Paillier ciphertexts.

    JS encrypts:  m_i = round(x_i * _FEATURE_SCALE)  with exponent -6.
    Server computes (purely in raw Paillier arithmetic — no phe encoding):
        E(Σ round(w_i*_WEIGHT_SCALE)*m_i  +  round(b*_TOTAL_SCALE))
    which decrypts to  (Σ w_i*x_i + b) * _TOTAL_SCALE  ≈ score * 1e12.

    Client divides the BigInt result by 10^12 to recover the score.
    Correctness: bypasses phe's base-16 encoding that would cause a ~16×
    magnitude error if mixed with the JS client's base-10 convention.
    """
    w        = TRAINED_WEIGHTS[model_id]
    weights  = w["weights"]
    bias     = w["bias"]

    n  = public_key.n
    n2 = n * n

    int_weights = [round(wi * _WEIGHT_SCALE) for wi in weights]
    int_bias    = round(bias * _TOTAL_SCALE)

    # Initialise accumulator with E(0) using a fresh random nonce
    r0 = secrets.randbelow(n) or 1
    result_c = pow(r0, n, n2)   # E(0, r0)

    # Accumulate E(Σ int_w_i * m_i) via raw Paillier scalar multiplication:
    #   E(m)^k  mod n²  =  E(k·m)          k > 0
    #   pow(c, k, n²) with k < 0 uses Python's modular inverse → E(-|k|·m)
    for ef, iw in zip(encrypted_features, int_weights):
        if iw == 0:
            continue
        c_i    = int(ef["ciphertext"])
        term_c = pow(c_i, iw, n2)          # handles negative iw via mod-inverse
        result_c = result_c * term_c % n2

    # Add bias:  g = n+1  →  g^m mod n² = (1 + m·n) mod n²  (binomial theorem)
    int_bias_mod = int_bias % n            # maps negatives into Z_n
    enc_bias     = (1 + int_bias_mod * n) % n2
    result_c     = result_c * enc_bias % n2

    # Re-randomise for semantic security
    r = secrets.randbelow(n) or 1
    result_c = result_c * pow(r, n, n2) % n2

    return {"ciphertext": str(result_c), "exponent": _RESULT_EXPONENT}


# ── Helpers ────────────────────────────────────────────────────────────────
def get_model_specs() -> dict:
    return MODEL_SPECS

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))
