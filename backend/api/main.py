"""
Heimdall - FastAPI Backend
Privacy-Preserving Medical Diagnosis System
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from typing import List, Optional
import time
import math
import logging
import phe as paillier

from core.encryption import deserialize_public_key, decrypt_value
from core.models import (
    normalize_features,
    encrypted_linear_inference,
    encrypted_linear_inference_raw,
    get_model_specs,
    sigmoid,
    MODEL_SPECS,
    TRAINED_WEIGHTS,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("heimdall")

app = FastAPI(
    title="Heimdall API",
    description="Privacy-Preserving Medical Diagnosis via Paillier HE",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class PublicKeyPayload(BaseModel):
    n: str  # Public key modulus as decimal string


class EncryptedFeature(BaseModel):
    ciphertext: str
    exponent: int


class PredictRequest(BaseModel):
    model_id: str
    public_key: PublicKeyPayload
    encrypted_features: List[EncryptedFeature]

    @field_validator("model_id")
    @classmethod
    def validate_model(cls, v):
        if v not in MODEL_SPECS:
            raise ValueError(f"Unknown model_id '{v}'. Valid: {list(MODEL_SPECS.keys())}")
        return v


class PredictResponse(BaseModel):
    model_id: str
    encrypted_result: dict       # {ciphertext, exponent} — never decrypted server-side
    threshold: float             # model decision threshold (not derived from patient data)
    inference_time_ms: float
    server_note: str = "Server performed HE inference without accessing plaintext data."


class ModelsResponse(BaseModel):
    models: dict


class BenchmarkResponse(BaseModel):
    model_id: str
    key_size_bits: int
    n_features: int
    keygen_time_ms: float
    encrypt_time_ms: float
    inference_time_ms: float
    decrypt_time_ms: float
    total_time_ms: float
    note: str


class PlaintextPredictRequest(BaseModel):
    model_id: str
    features: List[float]   # normalised [0, 1] — server never receives raw values

    @field_validator("model_id")
    @classmethod
    def validate_model(cls, v):
        if v not in MODEL_SPECS:
            raise ValueError(f"Unknown model_id '{v}'")
        return v


class PlaintextPredictResponse(BaseModel):
    model_id: str
    risk: str
    probability: float
    score: float
    inference_time_ms: float
    note: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "service": "Heimdall", "version": "2.0.0"}


@app.get("/api/models", response_model=ModelsResponse, tags=["Models"])
def list_models():
    """Return available medical models and their feature specifications."""
    return {"models": get_model_specs()}


@app.post("/api/predict", response_model=PredictResponse, tags=["Inference"])
def predict(req: PredictRequest):
    """
    True end-to-end encrypted inference.

    The client generates a Paillier keypair locally in the browser, encrypts
    each normalised feature, and sends only ciphertexts.  The server:
      1. Reconstructs the public key (no private key is ever sent).
      2. Computes E(y) = Σ wᵢ·E(xᵢ) + b entirely in ciphertext space.
      3. Returns the ENCRYPTED result — never decrypts it.

    The client decrypts the result with its private key (which never leaves
    the browser) and applies sigmoid + threshold to determine risk.
    """
    start = time.perf_counter()

    try:
        public_key = deserialize_public_key({"n": req.public_key.n})
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid public key: {e}")

    expected = len(MODEL_SPECS[req.model_id]["features"])
    if len(req.encrypted_features) != expected:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {expected} encrypted features, got {len(req.encrypted_features)}",
        )

    try:
        enc_features    = [f.model_dump() for f in req.encrypted_features]
        encrypted_result = encrypted_linear_inference_raw(public_key, enc_features, req.model_id)
    except Exception as e:
        logger.error(f"[{req.model_id}] Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    elapsed_ms = (time.perf_counter() - start) * 1000
    threshold  = TRAINED_WEIGHTS[req.model_id].get("threshold", 0.5)

    logger.info(f"[{req.model_id}] Encrypted inference in {elapsed_ms:.2f}ms")

    return PredictResponse(
        model_id=req.model_id,
        encrypted_result=encrypted_result,
        threshold=threshold,
        inference_time_ms=round(elapsed_ms, 2),
    )


@app.get("/api/benchmark", response_model=BenchmarkResponse, tags=["Benchmark"])
def benchmark(model_id: str = "diabetes", key_bits: int = 2048):
    """
    Full end-to-end benchmark: keygen → encrypt → HE inference → decrypt.
    Uses synthetic data only — no patient values involved.
    """
    if model_id not in MODEL_SPECS:
        raise HTTPException(status_code=400, detail="Unknown model_id")
    if key_bits not in (1024, 2048, 3072):
        raise HTTPException(status_code=400, detail="key_bits must be 1024, 2048 or 3072")

    n_features = len(MODEL_SPECS[model_id]["features"])

    # Key generation
    t_keygen = time.perf_counter()
    pub, priv = paillier.generate_paillier_keypair(n_length=key_bits)
    keygen_ms = (time.perf_counter() - t_keygen) * 1000

    # Encryption (server-side simulation using phe — same Paillier math)
    dummy = [0.5] * n_features
    t_enc = time.perf_counter()
    enc_list = [pub.encrypt(v) for v in dummy]
    enc_ms   = (time.perf_counter() - t_enc) * 1000
    enc_dicts = [{"ciphertext": str(e.ciphertext()), "exponent": e.exponent}
                 for e in enc_list]

    # HE inference (server-side, no decryption)
    t_inf = time.perf_counter()
    enc_result = encrypted_linear_inference(pub, enc_dicts, model_id)
    inf_ms = (time.perf_counter() - t_inf) * 1000

    # Decryption
    t_dec = time.perf_counter()
    enc_num = paillier.EncryptedNumber(pub, int(enc_result["ciphertext"]),
                                       enc_result["exponent"])
    priv.decrypt(enc_num)
    dec_ms = (time.perf_counter() - t_dec) * 1000

    total_ms = keygen_ms + enc_ms + inf_ms + dec_ms

    return BenchmarkResponse(
        model_id=model_id,
        key_size_bits=key_bits,
        n_features=n_features,
        keygen_time_ms=round(keygen_ms, 2),
        encrypt_time_ms=round(enc_ms, 2),
        inference_time_ms=round(inf_ms, 2),
        decrypt_time_ms=round(dec_ms, 2),
        total_time_ms=round(total_ms, 2),
        note="Benchmark uses synthetic data. No PHI involved.",
    )


@app.post("/api/predict-plaintext", response_model=PlaintextPredictResponse, tags=["Inference"])
def predict_plaintext(req: PlaintextPredictRequest):
    """
    Comparison endpoint: server receives normalised plaintext features,
    runs the same HE pipeline internally, and returns decrypted risk.

    Privacy model: features are TLS-protected in transit but the server does
    see normalised values.  Use /api/predict for full E2E privacy.
    """
    start = time.perf_counter()

    expected = len(MODEL_SPECS[req.model_id]["features"])
    if len(req.features) != expected:
        raise HTTPException(status_code=400, detail=f"Expected {expected} features")

    pub, priv = paillier.generate_paillier_keypair(n_length=2048)

    enc_features = []
    for v in req.features:
        enc = pub.encrypt(float(v))
        enc_features.append({"ciphertext": str(enc.ciphertext()), "exponent": enc.exponent})

    enc_result = encrypted_linear_inference(pub, enc_features, req.model_id)

    enc_num = paillier.EncryptedNumber(pub, int(enc_result["ciphertext"]),
                                       enc_result["exponent"])
    score       = priv.decrypt(enc_num)
    probability = sigmoid(score)
    threshold   = TRAINED_WEIGHTS[req.model_id].get("threshold", 0.5)
    risk        = "HIGH" if probability > threshold else "LOW"

    elapsed = (time.perf_counter() - start) * 1000
    logger.info(f"[{req.model_id}] Plaintext predict: score={score:.4f} risk={risk}")

    return PlaintextPredictResponse(
        model_id=req.model_id,
        risk=risk,
        probability=round(probability, 6),
        score=round(score, 6),
        inference_time_ms=round(elapsed, 2),
        note="Server received normalised features (TLS-only privacy). Use /api/predict for E2E HE privacy.",
    )
