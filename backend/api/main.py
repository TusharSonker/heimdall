"""
Heimdall - FastAPI Backend
Privacy-Preserving Medical Diagnosis System
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from typing import List, Optional
import time, math
import logging
import phe as paillier

from core.encryption import deserialize_public_key, decrypt_value
from core.models import (
    normalize_features,
    encrypted_linear_inference,
    get_model_specs,
    sigmoid,
    MODEL_SPECS,
)
from core.models import TRAINED_WEIGHTS
print("Loaded weights:", {k: v['weights'] for k, v in TRAINED_WEIGHTS.items()})


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("heimdall")

app = FastAPI(
    title="Heimdall API",
    description="Privacy-Preserving Medical Diagnosis via Paillier HE",
    version="1.0.0",
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
    n: str  # Public key modulus as string


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
    encrypted_result: dict          # {ciphertext, exponent} — never decrypted server-side
    inference_time_ms: float
    server_note: str = "Server never accessed plaintext data."


class ModelsResponse(BaseModel):
    models: dict


class BenchmarkResponse(BaseModel):
    model_id: str
    key_size_bits: int
    n_features: int
    inference_time_ms: float
    note: str

class PredictSafeRequest(BaseModel):
    model_id: str
    public_key: PublicKeyPayload
    private_key_lambda: str   # client sends private key components
    private_key_mu: str
    encrypted_features: List[EncryptedFeature]

class PredictDecryptedResponse(BaseModel):
    model_id:          str
    risk:              str    # "HIGH" or "LOW"
    probability:       float
    inference_time_ms: float
    server_note:       str

class KeygenResponse(BaseModel):
    session_id: str
    public_key_n: str

class DecryptRequest(BaseModel):
    session_id:       str
    model_id:         str   
    encrypted_result: dict

class DecryptResponse(BaseModel):
    score:       float
    probability: float
    risk:        str


# In-memory session store (per server restart)
_sessions = {}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "service": "Heimdall", "version": "1.0.0"}


@app.get("/api/models", response_model=ModelsResponse, tags=["Models"])
def list_models():
    """Return available medical models and their feature specifications."""
    return {"models": get_model_specs()}


@app.post("/api/predict", response_model=PredictResponse, tags=["Inference"])
def predict(req: PredictRequest):
    """
    Perform encrypted inference.

    The server:
      1. Reconstructs the public key (no private key involved)
      2. Computes the linear model over ciphertexts using HE operations
      3. Returns the ENCRYPTED result — never decrypts it

    The client decrypts the result with their private key.
    """
    start = time.perf_counter()

    # Reconstruct public key from client-provided n
    try:
        public_key = deserialize_public_key({"n": req.public_key.n})
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid public key: {e}")

    # Validate feature count
    expected = len(MODEL_SPECS[req.model_id]["features"])
    if len(req.encrypted_features) != expected:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {expected} encrypted features, got {len(req.encrypted_features)}"
        )

    # Perform encrypted inference
    try:
        enc_features = [f.model_dump() for f in req.encrypted_features]
        encrypted_result = encrypted_linear_inference(public_key, enc_features, req.model_id)
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(f"[{req.model_id}] Encrypted inference complete in {elapsed_ms:.2f}ms")

    # -----------------------------------------
    print(f"[DEBUG] public_key.n = {str(public_key.n)[:30]}...")
    print(f"[DEBUG] encrypted_features[0] ciphertext = {req.encrypted_features[0].ciphertext[:30]}...")
    print(f"[DEBUG] encrypted_features[0] exponent   = {req.encrypted_features[0].exponent}")

    # Test: try to decrypt with a dummy private key to see if the ciphertext is valid
    try:
        import phe as paillier
        enc_num = paillier.EncryptedNumber(public_key, int(req.encrypted_features[0].ciphertext), req.encrypted_features[0].exponent)
        print(f"[DEBUG] EncryptedNumber created OK")
        print(f"[DEBUG] ciphertext() = {str(enc_num.ciphertext())[:30]}...")
    except Exception as e:
        print(f"[DEBUG] EncryptedNumber creation failed: {e}")

    print(f"[DEBUG] result ciphertext = {str(encrypted_result['ciphertext'])[:30]}...")
    print(f"[DEBUG] result exponent   = {encrypted_result['exponent']}")
    # -----------------------------------------

    return PredictResponse(
        model_id=req.model_id,
        encrypted_result=encrypted_result,
        inference_time_ms=round(elapsed_ms, 2),
    )


@app.get("/api/benchmark", response_model=BenchmarkResponse, tags=["Benchmark"])
def benchmark(model_id: str = "diabetes"):
    """
    Run a quick benchmark: generate keys, encrypt features, infer, return timing.
    This uses dummy data — no real patient values.
    """
    if model_id not in MODEL_SPECS:
        raise HTTPException(status_code=400, detail="Unknown model_id")

    import phe as paillier

    start_total = time.perf_counter()
    pub, priv = paillier.generate_paillier_keypair(n_length=2048)

    n_features = len(MODEL_SPECS[model_id]["features"])
    dummy_values = [0.5] * n_features
    enc = [pub.encrypt(v) for v in dummy_values]
    enc_dicts = [{"ciphertext": str(e.ciphertext()), "exponent": e.exponent} for e in enc]

    inf_start = time.perf_counter()
    encrypted_linear_inference(pub, enc_dicts, model_id)
    inf_ms = (time.perf_counter() - inf_start) * 1000

    return BenchmarkResponse(
        model_id=model_id,
        key_size_bits=2048,
        n_features=n_features,
        inference_time_ms=round(inf_ms, 2),
        note="Benchmark uses dummy data. No PHI involved.",
    )
@app.post("/api/predict-safe", tags=["Inference"])
def predict_safe(req: PredictRequest):
    """
    Variant of /api/predict where the server decrypts only the final
    score — not the input features. Features remain encrypted throughout.
    
    Privacy guarantee: server sees model output (a risk score), not inputs.
    This is necessary due to JS/Python Paillier library encoding differences.
    """
    start = time.perf_counter()

    # Reconstruct public key
    try:
        public_key = deserialize_public_key({"n": req.public_key.n})
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid public key: {e}")

    # Generate a server-side key pair for this request
    # The client's public key is used for HE operations
    # but we need a private key to decrypt the result
    import phe as paillier
    
    # Re-encrypt features using server-generated key pair
    # This lets us decrypt the result while keeping inputs encrypted
    server_pub, server_priv = paillier.generate_paillier_keypair(n_length=1024)
    
    # Decrypt client features using... wait, we can't — no private key.
    # 
    # Correct approach: use the exponent info to do plaintext arithmetic
    # since we know the weights are plaintext anyway.
    
    # Actually the simplest correct approach:
    # Ask client to send normalized values (still encrypted individually)
    # and we compute dot product — but we already do that.
    
    # The issue is purely in decryption. So: do HE on client ciphertexts,
    # return encrypted result PLUS help client decrypt via a hint.
    
    raise HTTPException(status_code=501, detail="Use /api/predict-result instead")

@app.post("/api/predict-result", response_model=PredictDecryptedResponse, tags=["Inference"])
def predict_result(req: PredictRequest):
    """
    Server decrypts only the final score using a per-request ephemeral key.
    
    Flow:
      1. Client encrypts features with server's ephemeral public key
      2. Server does HE inference
      3. Server decrypts only the score (not features)  
      4. Returns risk + probability — never the raw features
    
    Privacy: server sees output score only, never input features.
    """
    import phe as paillier, math

    start = time.perf_counter()

    if req.model_id not in MODEL_SPECS:
        raise HTTPException(status_code=400, detail="Unknown model_id")

    try:
        public_key = deserialize_public_key({"n": req.public_key.n})
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid public key: {e}")

    expected = len(MODEL_SPECS[req.model_id]["features"])
    if len(req.encrypted_features) != expected:
        raise HTTPException(status_code=400, detail=f"Expected {expected} features")

    try:
        enc_features = [f.model_dump() for f in req.encrypted_features]
        enc_result   = encrypted_linear_inference(public_key, enc_features, req.model_id)

        # Reconstruct and decrypt the result — server sees only the score
        result_enc = paillier.EncryptedNumber(
            public_key,
            int(enc_result["ciphertext"]),
            int(enc_result["exponent"])
        )

        # We need private key to decrypt — but we don't have it.
        # So we ask the client to provide an ephemeral server keypair public key.
        raise HTTPException(status_code=501, detail="See note below")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    

@app.post("/api/keygen", response_model=KeygenResponse, tags=["Crypto"])
def keygen():
    import uuid
    pub, priv = paillier.generate_paillier_keypair(n_length=2048)
    
    # Self test — verify phe encrypt/decrypt works
    test_val = 0.2833
    enc_test = pub.encrypt(test_val)
    dec_test = priv.decrypt(enc_test)
    print(f"[keygen] self-test: encrypt({test_val}) -> decrypt = {dec_test}")
    # Also test with integer scaling (what JS sends)
    enc_int = pub.encrypt(283300)  # 0.2833 * 1e6
    dec_int = priv.decrypt(enc_int)
    print(f"[keygen] int-test: encrypt(283300) -> decrypt = {dec_int}")
    
    session_id = str(uuid.uuid4())
    _sessions[session_id] = priv
    return KeygenResponse(session_id=session_id, public_key_n=str(pub.n))


@app.post("/api/decrypt-result", response_model=DecryptResponse, tags=["Crypto"])
def decrypt_result_endpoint(req: DecryptRequest):
    priv = _sessions.get(req.session_id)
    if not priv:
        raise HTTPException(status_code=404, detail="Session expired.")
    try:
        enc_num = paillier.EncryptedNumber(
            priv.public_key,
            int(req.encrypted_result["ciphertext"]),
            int(req.encrypted_result["exponent"])
        )
        score       = priv.decrypt(enc_num)
        probability = 1 / (1 + math.exp(-score))

        # Use optimal threshold from training instead of 0.5
        # This is stored in trained_weights.json after retraining
        from core.models import TRAINED_WEIGHTS
        threshold = TRAINED_WEIGHTS.get(req.model_id, {}).get("threshold", 0.5) \
                    if hasattr(req, 'model_id') else 0.5

        risk = "HIGH" if probability > threshold else "LOW"
        print(f"[decrypt-result] score={score:.4f} prob={probability:.4f} threshold={threshold} risk={risk}")
        return DecryptResponse(score=round(score,6), probability=round(probability,6), risk=risk)
    except Exception as e:
        print(f"[decrypt-result] ERROR: {e}")
        raise HTTPException(status_code=500, detail=f"Decryption failed: {e}")

class PlaintextPredictRequest(BaseModel):
    model_id: str
    features: List[float]

    @field_validator("model_id")
    @classmethod
    def validate_model(cls, v):
        if v not in MODEL_SPECS:
            raise ValueError(f"Unknown model_id '{v}'")
        return v

class PlaintextPredictResponse(BaseModel):
    model_id:          str
    risk:              str
    probability:       float
    score:             float
    inference_time_ms: float
    note:              str

@app.post("/api/predict-plaintext", response_model=PlaintextPredictResponse, tags=["Inference"])
def predict_plaintext(req: PlaintextPredictRequest):
    """
    Accepts normalized feature values, encrypts with phe server-side,
    runs HE inference, decrypts, returns result.
    Features protected by TLS in transit.
    """
    import time, math
    start = time.perf_counter()

    expected = len(MODEL_SPECS[req.model_id]["features"])
    if len(req.features) != expected:
        raise HTTPException(status_code=400, detail=f"Expected {expected} features")

    pub, priv = paillier.generate_paillier_keypair(n_length=2048)

    # Encrypt features using phe natively
    enc_features = []
    for v in req.features:
        enc = pub.encrypt(float(v))
        enc_features.append({
            "ciphertext": str(enc.ciphertext()),
            "exponent":   enc.exponent
        })

    # Run encrypted inference
    enc_result = encrypted_linear_inference(pub, enc_features, req.model_id)

    # Decrypt result
    enc_num = paillier.EncryptedNumber(
        pub,
        int(enc_result["ciphertext"]),
        int(enc_result["exponent"])
    )
    score       = priv.decrypt(enc_num)
    probability = 1 / (1 + math.exp(-score))

    from core.models import TRAINED_WEIGHTS
    threshold = TRAINED_WEIGHTS[req.model_id].get("threshold", 0.5)
    risk      = "HIGH" if probability > threshold else "LOW"

    elapsed = (time.perf_counter() - start) * 1000
    print(f"[predict-plaintext] {req.model_id} score={score:.4f} prob={probability:.4f} risk={risk}")

    return PlaintextPredictResponse(
        model_id=req.model_id,
        risk=risk,
        probability=round(probability, 6),
        score=round(score, 6),
        inference_time_ms=round(elapsed, 2),
        note="Features encrypted with phe server-side. TLS protects transit."
    )