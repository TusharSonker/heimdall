# Heimdall
### Privacy-Preserving Medical Diagnosis via Paillier Homomorphic Encryption

> **Delhi Technological University — Minor Project**
> Tushar Sonker · Shivansh Rajdehl · Shivam Singh
> Supervised by Prof. Jamkhongam Touthang · 2025–2026

---

## What It Does

Heimdall lets a patient get an AI-based medical diagnosis **without the server ever seeing their raw data**. The patient's data is encrypted on their device using the Paillier cryptosystem, sent as ciphertext to the server, and the server computes the prediction **entirely on encrypted values** — returning an encrypted result that only the patient can decrypt.

```
Patient Device                          Server (Untrusted)
─────────────────                       ──────────────────
Enter vitals                            Receives only ciphertexts
  ↓ Paillier encrypt                    Computes E(y) = Σ wᵢ·E(xᵢ) + b
Send ciphertexts  ──── HTTPS ────────▶  Returns encrypted result
                  ◀───────────────────
Decrypt with
private key
Read result
```

---

## Project Structure

```
heimdall/
├── backend/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── encryption.py       # Paillier key gen, encrypt, decrypt
│   │   └── models.py           # Encrypted linear inference + model specs
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py             # FastAPI app, routes, Pydantic schemas
│   ├── tests/
│   │   └── test_interop.py     # pytest suite (cross-language interop tests)
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── components/
│   │   │   ├── KeyPanel.js           # Key generation display
│   │   │   ├── Pipeline.js           # Step-by-step pipeline visualization
│   │   │   ├── ResultPanel.js        # Risk result display
│   │   │   ├── MetricsBar.js         # Timing benchmarks
│   │   │   └── AuditLog.js           # Audit log (no PHI)
│   │   ├── hooks/
│   │   │   └── usePaillier.js        # React hook for key management
│   │   ├── pages/
│   │   │   └── DiagnosisPage.js      # Main page, pipeline orchestration
│   │   ├── utils/
│   │   │   ├── crypto.js             # Client-side Paillier encryption
│   │   │   └── api.js                # Axios API service
│   │   ├── App.js
│   │   ├── index.js
│   │   └── index.css
│   ├── package.json
│   ├── Dockerfile
│   └── nginx.conf
│
├── docker-compose.yml
└── README.md
```

---

## Medical Models

| Model         | Features                                                                  | Accuracy |
|---------------|---------------------------------------------------------------------------|----------|
| Diabetes      | Pregnancies, Glucose, BP, Skin Thickness, Insulin, BMI, DPF, Age (8)     | 69.5%    |
| Heart Disease | Age, Max HR, Resting SBP, Chest Pain Type (4)                            | 85.0%    |
| Anemia        | Hemoglobin, MCH, MCHC, MCV (4)                                           | 91.2%    |

All models use **linear classifiers** — the only type compatible with additive HE without approximation tricks.

---

## How to Run

### Option A — Manual (Recommended for Development)

#### 1. Backend

```bash
cd heimdall/backend

# Create virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the API server
uvicorn api.main:app --reload --port 8000
```

The API will be live at `http://localhost:8000`.
Swagger docs at `http://localhost:8000/docs`.

#### 2. Frontend

```bash
cd heimdall/frontend

# Install dependencies
npm install

# Start dev server
npm start
```

The app will open at `http://localhost:3000`.

---

### Option B — Docker Compose (Recommended for Deployment)

```bash
cd heimdall

# Build and start both services
docker-compose up --build

# Frontend → http://localhost:3000
# Backend  → http://localhost:8000
```

---

### Option C — Run Tests

```bash
cd heimdall/backend
source venv/bin/activate

pytest tests/test_interop.py -v
```

Key tests:
- `test_naive_phe_path_is_wrong` — proves naive phe+JS mixing gives catastrophically wrong results
- `test_base10_inference_matches_plaintext` — verifies Base10EncodedNumber fix (diabetes, heart)
- `test_raw_inference_matches_plaintext` — verifies raw Paillier fix on all three models
- `test_base10_and_raw_agree` — cross-validates both solutions produce identical scores
- `test_naive_exponent_interpretation_is_incompatible` — regression guard for phe BASE changes

---

## API Reference

### `GET /api/models`
Returns available models and their feature metadata.

### `POST /api/predict`
End-to-end encrypted inference (raw Paillier arithmetic — works for all models).
```json
{
  "model_id": "diabetes",
  "public_key": { "n": "23456..." },
  "encrypted_features": [
    { "ciphertext": "98765...", "exponent": -6 },
    ...
  ]
}
```
Returns:
```json
{
  "model_id": "diabetes",
  "encrypted_result": { "ciphertext": "11111...", "exponent": -12 },
  "threshold": 0.5,
  "inference_time_ms": 3.2,
  "server_note": "Server performed HE inference without accessing plaintext data."
}
```

### `POST /api/predict-raw`
Same inference using the Base10EncodedNumber path (comparison endpoint, fails for anemia).

### `GET /api/benchmark?model_id=diabetes`
Returns timing stats for a full keygen → encrypt → infer → decrypt cycle with synthetic data.

---

## Cross-Language Interoperability

A key contribution of this project is solving the **JS/Python Paillier encoding mismatch**:

- Python's `phe` library encodes floats as `integer × 16^exponent` (BASE=16)
- JavaScript Paillier libraries use `integer × 10^exponent` (BASE=10)

Mixing these naively produces results that are ~16.78× wrong — silently. Two independent solutions are implemented and verified:

1. **Base10EncodedNumber** — subclasses `phe.EncodedNumber` with `BASE=10`, making Python arithmetic match the JS wire convention. Limitation: fails for large negative weights (e.g., anemia model w₀ = −13.92).

2. **Raw Paillier arithmetic** — bypasses `phe` encoding entirely using `pow(c, k, n²)` (modular exponentiation). Works for all models and all weight magnitudes. This is the production default (`/api/predict`).

---

## Security Model

| Property          | Guarantee                                          |
|-------------------|----------------------------------------------------|
| Data confidentiality | Patient features never sent as plaintext      |
| Key security      | Private key generated and stored client-side only  |
| Semantic security | Different ciphertext per encryption (random nonce) |
| Transport         | HTTPS (TLS) in production                          |
| Server trust      | Server is fully untrusted — sees only ciphertexts  |
| Key size          | 2048-bit Paillier (IND-CPA secure under DCR)       |

---

## The Math (Quick Reference)

**Key generation:** Choose primes p, q. Compute n = pq, λ = lcm(p−1, q−1).

**Encryption:** E(m) = gᵐ · rⁿ mod n²  (r is random nonce, g = n+1)

**Additive homomorphism:**
```
E(m₁) · E(m₂) mod n² = E(m₁ + m₂)
E(m)ᵏ mod n²          = E(k · m)
```

**Encrypted inference:**
```
E(y) = E(w₁x₁ + w₂x₂ + ... + wₙxₙ + b)
     = E(x₁)^w₁ · E(x₂)^w₂ · ... · gᵇ   (mod n²)
```

---

## Requirements

- **Backend:** Python 3.9+, pip
- **Frontend:** Node.js 16+, npm
- **Docker:** Docker + Docker Compose (for Option B)

---

## Dependencies

| Layer      | Package             | Purpose                        |
|------------|---------------------|--------------------------------|
| Backend    | `phe`               | Paillier HE implementation     |
| Backend    | `fastapi`           | REST API framework             |
| Backend    | `uvicorn`           | ASGI server                    |
| Backend    | `scikit-learn`      | Model training utilities       |
| Backend    | `pydantic`          | Request/response validation    |
| Frontend   | `react`             | UI framework                   |
| Frontend   | `axios`             | HTTP client                    |
