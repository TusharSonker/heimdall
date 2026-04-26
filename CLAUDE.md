# Heimdall — CLAUDE.md

Privacy-Preserving Medical Diagnosis System (BTech Project, DTU Sem 8).
Student: Shivansh Raj · pinkman.builds@gmail.com

---

## What This Project Does

Patients enter medical vitals in a browser. Those values are **encrypted client-side
using Paillier homomorphic encryption** (2048-bit), sent as ciphertexts to a FastAPI
server, and the server computes a logistic regression score **entirely on ciphertexts**
without ever seeing the plaintext. The browser decrypts the result with a private key
that never leaves it.

Three diagnosis models: **Diabetes**, **Heart Disease**, **Anemia**.

---

## Repository Layout

```
heimdall/                   ← git root (CLAUDE.md lives here)
└── heimdall/
    ├── backend/
    │   ├── api/main.py             ← FastAPI app, all routes
    │   ├── core/
    │   │   ├── encryption.py       ← Base10EncodedNumber + key helpers
    │   │   ├── models.py           ← encrypted_linear_inference (phe path)
    │   │   │                         encrypted_linear_inference_raw (raw path)
    │   │   └── trained_weights.json← committed — DO NOT delete or gitignore
    │   ├── scripts/
    │   │   └── train_models.py     ← regenerates trained_weights.json
    │   │                             (datasets not committed — see data/README.md)
    │   └── tests/
    │       ├── test_heimdall.py    ← core unit tests
    │       └── test_interop.py     ← JS/Python interop tests (main focus)
    └── frontend/
        └── src/
            ├── utils/crypto.js     ← pure JS Paillier (keygen, encrypt, decrypt)
            ├── utils/api.js        ← Axios calls to backend
            ├── hooks/usePaillier.js← React hook: key lifecycle, encrypt, decrypt
            └── pages/DiagnosisPage.js ← main pipeline orchestrator
```

---

## The Core Technical Problem (and Solution)

### The BASE mismatch

JavaScript encrypts a normalised feature `xi` as:

```
m = round(xi × 10^6),  exponent = -6
decoded = m × 10^(-6) = xi          ← BASE=10 convention
```

Python's `phe` library defaults to **BASE=16**:

```python
priv.decrypt(enc)  # interprets exponent as BASE=16
→ m × 16^(-6) = xi × (10/16)^6 ≈ xi × 0.0596   ← WRONG, off by ~16.8×
```

### The fix — `Base10EncodedNumber` (encryption.py)

```python
class Base10EncodedNumber(EncodedNumber):
    BASE = 10
    LOG2_BASE = math.log(10, 2)
```

This subclasses phe's `EncodedNumber` using phe's own documented extension point.
Two things it fixes:

1. **Weight encoding** — `enc * Base10EncodedNumber.encode(pub, wi)` passes a
   `Base10EncodedNumber` instance; phe's `isinstance(other, EncodedNumber)` check
   uses it directly instead of calling `EncodedNumber.encode()` with BASE=16.

2. **Decoding** — `priv.decrypt_encoded(result, Encoding=Base10EncodedNumber).decode()`
   computes `raw_int × 10^exponent` instead of `raw_int × 16^exponent`.

This is NOT replacing the phe library. All Paillier cryptographic operations
(`pow(c, k, n²)`, `c1×c2 mod n²`, key generation, `raw_decrypt`) still come from phe.
Only the fixed-point encoding convention (BASE=16 → BASE=10) is overridden.

### The two inference paths

Both endpoints exist and both are called simultaneously from the frontend:

| Endpoint | Function | Approach | Exponent |
|---|---|---|---|
| `POST /api/predict` | `encrypted_linear_inference` | phe + Base10EncodedNumber | dynamic (~-24) |
| `POST /api/predict-raw` | `encrypted_linear_inference_raw` | raw Z_{n²} arithmetic | fixed -12 |

The frontend calls both in parallel (`Promise.all`), decrypts both, and shows a
side-by-side comparison panel proving the outputs are identical (Δ < 1e-4).

---

## Data Flow (9 Steps)

```
1. User enters 4 raw values (e.g. glucose=120, BMI=25, age=45, BP=80)
2. Browser normalises: (value - min) / (max - min) → [0.4, 0.3, 0.3697, 0.25]
3. Browser generates 2048-bit Paillier keypair (private key never leaves browser)
4. Browser encrypts each:
      m = round(xi × 10^6)
      gm = (1 + m×n) mod n²        ← g = n+1 shortcut
      ciphertext = gm × r^n mod n²  ← r is fresh random per value
      → {ciphertext: "huge number", exponent: -6}
5. POST /api/predict + POST /api/predict-raw (parallel)
      {public_key: {n: "..."}, encrypted_features: [{ciphertext, exponent}, ...]}
6. Server reconstructs public key from n (has no λ, μ — cannot decrypt)
7. Server computes E(score) = Σ wᵢ·E(xᵢ) + b using homomorphic properties:
      E(a)^k mod n² = E(k×a)    ← scalar multiplication
      E(a) × E(b) mod n² = E(a+b) ← addition
   Returns {ciphertext: "...", exponent: -24}  — NEVER decrypts
8. Browser decrypts with private key (λ, μ):
      cl = c^λ mod n²
      m  = L(cl) × μ mod n
      signed = m > n/2 ? m - n : m
      score = signed / 10^(-exponent)   ← reads exponent from response
9. probability = sigmoid(score)
   risk = probability > threshold ? 'HIGH' : 'LOW'
```

---

## Key Findings from Development

### phe wraps negative weights (confirmed by test)

`Base10EncodedNumber.encode(pub, -13.922)` stores `n - abs_int` (a huge positive
close to n), NOT a negative integer. This means `_raw_mul` does:

```
pow(ciphertext, n - abs_int, n²)    ← ~2048-bit exponent (SLOW)
```

vs positive weights:

```
pow(ciphertext, 57553362300000000, n²)  ← ~57-bit exponent (fast)
```

**Correctness is unaffected** — `E(m)^(n-k) = E(-k×m)` holds in Paillier.
**Performance**: negative weights are ~35× slower per multiplication.
The anemia model has 3 negative weights (`[-13.922, -0.194, -0.741]`).
The raw path handles negatives faster (Python's `pow(c, -k, n²)` uses modular inverse).

### The claim "Base10 breaks for anemia" is FALSE

All tests pass for the anemia model including 200 random vectors and 5 fixed
test vectors. The claim confused "slow" with "broken".

### The "5755336 / 707022" integers belong to the raw path, not phe

`round(5.75533623 × 10^6) = 5755336` — this is the **raw arithmetic path's** scale.
phe + Base10EncodedNumber uses float precision (~10^16), producing `57553362300000000`.
Both round-trip correctly; they're just different precision levels.

### Collaborator reproducibility

`trained_weights.json` IS committed to git (do not gitignore it).
Dataset CSVs are gitignored — collaborators cannot retrain without downloading them.
`requirements.txt` uses `>=` (unpinned) — run `pip freeze > requirements.txt` to pin
exact versions for fully reproducible results across machines.

---

## Test Structure (test_interop.py)

| Class | What it tests |
|---|---|
| `TestBugExists` | Proves the OLD naive BASE=16 path was wrong — off by `(10/16)^6` |
| `TestBase10Fix` | Proves Base10EncodedNumber gives correct results |
| `TestStatisticalCorrectness` | 200 random vectors × 3 models, all within 1e-4 |
| `TestExponentRoundtrip` | JS-simulated decode works for both phe path AND raw path |
| `TestBase10EncodingClaim` | Verifies/refutes claim about positive vs negative weights |

Run tests:
```bash
cd heimdall/backend
pytest tests/test_interop.py -v -s
```

---

## Important Code Invariants

- **Private key never leaves the browser** — only `n` (public modulus) is sent to server
- **Server never decrypts** — only homomorphic operations on ciphertexts
- **Exponent on wire is BASE=10** — `exponent=-6` means `integer × 10^(-6)`
- **JS `decryptResult()`** reads exponent from response and applies `signed / 10^(-exp)`
  — works for any negative BASE=10 exponent (phe dynamic or raw fixed -12)
- **`trained_weights.json` must exist** at `backend/core/trained_weights.json` —
  otherwise backend silently falls back to placeholder weights with completely
  different magnitudes, causing wrong predictions

---

## Running the Project

```bash
# Backend
cd heimdall/backend
pip install -r requirements.txt
uvicorn api.main:app --reload

# Frontend (separate terminal)
cd heimdall/frontend
npm install
npm start
```

Frontend: http://localhost:3000 · Backend: http://localhost:8000 · Docs: http://localhost:8000/docs
