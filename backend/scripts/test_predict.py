"""
Generates a real Paillier key pair, encrypts sample values,
calls /api/predict, and decrypts the result.
Run with: python scripts/test_predict.py
"""

import phe as paillier
import math
import requests

API = "http://localhost:8000"

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Step 1 — Generate real keys
print("Generating keys...")
pub, priv = paillier.generate_paillier_keypair(n_length=1024)  # 1024 for speed
print(f"Public key n = {str(pub.n)[:40]}...")

# Step 2 — Sample patient values (diabetes model)
# Normalize using the same ranges as the model
raw = {"glucose": 148, "bmi": 33.6, "age": 50, "bp": 72}
ranges = {"glucose": (0,300), "bmi": (10,60), "age": (1,120), "bp": (40,200)}

norm = {k: (v - ranges[k][0]) / (ranges[k][1] - ranges[k][0])
        for k, v in raw.items()}
print(f"Normalized: {norm}")

# Step 3 — Encrypt each feature
encrypted_features = []
for k, v in norm.items():
    enc = pub.encrypt(v)
    encrypted_features.append({
        "ciphertext": str(enc.ciphertext()),
        "exponent":   enc.exponent
    })
    print(f"Encrypted {k}: {str(enc.ciphertext())[:30]}...")

# Step 4 — Call the API
payload = {
    "model_id": "diabetes",
    "public_key": {"n": str(pub.n)},
    "encrypted_features": encrypted_features
}

print("\nCalling /api/predict...")
resp = requests.post(f"{API}/api/predict", json=payload)
print(f"Status: {resp.status_code}")

if resp.status_code != 200:
    print(f"Error: {resp.json()}")
    exit(1)

data = resp.json()
print(f"Inference time: {data['inference_time_ms']}ms")
print(f"Server note: {data['server_note']}")

# Step 5 — Decrypt the result (client-side only)
enc_result = data["encrypted_result"]
enc_number = paillier.EncryptedNumber(
    pub,
    int(enc_result["ciphertext"]),
    int(enc_result["exponent"])
)
score = priv.decrypt(enc_number)
prob  = sigmoid(score)
risk  = "HIGH" if prob > 0.5 else "LOW"

print(f"\nDecrypted score:  {score:.4f}")
print(f"P(diabetes):      {prob*100:.1f}%")
print(f"Risk:             {risk}")
print(f"\nInput was: {raw}")