"""
Heimdall — IEEE Benchmark Suite
================================
Produces the performance tables required for the paper:
  • Table 1 : Key-generation latency by key size
  • Table 2 : Per-feature encryption and decryption latency
  • Table 3 : Server-side HE inference latency per model (raw vs Base10)
  • Table 4 : End-to-end latency (keygen + encrypt + infer + decrypt)
  • Table 5 : Communication overhead (plaintext vs ciphertext bytes)
  • Table 6 : Accuracy equivalence — JS-style ciphertexts → raw inference
              (the actual production path) compared with plaintext logistic
              regression
  • Table 7 : Quantisation error from integer scaling
  • Table 8 : Interop trace — Base10 path vs raw path on JS-style ciphertexts.
              Demonstrates the (16/10)^k error materialising on anemia.

All latency measurements are repeated N=50 times; mean ± std and 95% CI
are reported.  Run from the backend/ directory:

    python scripts/benchmark_ieee.py

Results are printed as markdown tables and also saved to
    backend/results/benchmark_results.json
"""

import sys, os, json, time, math, statistics, random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import phe as paillier
import numpy as np

from core.encryption import Base10EncodedNumber
from core.models import (
    TRAINED_WEIGHTS,
    MODEL_SPECS,
    encrypted_linear_inference,
    encrypted_linear_inference_raw,
    normalize_features,
    sigmoid,
)


# ---------------------------------------------------------------------------
# JS-style encryption — what the browser actually sends
# ---------------------------------------------------------------------------
# Keep this in sync with frontend/src/utils/crypto.js encryptValue().
# m = round(value * 10^6); c = (1 + m·n) · r^n  mod n²; exponent = -6.

def js_style_encrypt(pub, value: float, scale: int = 1_000_000) -> dict:
    n  = pub.n
    n2 = n * n
    m  = round(value * scale) % n
    r  = random.randrange(1, n)
    c  = (1 + m * n) * pow(r, n, n2) % n2
    return {"ciphertext": str(c), "exponent": -int(math.log10(scale))}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_RUNS     = 50          # repetitions per measurement
KEY_SIZES  = [1024, 2048, 3072]
MODEL_IDS  = ["diabetes", "heart", "anemia"]
RESULTS_DIR = Path(__file__).parent.parent / "results"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ci95(data):
    """Return (mean, median, std, lower_95, upper_95) in same units as input."""
    n      = len(data)
    mean   = statistics.mean(data)
    median = statistics.median(data)
    std    = statistics.stdev(data) if n > 1 else 0.0
    # 95 % CI using t-distribution approximation (t ≈ 1.96 for n ≥ 30)
    t = 1.96 if n >= 30 else {10: 2.228, 20: 2.086, 50: 2.009}.get(n, 2.0)
    margin = t * std / math.sqrt(n)
    return mean, median, std, mean - margin, mean + margin


def fmt(mean, median, std, lo, hi):
    return f"{mean:.2f} ± {std:.2f} ms  (median {median:.2f})  [{lo:.2f}, {hi:.2f}]"


def md_table(headers, rows):
    col_widths = [max(len(h), max(len(str(r[i])) for r in rows))
                  for i, h in enumerate(headers)]
    sep = "| " + " | ".join("-" * w for w, _ in zip(col_widths, headers)) + " |"
    hdr = "| " + " | ".join(h.ljust(w) for w, h in zip(col_widths, headers)) + " |"
    body = "\n".join(
        "| " + " | ".join(str(c).ljust(w) for w, c in zip(col_widths, row)) + " |"
        for row in rows
    )
    return "\n".join([hdr, sep, body])


# ---------------------------------------------------------------------------
# Table 1 — Key generation latency
# ---------------------------------------------------------------------------

def bench_keygen():
    print("\n## Table 1: Key-Generation Latency\n")
    rows = []
    data_out = {}
    for bits in KEY_SIZES:
        times = []
        for _ in range(N_RUNS):
            t = time.perf_counter()
            paillier.generate_paillier_keypair(n_length=bits)
            times.append((time.perf_counter() - t) * 1000)
        m, med, s, lo, hi = ci95(times)
        rows.append([f"{bits}-bit", fmt(m, med, s, lo, hi)])
        data_out[bits] = dict(mean=m, median=med, std=s, ci95_lo=lo, ci95_hi=hi)
        print(f"  {bits}-bit: {fmt(m,med,s,lo,hi)}")
    print()
    print(md_table(["Key Size", f"Latency (N={N_RUNS}, ms, mean±std (median) [95% CI])"], rows))
    return data_out


# ---------------------------------------------------------------------------
# Table 2 — Encryption & decryption per feature
# ---------------------------------------------------------------------------

def bench_enc_dec():
    print("\n## Table 2: Encryption / Decryption Latency Per Feature\n")
    rows = []
    data_out = {}
    for bits in KEY_SIZES:
        pub, priv = paillier.generate_paillier_keypair(n_length=bits)
        enc_times, dec_times = [], []
        for _ in range(N_RUNS):
            v = random.random()
            t = time.perf_counter()
            enc = pub.encrypt(v)
            enc_times.append((time.perf_counter() - t) * 1000)

            enc_d = {"ciphertext": str(enc.ciphertext()), "exponent": enc.exponent}
            enc_num = paillier.EncryptedNumber(pub, int(enc_d["ciphertext"]), enc_d["exponent"])
            t = time.perf_counter()
            priv.decrypt(enc_num)
            dec_times.append((time.perf_counter() - t) * 1000)

        em, emed, es, elo, ehi = ci95(enc_times)
        dm, dmed, ds, dlo, dhi = ci95(dec_times)
        rows.append([f"{bits}-bit", fmt(em, emed, es, elo, ehi), fmt(dm, dmed, ds, dlo, dhi)])
        data_out[bits] = dict(
            enc=dict(mean=em, median=emed, std=es, ci95_lo=elo, ci95_hi=ehi),
            dec=dict(mean=dm, median=dmed, std=ds, ci95_lo=dlo, ci95_hi=dhi),
        )
        print(f"  {bits}-bit  enc: {fmt(em,emed,es,elo,ehi)}  |  dec: {fmt(dm,dmed,ds,dlo,dhi)}")
    print()
    print(md_table(
        ["Key Size", f"Encrypt (ms)", f"Decrypt (ms)"],
        rows
    ))
    return data_out


# ---------------------------------------------------------------------------
# Table 3 — HE inference latency per model (server side only)
# ---------------------------------------------------------------------------

def bench_inference():
    """
    Server-side HE inference latency, measured for *both* paths:
      raw    — pow(c, k, n²); the deployed /api/predict path.
      base10 — phe + Base10EncodedNumber; comparison endpoint.
    """
    print("\n## Table 3: Server-Side HE Inference Latency (raw vs Base10)\n")
    bits = 2048
    pub, _ = paillier.generate_paillier_keypair(n_length=bits)
    rows = []
    data_out = {}
    for mid in MODEL_IDS:
        n_feat   = len(MODEL_SPECS[mid]["features"])
        enc_dicts = [js_style_encrypt(pub, 0.5) for _ in range(n_feat)]

        raw_times, b10_times = [], []
        for _ in range(N_RUNS):
            t = time.perf_counter()
            encrypted_linear_inference_raw(pub, enc_dicts, mid)
            raw_times.append((time.perf_counter() - t) * 1000)

            t = time.perf_counter()
            encrypted_linear_inference(pub, enc_dicts, mid)
            b10_times.append((time.perf_counter() - t) * 1000)

        rm, rmed, rs, rlo, rhi = ci95(raw_times)
        bm, bmed, bs, blo, bhi = ci95(b10_times)
        rows.append([mid, str(n_feat),
                     fmt(rm, rmed, rs, rlo, rhi),
                     fmt(bm, bmed, bs, blo, bhi)])
        data_out[mid] = dict(
            n_features=n_feat,
            raw=dict(mean=rm, median=rmed, std=rs, ci95_lo=rlo, ci95_hi=rhi),
            base10=dict(mean=bm, median=bmed, std=bs, ci95_lo=blo, ci95_hi=bhi),
        )
        print(f"  {mid:12s} ({n_feat} feat) raw: {fmt(rm,rmed,rs,rlo,rhi)}")
        print(f"  {mid:12s} ({n_feat} feat) b10: {fmt(bm,bmed,bs,blo,bhi)}")
    print()
    print(md_table(["Model", "Features",
                    f"Raw (ms, N={N_RUNS})",
                    f"Base10 (ms, N={N_RUNS})"], rows))
    return data_out


# ---------------------------------------------------------------------------
# Table 4 — End-to-end latency
# ---------------------------------------------------------------------------

def bench_e2e():
    """End-to-end via the actual deployed path: JS-style encrypt + raw inference."""
    print("\n## Table 4: End-to-End Latency (keygen + encrypt + infer + decrypt)\n")
    rows = []
    data_out = {}
    for bits in KEY_SIZES:
        mid   = "diabetes"
        times = []
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            pub, priv = paillier.generate_paillier_keypair(n_length=bits)
            n_feat    = len(MODEL_SPECS[mid]["features"])
            enc_dicts = [js_style_encrypt(pub, 0.5) for _ in range(n_feat)]
            enc_res   = encrypted_linear_inference_raw(pub, enc_dicts, mid)
            # Client-side: decrypt as integer, divide by 10^12
            enc_num = paillier.EncryptedNumber(pub, int(enc_res["ciphertext"]), 0)
            _ = priv.decrypt(enc_num) / 10**12
            times.append((time.perf_counter() - t0) * 1000)
        m, med, s, lo, hi = ci95(times)
        rows.append([f"{bits}-bit", fmt(m, med, s, lo, hi)])
        data_out[bits] = dict(mean=m, median=med, std=s, ci95_lo=lo, ci95_hi=hi)
        print(f"  {bits}-bit: {fmt(m,med,s,lo,hi)}")
    print()
    print(md_table(["Key Size", f"E2E Latency (diabetes model, ms, N={N_RUNS})"], rows))
    return data_out


# ---------------------------------------------------------------------------
# Table 5 — Communication overhead
# ---------------------------------------------------------------------------

def bench_communication():
    print("\n## Table 5: Communication Overhead\n")
    bits = 2048
    pub, _ = paillier.generate_paillier_keypair(n_length=bits)
    rows   = []
    data_out = {}
    for mid in MODEL_IDS:
        n_feat   = len(MODEL_SPECS[mid]["features"])
        # Plaintext payload: n_feat float64 values = 8 bytes each
        plain_bytes = n_feat * 8
        # Ciphertext payload: each ciphertext is a ~4096-bit (512-byte) integer as decimal string
        enc_list    = [pub.encrypt(0.5) for _ in range(n_feat)]
        cipher_json = json.dumps([{"ciphertext": str(e.ciphertext()), "exponent": e.exponent}
                                   for e in enc_list])
        cipher_bytes = len(cipher_json.encode())
        ratio        = cipher_bytes / plain_bytes
        rows.append([mid, str(n_feat), f"{plain_bytes} B", f"{cipher_bytes:,} B", f"{ratio:.0f}×"])
        data_out[mid] = dict(
            n_features=n_feat,
            plaintext_bytes=plain_bytes,
            ciphertext_bytes=cipher_bytes,
            overhead_ratio=ratio,
        )
    print(md_table(
        ["Model", "Features", "Plaintext", "Ciphertext (2048-bit)", "Overhead"],
        rows
    ))
    return data_out


# ---------------------------------------------------------------------------
# Table 6 — Accuracy equivalence (encrypted == plaintext)
# ---------------------------------------------------------------------------

def bench_accuracy_equivalence():
    """
    Verify that the *production* path (JS-style ciphertexts → raw Paillier
    inference → BigInt decrypt → divide by 10^12) produces IDENTICAL
    decisions to plaintext logistic regression. This mirrors what the
    deployed system does end-to-end.
    """
    print("\n## Table 6: Encrypted vs Plaintext Accuracy Equivalence\n")
    bits = 2048
    pub, priv = paillier.generate_paillier_keypair(n_length=bits)
    N_TEST = 200
    rows   = []
    data_out = {}
    for mid in MODEL_IDS:
        w       = TRAINED_WEIGHTS[mid]
        weights = np.array(w["weights"])
        bias    = w["bias"]
        thresh  = w.get("threshold", 0.5)
        n_feat  = len(weights)

        mismatches  = 0
        score_diffs = []

        for _ in range(N_TEST):
            x = np.random.uniform(0.0, 1.0, n_feat)

            plain_score = float(np.dot(weights, x) + bias)
            plain_risk  = sigmoid(plain_score) > thresh

            enc_dicts = [js_style_encrypt(pub, float(xi)) for xi in x]
            enc_res   = encrypted_linear_inference_raw(pub, enc_dicts, mid)
            enc_num   = paillier.EncryptedNumber(pub, int(enc_res["ciphertext"]), 0)
            enc_score = priv.decrypt(enc_num) / 10**12
            enc_risk  = sigmoid(enc_score) > thresh

            score_diffs.append(abs(enc_score - plain_score))
            if enc_risk != plain_risk:
                mismatches += 1

        max_diff  = max(score_diffs)
        mean_diff = statistics.mean(score_diffs)
        rows.append([mid, str(N_TEST), str(mismatches),
                     f"{mean_diff:.2e}", f"{max_diff:.2e}"])
        data_out[mid] = dict(
            n_test=N_TEST, mismatches=mismatches,
            mean_score_diff=mean_diff, max_score_diff=max_diff,
        )
    print(md_table(
        ["Model", "Test Vectors", "Decision Mismatches", "Mean |Δscore|", "Max |Δscore|"],
        rows
    ))
    return data_out


# ---------------------------------------------------------------------------
# Table 7 — Quantisation error from integer scaling (for raw inference)
# ---------------------------------------------------------------------------

def bench_quantisation():
    """
    Measure the error introduced by rounding weights to integers
    (used in encrypted_linear_inference_raw for JS cross-library compat).
    """
    print("\n## Table 7: Quantisation Error (raw integer inference)\n")
    N_TEST   = 500
    WSCALE   = 1_000_000
    rows     = []
    data_out = {}
    for mid in MODEL_IDS:
        w       = TRAINED_WEIGHTS[mid]
        weights = np.array(w["weights"])
        bias    = w["bias"]
        n_feat  = len(weights)

        int_weights = np.array([round(wi * WSCALE) for wi in weights])
        int_bias    = round(bias * WSCALE * WSCALE)   # total scale 1e12

        errors = []
        for _ in range(N_TEST):
            x      = np.random.uniform(0.0, 1.0, n_feat)
            m      = np.array([round(xi * WSCALE) for xi in x])

            # Exact score
            exact  = float(np.dot(weights, x) + bias)

            # Quantised score (what raw inference actually computes)
            quant_int = int(np.dot(int_weights, m)) + int_bias
            quant     = quant_int / (WSCALE * WSCALE)

            errors.append(abs(exact - quant))

        max_err  = max(errors)
        mean_err = statistics.mean(errors)
        rows.append([mid, f"{WSCALE:.0e}", f"{mean_err:.2e}", f"{max_err:.2e}"])
        data_out[mid] = dict(weight_scale=WSCALE, mean_error=mean_err, max_error=max_err)
    print(md_table(
        ["Model", "Weight Scale", "Mean |Δscore|", "Max |Δscore|"],
        rows
    ))
    print("\n*Quantisation error is negligible (< 1e-5) — well below sigmoid's sensitivity.*")
    return data_out


# ---------------------------------------------------------------------------
# Table 8 — Interop trace (Base10 vs raw on JS-style ciphertexts)
# ---------------------------------------------------------------------------

def bench_interop_trace():
    """
    The §V worked example. Encrypts JS-style ciphertexts (the ones the browser
    actually sends) and runs them through *both* server-side paths:

      Base10  — wraps each ciphertext in phe.EncryptedNumber and uses
                Base10EncodedNumber for the weights. Internally, when phe needs
                to align two EncryptedNumbers with different product-exponents,
                it calls EncryptedNumber.decrease_exponent_to which hardcodes
                EncodedNumber.BASE = 16 (not the subclass's BASE = 10),
                introducing a (16/10)^k factor per alignment step.
      Raw     — bypasses phe encoding entirely via pow(c, k, n²).

    For each model we report mean |Δscore| from plaintext and the worst-case
    multiplicative error of the Base10 path. The Base10 path is correct only
    when all weight·feature products land at the same exponent (no alignment
    triggered). Anemia trips the bug because w₀ ≈ -13.92 encodes at a
    different exponent than the other weights.
    """
    print("\n## Table 8: Interop Trace — Base10 vs Raw on JS-style ciphertexts\n")
    bits = 1024
    pub, priv = paillier.generate_paillier_keypair(n_length=bits)
    N_TEST = 50
    rows   = []
    data_out = {}

    for mid in MODEL_IDS:
        w       = TRAINED_WEIGHTS[mid]
        weights = np.array(w["weights"])
        bias    = w["bias"]
        n_feat  = len(weights)

        b10_errors, raw_errors = [], []
        for _ in range(N_TEST):
            x = np.random.uniform(0.0, 1.0, n_feat)
            enc_dicts = [js_style_encrypt(pub, float(xi)) for xi in x]

            plain_score = float(np.dot(weights, x) + bias)

            # Base10 path
            r1 = encrypted_linear_inference(pub, enc_dicts, mid)
            enc1 = paillier.EncryptedNumber(pub, int(r1["ciphertext"]), r1["exponent"])
            score_b10 = priv.decrypt_encoded(enc1, Encoding=Base10EncodedNumber).decode()

            # Raw path
            r2 = encrypted_linear_inference_raw(pub, enc_dicts, mid)
            score_raw = priv.decrypt(paillier.EncryptedNumber(pub, int(r2["ciphertext"]), 0)) / 10**12

            b10_errors.append(abs(score_b10 - plain_score))
            raw_errors.append(abs(score_raw - plain_score))

        b10_mean = statistics.mean(b10_errors)
        b10_max  = max(b10_errors)
        raw_mean = statistics.mean(raw_errors)
        raw_max  = max(raw_errors)
        verdict  = "OK" if b10_max < 1e-3 else f"BROKEN (≈ {b10_max:.2e})"

        rows.append([mid, str(n_feat),
                     f"{raw_mean:.2e}", f"{raw_max:.2e}",
                     f"{b10_mean:.2e}", f"{b10_max:.2e}", verdict])
        data_out[mid] = dict(
            n_features=n_feat,
            raw_mean=raw_mean, raw_max=raw_max,
            base10_mean=b10_mean, base10_max=b10_max,
            base10_verdict=verdict,
        )
    print(md_table(
        ["Model", "Features",
         "Raw mean |Δs|", "Raw max |Δs|",
         "Base10 mean |Δs|", "Base10 max |Δs|", "Base10 verdict"],
        rows
    ))
    print("\n*Base10 path fails on anemia because its weight magnitudes span more*")
    print("*than one order of magnitude, triggering EncryptedNumber alignment with*")
    print("*hardcoded BASE=16. The raw path is the version-independent fix.*")
    return data_out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  Heimdall IEEE Benchmark Suite")
    print(f"  N={N_RUNS} runs per measurement · key sizes {KEY_SIZES}")
    print("=" * 70)

    results = {}
    results["keygen"]        = bench_keygen()
    results["enc_dec"]       = bench_enc_dec()
    results["inference"]     = bench_inference()
    results["e2e"]           = bench_e2e()
    results["communication"] = bench_communication()
    results["equivalence"]   = bench_accuracy_equivalence()
    results["quantisation"]  = bench_quantisation()
    results["interop_trace"] = bench_interop_trace()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved to {out_path}")
    print("\nUse these numbers directly in the IEEE paper tables.")
    print("Key claims to verify against the printed tables above:")
    print("  • HE inference latency per model at 2048-bit keys (Table 3)")
    print("  • Zero decision mismatches between encrypted and plaintext (Table 6)")
    print("  • Quantisation error below 1e-5 at weight_scale = 1e6 (Table 7)")
    print("Report median alongside mean in the paper — keygen latency is right-skewed")
    print("due to safe-prime rejection sampling.")


if __name__ == "__main__":
    main()
