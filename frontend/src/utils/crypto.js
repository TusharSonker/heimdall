// ── Math helpers ──────────────────────────────────────────────────────────

function modPow(base, exp, mod) {
  let result = 1n;
  base = base % mod;
  while (exp > 0n) {
    if (exp % 2n === 1n) result = result * base % mod;
    exp = exp / 2n;
    base = base * base % mod;
  }
  return result;
}

function millerRabin(n, rounds = 12) {
  if (n < 2n) return false;
  if (n === 2n || n === 3n) return true;
  if (n % 2n === 0n) return false;
  let d = n - 1n, r = 0n;
  while (d % 2n === 0n) { d /= 2n; r++; }
  const witnesses = [2n, 3n, 5n, 7n, 11n, 13n, 17n, 19n, 23n, 29n, 31n, 37n];
  for (const a of witnesses) {
    if (a >= n) continue;
    let x = modPow(a, d, n);
    if (x === 1n || x === n - 1n) continue;
    let composite = true;
    for (let j = 0n; j < r - 1n; j++) {
      x = x * x % n;
      if (x === n - 1n) { composite = false; break; }
    }
    if (composite) return false;
  }
  return true;
}

function randomBigInt(bits) {
  const bytes = Math.ceil(bits / 8);
  const arr = new Uint8Array(bytes);
  window.crypto.getRandomValues(arr);
  arr[0] |= 0x80;
  return BigInt('0x' + Array.from(arr).map(b => b.toString(16).padStart(2, '0')).join(''));
}

async function generatePrime(bits) {
  while (true) {
    let p = randomBigInt(bits);
    if (p % 2n === 0n) p += 1n;
    if (millerRabin(p)) return p;
    await new Promise(r => setTimeout(r, 0)); // don't block UI
  }
}

function gcd(a, b) {
  while (b) { [a, b] = [b, a % b]; }
  return a;
}

function lcm(a, b) { return (a / gcd(a, b)) * b; }

function modInverse(a, m) {
  let [old_r, r] = [a, m];
  let [old_s, s] = [1n, 0n];
  while (r !== 0n) {
    const q = old_r / r;
    [old_r, r] = [r, old_r - q * r];
    [old_s, s] = [s, old_s - q * s];
  }
  return ((old_s % m) + m) % m;
}

function L(x, n) { return (x - 1n) / n; }

// ── Key storage ───────────────────────────────────────────────────────────

let _publicKey = null;
let _privateKey = null;

// ── Key generation ────────────────────────────────────────────────────────

export async function generateKeyPair(bits = 2048) {
  const half = bits / 2;
  let p, q, n;
  do {
    p = await generatePrime(half);
    q = await generatePrime(half);
    n = p * q;
  } while (p === q);

  const n2 = n * n;
  const g = n + 1n;                        // standard simplification
  const lambda = lcm(p - 1n, q - 1n);
  const mu = modInverse(L(modPow(g, lambda, n2), n), n);

  _publicKey = { n, n2, g };
  _privateKey = { lambda, mu, publicKey: _publicKey };

  return {
    publicKey: _publicKey,
    privateKey: _privateKey,
    n: n.toString(),
  };
}

// ── Encryption ────────────────────────────────────────────────────────────

export function encryptValue(value, publicKey) {
  const pk = publicKey || _publicKey;
  if (!pk) throw new Error('No public key. Call generateKeyPair() first.');

  const { n, n2 } = pk;

  // Scale float to integer — multiply by 1e6 to preserve 6 decimal places
  const scaled = BigInt(Math.round(value * 1_000_000));

  // Handle negatives: work in [0, n)
  const m = ((scaled % n) + n) % n;

  // Paillier encrypt: c = g^m * r^n mod n^2
  // Since g = n+1: g^m mod n^2 = (1 + m*n) mod n^2
  const gm = (1n + m * n) % n2;

  let r;
  const bits = n.toString(2).length;
  do { r = randomBigInt(bits) % n; } while (r === 0n);

  const rn = modPow(r, n, n2);
  const ciphertext = gm * rn % n2;

  return {
    ciphertext: ciphertext.toString(),
    exponent: -6,
  };
}

export function encryptVector(values, publicKey) {
  return values.map(v => encryptValue(v, publicKey));
}

// ── Decryption ────────────────────────────────────────────────────────────

export function decryptResult(encResult, privateKey) {
  const pk = privateKey || _privateKey;
  if (!pk) throw new Error('No private key.');

  const { lambda, mu, publicKey: { n, n2 } } = pk;
  const c = BigInt(encResult.ciphertext);

  // Standard Paillier decrypt
  const cl = modPow(c, lambda, n2);
  const lval = L(cl, n);
  const m = lval * mu % n;

  // Handle negative numbers
  const signed = m > n / 2n ? m - n : m;

  // The exponent comes from Python phe — it tells us the scaling factor
  // phe uses base 10: actual_value = signed * 10^exponent
  const exponent = encResult.exponent;  // e.g. -6, -15, -32 etc.

  console.log('[decrypt] signed:', signed.toString(), 'exponent:', exponent);

  if (exponent >= 0) {
    // positive exponent: multiply
    const scale = BigInt(10) ** BigInt(exponent);
    return Number(signed * scale);
  } else {
    // negative exponent: divide — use string method to avoid overflow
    const absExp = -exponent;
    const divisor = BigInt(10) ** BigInt(absExp);

    const intPart = signed / divisor;
    const rem = signed % divisor;

    // Convert remainder to decimal string, pad to absExp digits
    const remStr = (rem < 0n ? -rem : rem).toString().padStart(absExp, '0');
    const sign = signed < 0n ? '-' : '';
    const floatStr = `${sign}${intPart < 0n ? -intPart : intPart}.${remStr}`;

    const result = parseFloat(floatStr);
    console.log('[decrypt] floatStr:', floatStr, '→', result);
    return result;
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────

export function normalizeFeature(value, min, max) {
  return Math.max(0, Math.min(1, (value - min) / (max - min)));
}

export function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

// Testing 
// // SELF TEST — delete after confirming
// async function selfTest() {
//   console.log('[selfTest] Starting...');
//   const { publicKey, privateKey } = await generateKeyPair(512);
//   const tests = [0.2833, 0.5, 0.9999];
//   for (const v of tests) {
//     const enc = encryptValue(v, publicKey);
//     const dec = decryptResult(enc, privateKey);
//     const ok = Math.abs(dec - v) < 0.000002;
//     console.log(`[selfTest] ${v} → ${dec}  ${ok ? '✓' : '✗ FAIL'}`);
//   }
//   console.log('[selfTest] Done.');
// }
// selfTest();

// async function crossLibTest() {
//   console.log('[crossLibTest] Starting...');
//   const { publicKey, privateKey, n } = await generateKeyPair(512);

//   // Encrypt a known value
//   const testVal = 0.2833;
//   const enc = encryptValue(testVal, publicKey);
//   console.log('[crossLibTest] Encrypted:', enc.ciphertext.slice(0, 30) + '...');
//   console.log('[crossLibTest] Exponent:', enc.exponent);
//   console.log('[crossLibTest] Public key n:', n.slice(0, 30) + '...');

//   // Send to backend directly
//   const response = await fetch('http://localhost:8000/api/predict', {
//     method: 'POST',
//     headers: { 'Content-Type': 'application/json' },
//     body: JSON.stringify({
//       model_id: 'diabetes',
//       public_key: { n },
//       encrypted_features: [enc, enc, enc, enc],  // same value 4 times
//     }),
//   });

//   const data = await response.json();
//   console.log('[crossLibTest] Server response:', data);

//   if (data.encrypted_result) {
//     const score = decryptResult(data.encrypted_result, privateKey);
//     console.log('[crossLibTest] Decrypted score:', score);
//     console.log('[crossLibTest] Probability:', 1 / (1 + Math.exp(-score)));
//   } else {
//     console.log('[crossLibTest] Server error:', data.detail);
//   }
// }
// crossLibTest();