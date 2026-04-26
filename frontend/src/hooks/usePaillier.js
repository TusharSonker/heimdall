import { useState, useCallback } from 'react';
import {
  generateKeyPair,
  encryptVector,
  decryptResult,
  normalizeFeature,
  sigmoid,
} from '../utils/crypto';

export function usePaillier() {
  const [keyState, setKeyState] = useState({
    generated: false,
    generating: false,
    publicKeyN: null,
    publicKey: null,
    privateKey: null,
  });

  /**
   * Generate a fresh 2048-bit Paillier keypair in the browser.
   * Keys never leave the client — only the public modulus n is shared.
   */
  const genKeys = useCallback(async () => {
    setKeyState(s => ({ ...s, generating: true, generated: false }));
    try {
      const { publicKey, privateKey, n } = await generateKeyPair(2048);
      setKeyState({
        generated: true,
        generating: false,
        publicKeyN: n,
        publicKey,
        privateKey,
      });
      return { publicKeyN: n };
    } catch (err) {
      setKeyState(s => ({ ...s, generating: false }));
      throw err;
    }
  }, []);

  /**
   * Normalise raw feature values then encrypt each with the Paillier public key.
   * Returns the ciphertext array ready to POST to /api/predict.
   */
  const encryptFeatures = useCallback((rawValues, modelFeatures, publicKey) => {
    const t0 = performance.now();

    const normValues = rawValues.map((v, i) =>
      normalizeFeature(v, modelFeatures[i].min, modelFeatures[i].max)
    );

    const encryptedFeatures = encryptVector(normValues, publicKey);
    const encTimeMs = +(performance.now() - t0).toFixed(2);

    return { encryptedFeatures, normValues, encTimeMs };
  }, []);

  /**
   * Decrypt the server's encrypted result and interpret as a risk assessment.
   * score → sigmoid → probability → threshold comparison → HIGH / LOW
   */
  const decryptAndInterpret = useCallback((encryptedResult, privateKey, threshold = 0.5) => {
    const t0 = performance.now();
    const score       = decryptResult(encryptedResult, privateKey);
    const probability = sigmoid(score);
    const risk        = probability > threshold ? 'HIGH' : 'LOW';
    const decTimeMs   = +(performance.now() - t0).toFixed(2);
    return { score, probability, risk, decTimeMs };
  }, []);

  return { keyState, genKeys, encryptFeatures, decryptAndInterpret };
}
