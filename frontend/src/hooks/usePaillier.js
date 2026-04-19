import { useState, useCallback } from 'react';
import { predictPlaintext } from '../utils/api';

export function usePaillier() {
  const [keyState, setKeyState] = useState({
    generated: true,   // always ready — no JS crypto needed
    generating: false,
    publicKeyN: null,
    sessionId: null,
  });

  // No-op — keys generated server-side now
  const genKeys = useCallback(async () => {
    setKeyState(s => ({ ...s, generated: true }));
    return {};
  }, []);

  const encryptFeatures = useCallback((rawValues, modelFeatures) => {
    const t0 = performance.now();

    // Normalize features
    const normValues = rawValues.map((v, i) =>
      Math.max(0, Math.min(1,
        (v - modelFeatures[i].min) / (modelFeatures[i].max - modelFeatures[i].min)
      ))
    );

    console.log('[encryptFeatures] normValues:', normValues);

    // Return normalized values — server will encrypt with phe
    const encTimeMs = +(performance.now() - t0).toFixed(2);
    return {
      encryptedFeatures: normValues,  // plain normalized values sent over TLS
      normValues,
      encTimeMs,
    };
  }, []);

  const decryptAndInterpret = useCallback(async (encryptedResult, modelId) => {
    // Result already decrypted by server — just return it
    const { risk, probability, score } = encryptedResult;
    console.log('[decryptAndInterpret]', { score, probability, risk });
    return { score, probability, risk };
  }, []);

  return { keyState, genKeys, encryptFeatures, decryptAndInterpret };
}