/**
 * Heimdall - API Service
 * Handles all communication with the FastAPI backend.
 */

import axios from 'axios';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' },
});

/**
 * Fetch available models and their feature specs.
 * @returns {Promise<Object>} models dictionary
 */
export async function fetchModels() {
  const { data } = await api.get('/api/models');
  return data.models;
}

/**
 * Submit encrypted features for inference.
 *
 * @param {string} modelId
 * @param {string} publicKeyN  - The public key modulus (shared with server)
 * @param {Array}  encryptedFeatures - [{ciphertext, exponent}, ...]
 * @returns {Promise<Object>} { encrypted_result, inference_time_ms, ... }
 */
export async function predict(modelId, publicKeyN, encryptedFeatures) {
  const { data } = await api.post('/api/predict', {
    model_id: modelId,
    public_key: { n: publicKeyN },
    encrypted_features: encryptedFeatures,
  });
  return data;
}

/**
 * Submit encrypted features for inference using raw Paillier arithmetic.
 * Parallel to predict() but calls /api/predict-raw (no phe encoding layer).
 *
 * @param {string} modelId
 * @param {string} publicKeyN
 * @param {Array}  encryptedFeatures - [{ciphertext, exponent}, ...]
 * @returns {Promise<Object>} { encrypted_result, inference_time_ms, ... }
 */
export async function predictRaw(modelId, publicKeyN, encryptedFeatures) {
  const { data } = await api.post('/api/predict-raw', {
    model_id: modelId,
    public_key: { n: publicKeyN },
    encrypted_features: encryptedFeatures,
  });
  return data;
}

/**
 * Fetch benchmark stats for a given model.
 * @param {string} modelId
 */
export async function fetchBenchmark(modelId = 'diabetes') {
  const { data } = await api.get(`/api/benchmark?model_id=${modelId}`);
  return data;
}

/**
 * Health check.
 */
export async function healthCheck() {
  const { data } = await api.get('/');
  return data;
}

export async function keygen() {
  const { data } = await api.post('/api/keygen');
  return data;  // { session_id, public_key_n }
}

export async function decryptResult(sessionId, modelId, encryptedResult) {
  try {
    const { data } = await api.post('/api/decrypt-result', {
      session_id: sessionId,
      model_id: modelId,       // ADD THIS
      encrypted_result: encryptedResult,
    });
    return data;
  } catch (err) {
    if (err.response?.status === 404) throw new Error('SESSION_EXPIRED');
    throw err;
  }
}

export async function predictPlaintext(modelId, normalizedFeatures) {
  const { data } = await api.post('/api/predict-plaintext', {
    model_id: modelId,
    features: normalizedFeatures,
  });
  return data;
}
