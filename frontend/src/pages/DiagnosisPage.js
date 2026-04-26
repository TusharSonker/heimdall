import React, { useState, useEffect, useCallback } from 'react';
import KeyPanel from '../components/KeyPanel';
import Pipeline from '../components/Pipeline';
import ResultPanel from '../components/ResultPanel';
import MetricsBar from '../components/MetricsBar';
import AuditLog from '../components/AuditLog';
import { usePaillier } from '../hooks/usePaillier';
import { fetchModels, predict, predictRaw } from '../utils/api';
import styles from './DiagnosisPage.module.css';

const INITIAL_STEPS = {
  validate:  { status: '', detail: 'Waiting for input...', time: null },
  normalize: { status: '', detail: 'Awaiting step 1...', time: null },
  encrypt:   { status: '', detail: 'Awaiting step 2...', time: null },
  infer:     { status: '', detail: 'Awaiting step 3...', time: null },
  decrypt:   { status: '', detail: 'Awaiting step 4...', time: null },
};

function now() {
  return new Date().toLocaleTimeString('en-US', { hour12: false });
}

export default function DiagnosisPage() {
  const [models, setModels]           = useState({});
  const [activeModel, setActiveModel] = useState('diabetes');
  const [fieldValues, setFieldValues] = useState({});
  const [steps, setSteps]             = useState(INITIAL_STEPS);
  const [result, setResult]           = useState(null);
  const [metrics, setMetrics]         = useState({ keyBits: 2048 });
  const [log, setLog]                 = useState([]);
  const [running, setRunning]         = useState(false);
  const [serverOnline, setServerOnline] = useState(null);

  const { keyState, genKeys, encryptFeatures, decryptAndInterpret } = usePaillier();

  const addLog = useCallback((msg, type = '') => {
    setLog(prev => [...prev, { ts: now(), msg, type }]);
  }, []);

  const setStep = useCallback((id, status, detail, time) => {
    setSteps(prev => ({
      ...prev,
      [id]: {
        status,
        detail: detail ?? prev[id].detail,
        time:   time   ?? prev[id].time,
      },
    }));
  }, []);

  // Connect to API and generate keys on mount
  useEffect(() => {
    fetchModels()
      .then(data => {
        setModels(data);
        setServerOnline(true);
        addLog('Connected to Heimdall API server', 'success');
      })
      .catch(() => {
        setServerOnline(false);
        addLog('API server offline — run: uvicorn api.main:app --reload', 'error');
      });
  }, [addLog]);

  // Generate keys once the server is confirmed online
  useEffect(() => {
    if (serverOnline === true && !keyState.generated && !keyState.generating) {
      addLog('Generating 2048-bit Paillier keypair in browser...', '');
      genKeys()
        .then(({ publicKeyN }) => {
          addLog(
            `Keypair ready — public key n: 0x${BigInt(publicKeyN).toString(16).slice(0, 16)}...`,
            'success'
          );
        })
        .catch(err => addLog('Key generation failed: ' + err.message, 'error'));
    }
  }, [serverOnline, keyState.generated, keyState.generating, genKeys, addLog]);

  const currentModel = models[activeModel];
  const features     = currentModel?.features || [];

  function handleFieldChange(id, value) {
    setFieldValues(prev => ({ ...prev, [id]: value }));
  }

  function sleep(ms) {
    return new Promise(r => setTimeout(r, ms));
  }

  async function runPipeline() {
    if (running) return;
    if (!serverOnline) { addLog('API server is offline', 'error'); return; }
    if (!keyState.generated) { addLog('Keys not yet ready — please wait', 'error'); return; }

    const rawValues = features.map(f => parseFloat(fieldValues[f.id] ?? ''));
    const invalid   = features.filter((_, i) => isNaN(rawValues[i]));
    if (invalid.length > 0) {
      addLog(`Missing fields: ${invalid.map(f => f.label).join(', ')}`, 'error');
      return;
    }

    setRunning(true);
    setResult(null);
    setSteps(INITIAL_STEPS);

    const t0 = performance.now();

    try {
      // ── Step 1: Validate ─────────────────────────────────────────────────
      setStep('validate', 'active', `Validating ${features.length} input features...`);
      await sleep(80);
      setStep('validate', 'done',
        rawValues.map((v, i) => `${features[i].id}=${v}`).join(', '),
        +(performance.now() - t0).toFixed(1)
      );
      addLog(`Input validated: ${features.length} features`, '');

      // ── Step 2: Normalise ─────────────────────────────────────────────────
      setStep('normalize', 'active', 'Applying min-max normalisation to [0, 1]...');
      await sleep(80);
      const { encryptedFeatures, normValues, encTimeMs } = encryptFeatures(
        rawValues, features, keyState.publicKey
      );
      // normValues come back alongside ciphertexts so we can display them
      setStep('normalize', 'done',
        `Normalised: [${normValues.map(v => v.toFixed(4)).join(', ')}]`,
        +(performance.now() - t0).toFixed(1)
      );
      addLog('Features normalised', '');

      // ── Step 3: Encrypt (client-side, Paillier 2048-bit) ─────────────────
      setStep('encrypt', 'active', `Encrypting ${features.length} features with Paillier public key...`);
      // encryptFeatures already ran synchronously above; show ciphertext preview
      const preview = encryptedFeatures
        .map(e => e.ciphertext.slice(0, 10) + '…')
        .join(', ');
      setStep('encrypt', 'done',
        `E(x): [${preview}] · exponent=${encryptedFeatures[0]?.exponent}`,
        encTimeMs
      );
      addLog(`Client-side Paillier encryption: ${encTimeMs}ms`, 'success');

      // ── Step 4: Server HE inference (both paths in parallel) ─────────────
      setStep('infer', 'active',
        'Server computing E(y) = Σ wᵢ·E(xᵢ) + b — phe path & raw path in parallel...'
      );
      const inferStart = performance.now();

      const [serverResp, serverRespRaw] = await Promise.all([
        predict(activeModel, keyState.publicKeyN, encryptedFeatures),
        predictRaw(activeModel, keyState.publicKeyN, encryptedFeatures),
      ]);
      const inferTimeMs = +(performance.now() - inferStart).toFixed(1);

      setStep('infer', 'done',
        `E(score) received · phe=${serverResp.inference_time_ms}ms · raw=${serverRespRaw.inference_time_ms}ms`,
        inferTimeMs
      );
      addLog(
        `Encrypted inference: phe=${serverResp.inference_time_ms}ms, raw=${serverRespRaw.inference_time_ms}ms`,
        'success'
      );

      // ── Step 5: Client-side decryption (both ciphertexts) ────────────────
      setStep('decrypt', 'active', 'Decrypting both E(score) ciphertexts with private key...');

      const { score: pheScore, probability: pheProb, risk: pheRisk, decTimeMs } =
        decryptAndInterpret(serverResp.encrypted_result, keyState.privateKey, serverResp.threshold);

      const { score: rawScore, probability: rawProb, risk: rawRisk } =
        decryptAndInterpret(serverRespRaw.encrypted_result, keyState.privateKey, serverRespRaw.threshold);

      const delta   = Math.abs(pheScore - rawScore);
      const isMatch = delta < 1e-4;
      const totalMs = +(performance.now() - t0).toFixed(1);

      setStep('decrypt', 'done',
        `phe: ${pheScore.toFixed(4)} → ${pheRisk} · raw: ${rawScore.toFixed(4)} → ${rawRisk} · Δ=${delta.toExponential(2)}`,
        totalMs
      );
      addLog(
        `phe: ${pheRisk} (${(pheProb * 100).toFixed(1)}%) · raw: ${rawRisk} (${(rawProb * 100).toFixed(1)}%) · Δ=${delta.toExponential(2)} · ${isMatch ? 'MATCH' : 'MISMATCH'}`,
        isMatch ? 'success' : 'error'
      );

      setResult({
        risk:            pheRisk,
        probability:     pheProb,
        score:           pheScore,
        modelLabel:      currentModel.label,
        encryptedResult: serverResp.encrypted_result,
        inferenceTimeMs: serverResp.inference_time_ms,
        decTimeMs,
        comparison: {
          phe: { score: pheScore, probability: pheProb, risk: pheRisk, inferenceTimeMs: serverResp.inference_time_ms },
          raw: { score: rawScore, probability: rawProb, risk: rawRisk, inferenceTimeMs: serverRespRaw.inference_time_ms },
          delta,
          match: isMatch,
        },
      });
      setMetrics({
        keyBits:         2048,
        encTimeMs,
        pheInferTimeMs:  serverResp.inference_time_ms,
        rawInferTimeMs:  serverRespRaw.inference_time_ms,
        totalTimeMs:     totalMs,
      });

    } catch (err) {
      console.error('Pipeline error:', err);
      addLog('Pipeline error: ' + err.message, 'error');
      // Mark the active step as errored
      setSteps(prev => {
        const updated = { ...prev };
        for (const k of Object.keys(updated)) {
          if (updated[k].status === 'active') {
            updated[k] = { ...updated[k], status: 'error', detail: err.message };
          }
        }
        return updated;
      });
    } finally {
      setRunning(false);
    }
  }

  const keysReady = keyState.generated && !keyState.generating;

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <div className={styles.logo}>HEIMDALL</div>
        <div className={styles.tagline}>Privacy-Preserving Medical Diagnosis · Paillier HE · E2E Encrypted</div>
        <div className={styles.statusBar}>
          <StatusPill
            active={keysReady}
            loading={keyState.generating}
            label={
              keyState.generating ? 'Keys: Generating…' :
              keysReady           ? 'Keys: Ready'       : 'Keys: Pending'
            }
          />
          <StatusPill
            active={serverOnline === true}
            danger={serverOnline === false}
            label={
              serverOnline === null  ? 'Server: Checking…' :
              serverOnline           ? 'Server: Online'    : 'Server: Offline'
            }
          />
          <StatusPill active label="TLS: Active" />
          <StatusPill active={keysReady} label={keysReady ? 'PHI: Encrypted' : 'PHI: Pending Keys'} />
        </div>
      </header>

      {serverOnline === false && (
        <div className={styles.offlineBanner}>
          ⚠ Backend offline. Run:{' '}
          <code>cd backend &amp;&amp; uvicorn api.main:app --reload</code>
        </div>
      )}

      <KeyPanel keyState={keyState} onGenerate={genKeys} />

      <div className={styles.modelTabs}>
        {Object.entries(models).map(([id, m]) => (
          <button
            key={id}
            className={[styles.tab, activeModel === id ? styles.tabActive : ''].join(' ')}
            onClick={() => {
              setActiveModel(id);
              setResult(null);
              setFieldValues({});
              setSteps(INITIAL_STEPS);
            }}
          >
            {m.label}
            <span className={styles.tabAcc}>{m.accuracy}%</span>
          </button>
        ))}
        {Object.keys(models).length === 0 && (
          <div className={styles.loadingModels}>Loading models from API…</div>
        )}
      </div>

      {features.length > 0 && (
        <div className={styles.formPanel}>
          <div className={styles.formGrid}>
            {features.map(f => (
              <div key={f.id} className={styles.field}>
                <label className={styles.fieldLabel}>{f.label}</label>
                <input
                  type="number"
                  className={styles.input}
                  placeholder={String((f.min + f.max) / 2)}
                  min={f.min}
                  max={f.max}
                  step="any"
                  value={fieldValues[f.id] ?? ''}
                  onChange={e => handleFieldChange(f.id, e.target.value)}
                />
                {f.hint && <span className={styles.hint}>{f.hint}</span>}
              </div>
            ))}
          </div>
          <button
            className={styles.runBtn}
            onClick={runPipeline}
            disabled={running || !serverOnline || !keysReady}
          >
            {running         ? '⟳ Running Pipeline…'   :
             !keysReady      ? '⏳ Awaiting Key Generation…' :
                               '⚡ Encrypt & Predict'}
          </button>
        </div>
      )}

      <Pipeline steps={steps} />
      <ResultPanel result={result} />
      <MetricsBar metrics={metrics} />
      <AuditLog entries={log} />
    </div>
  );
}

function StatusPill({ active, danger, loading, label }) {
  return (
    <div className={[
      styles.pill,
      active  ? styles.pillActive  : '',
      danger  ? styles.pillDanger  : '',
      loading ? styles.pillLoading : '',
    ].join(' ')}>
      <span className={styles.dot} />
      {label}
    </div>
  );
}
