import React from 'react';
import styles from './ResultPanel.module.css';

export default function ResultPanel({ result }) {
  if (!result) return null;

  const { risk, probability, score, modelLabel, encryptedResult, inferenceTimeMs, comparison } = result;
  const isHigh = risk === 'HIGH';

  return (
    <div className={[styles.panel, isHigh ? styles.high : styles.low].join(' ')}>
      <div className={styles.modelLabel}>{modelLabel} — Risk Assessment (phe + Base10)</div>
      <div className={styles.riskValue}>{risk} RISK</div>
      <div className={styles.prob}>
        P(disease) = {(probability * 100).toFixed(2)}%
        &nbsp;·&nbsp; Decision boundary: 50%
        &nbsp;·&nbsp; Raw score: {score.toFixed(4)}
      </div>

      <div className={styles.divider} />

      <div className={styles.encSection}>
        <div className={styles.encLabel}>
          Encrypted ciphertext received from server — E(y):
        </div>
        <div className={styles.encValue}>
          {encryptedResult?.ciphertext?.slice(0, 80)}...
        </div>
        <div className={styles.encMeta}>
          Server inference: {inferenceTimeMs}ms &nbsp;·&nbsp;
          Exponent: {encryptedResult?.exponent} &nbsp;·&nbsp;
          Decrypted locally with private key
        </div>
      </div>

      {comparison && <ComparisonSection comparison={comparison} />}
    </div>
  );
}

function ComparisonSection({ comparison }) {
  const { phe, raw, delta, match } = comparison;
  return (
    <>
      <div className={styles.divider} />
      <div className={styles.compareLabel}>Method Comparison — Visual Proof</div>
      <div className={styles.compareGrid}>
        <PathCard label="phe + Base10EncodedNumber" path={phe} />
        <PathCard label="Raw Paillier Arithmetic" path={raw} />
      </div>
      <div className={[styles.deltaBar, match ? styles.deltaMatch : styles.deltaMismatch].join(' ')}>
        <span className={styles.deltaIcon}>{match ? '✓' : '✗'}</span>
        <span>Δ score = {delta.toExponential(3)}</span>
        <span className={styles.deltaVerdict}>
          {match ? 'OUTPUTS IDENTICAL  (< 1e-4)' : 'MISMATCH DETECTED'}
        </span>
      </div>
    </>
  );
}

function PathCard({ label, path }) {
  const isHigh = path.risk === 'HIGH';
  return (
    <div className={[styles.pathCard, isHigh ? styles.pathHigh : styles.pathLow].join(' ')}>
      <div className={styles.pathLabel}>{label}</div>
      <div className={[styles.pathRisk, isHigh ? styles.riskHigh : styles.riskLow].join(' ')}>
        {path.risk} RISK
      </div>
      <div className={styles.pathMeta}>P = {(path.probability * 100).toFixed(4)}%</div>
      <div className={styles.pathMeta}>score = {path.score.toFixed(6)}</div>
      <div className={styles.pathTime}>server: {path.inferenceTimeMs}ms</div>
    </div>
  );
}
