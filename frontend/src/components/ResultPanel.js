import React from 'react';
import styles from './ResultPanel.module.css';

export default function ResultPanel({ result }) {
  if (!result) return null;

  const { risk, probability, score, modelLabel, encryptedResult, inferenceTimeMs } = result;
  const isHigh = risk === 'HIGH';

  return (
    <div className={[styles.panel, isHigh ? styles.high : styles.low].join(' ')}>
      <div className={styles.modelLabel}>{modelLabel} — Risk Assessment</div>
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
    </div>
  );
}
