import React from 'react';
import styles from './MetricsBar.module.css';

export default function MetricsBar({ metrics }) {
  const { encTimeMs, inferTimeMs, totalTimeMs, keyBits } = metrics;

  const cards = [
    { label: 'Key Size',        value: keyBits    ? `${keyBits}-bit`      : '—' },
    { label: 'Encrypt (ms)',    value: encTimeMs  ? encTimeMs.toFixed(1)  : '—' },
    { label: 'Inference (ms)', value: inferTimeMs ? inferTimeMs.toFixed(1): '—' },
    { label: 'Total (ms)',      value: totalTimeMs ? totalTimeMs.toFixed(1): '—' },
  ];

  return (
    <div className={styles.grid}>
      {cards.map(c => (
        <div key={c.label} className={styles.card}>
          <div className={styles.val}>{c.value}</div>
          <div className={styles.lbl}>{c.label}</div>
        </div>
      ))}
    </div>
  );
}
