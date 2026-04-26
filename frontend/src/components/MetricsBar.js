import React from 'react';
import styles from './MetricsBar.module.css';

export default function MetricsBar({ metrics }) {
  const { encTimeMs, pheInferTimeMs, rawInferTimeMs, totalTimeMs, keyBits } = metrics;

  const cards = [
    { label: 'Key Size',           value: keyBits          ? `${keyBits}-bit`             : '—' },
    { label: 'Encrypt (ms)',       value: encTimeMs        ? encTimeMs.toFixed(1)         : '—' },
    { label: 'phe Infer (ms)',     value: pheInferTimeMs   ? pheInferTimeMs.toFixed(1)    : '—', accent: true },
    { label: 'Raw Infer (ms)',     value: rawInferTimeMs   ? rawInferTimeMs.toFixed(1)    : '—', accent: true },
    { label: 'Total (ms)',         value: totalTimeMs      ? totalTimeMs.toFixed(1)       : '—' },
  ];

  return (
    <div className={styles.grid}>
      {cards.map(c => (
        <div key={c.label} className={[styles.card, c.accent ? styles.cardAccent : ''].join(' ')}>
          <div className={styles.val}>{c.value}</div>
          <div className={styles.lbl}>{c.label}</div>
        </div>
      ))}
    </div>
  );
}
