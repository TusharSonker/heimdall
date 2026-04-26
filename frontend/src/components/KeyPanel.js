import React from 'react';
import styles from './KeyPanel.module.css';

export default function KeyPanel({ keyState, onGenerate }) {
  const { n, generating, generated } = keyState;

  return (
    <div className={styles.panel}>
      <div className={styles.header}>
        <span className={styles.title}>🔐 Cryptographic Keys</span>
        <span className={styles.note}>Private key never leaves your browser</span>
        <button
          className={styles.btn}
          onClick={onGenerate}
          disabled={generating}
        >
          {generating ? 'Generating...' : generated ? 'Regenerate' : 'Generate Keys'}
        </button>
      </div>

      <div className={styles.grid}>
        <div className={styles.keyBox}>
          <div className={styles.label}>Public Key (n) — shared with server</div>
          <div className={styles.value}>
            {n ? '0x' + BigInt(n).toString(16).slice(0, 48) + '...' : '—'}
          </div>
        </div>
        <div className={styles.keyBox}>
          <div className={styles.label}>Private Key (λ, μ) — client only</div>
          <div className={styles.value}>
            {generated ? '████████████████████████████████ [hidden]' : '—'}
          </div>
        </div>
      </div>

      {generating && (
        <div className={styles.progress}>
          Generating 2048-bit Paillier key pair... this takes a few seconds.
        </div>
      )}
    </div>
  );
}
