import React from 'react';
import styles from './Pipeline.module.css';

const STEPS = [
  { id: 'validate',   label: 'Input Validation' },
  { id: 'normalize',  label: 'Feature Normalization' },
  { id: 'encrypt',    label: 'Paillier Encryption' },
  { id: 'infer',      label: 'Encrypted Inference (Server)' },
  { id: 'decrypt',    label: 'Decryption & Result' },
];

export default function Pipeline({ steps }) {
  return (
    <div className={styles.panel}>
      <div className={styles.title}>🔄 Encryption Pipeline</div>
      <div className={styles.steps}>
        {STEPS.map((step, i) => {
          const s = steps[step.id] || {};
          return (
            <div
              key={step.id}
              className={[
                styles.step,
                s.status === 'done'   ? styles.done   : '',
                s.status === 'active' ? styles.active : '',
                s.status === 'error'  ? styles.error  : '',
              ].join(' ')}
            >
              <div className={styles.icon}>{String(i + 1).padStart(2, '0')}</div>
              <div className={styles.info}>
                <div className={styles.name}>{step.label}</div>
                {s.detail && <div className={styles.detail}>{s.detail}</div>}
              </div>
              {s.time && <div className={styles.time}>{s.time}ms</div>}
            </div>
          );
        })}
      </div>
    </div>
  );
}
