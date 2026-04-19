import React, { useRef, useEffect } from 'react';
import styles from './AuditLog.module.css';

export default function AuditLog({ entries }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [entries]);

  return (
    <div className={styles.panel}>
      <div className={styles.title}>Audit Log — No PHI stored server-side</div>
      <div className={styles.log}>
        {entries.length === 0 && (
          <div className={styles.empty}>No events yet.</div>
        )}
        {entries.map((e, i) => (
          <div key={i} className={[styles.entry, styles[e.type] || ''].join(' ')}>
            <span className={styles.ts}>[{e.ts}]</span>
            <span className={styles.msg}>{e.msg}</span>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
