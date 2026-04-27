"""
Heimdall - Model Training Script
=================================
Trains logistic regression models on real medical datasets,
extracts weights/bias, and writes them to:
  backend/core/trained_weights.json

This file is then loaded by backend/core/models.py at runtime
instead of using the hard-coded placeholder weights.

Usage:
  cd heimdall/backend
  source venv/bin/activate
  python scripts/train_models.py

Requirements:
  pip install scikit-learn pandas numpy joblib
  (already in requirements.txt)
"""

import os
import json
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
from sklearn.metrics import roc_curve

# ── Paths ─────────────────────────────────────────────────────────────────
BASE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, '..', 'data')
OUT_FILE = os.path.join(BASE, '..', 'core', 'trained_weights.json')


# ══════════════════════════════════════════════════════════════════════════
# 1. DIABETES
# ══════════════════════════════════════════════════════════════════════════
def train_diabetes():
    """
    Dataset: Pima Indians Diabetes (UCI)
    768 samples, 8 features, binary outcome (0=no diabetes, 1=diabetes)

    Uses all 8 native Pima features for full clinical fidelity:
      Pregnancies, Glucose, BloodPressure (diastolic),
      SkinThickness (triceps), Insulin (2-hr serum), BMI,
      DiabetesPedigreeFunction, Age.
    """
    path = os.path.join(DATA_DIR, 'diabetes.csv')
    _check_file(path, 'diabetes')

    cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
            'Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
    df = pd.read_csv(path, header=None, names=cols)

    # 0 in physiological columns encodes missing — replace with column median
    for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())

    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    X = df[features].values
    y = df['Outcome'].values

    return _fit_and_report('diabetes', features, X, y,
        feature_mins=[0,   0,   40,  0,   0,   10,  0.0, 1],
        feature_maxs=[17,  300, 200, 100, 900, 60,  2.5, 120],
    )


# ══════════════════════════════════════════════════════════════════════════
# 2. HEART DISEASE
# ══════════════════════════════════════════════════════════════════════════
def train_heart():
    """
    Dataset: Heart Disease Cleveland (UCI)
    303 samples, 13 features, target: 0 = no disease, 1-4 = disease
    We binarize target to 0/1.

    4 features used:
      - Age
      - Chol   (serum cholesterol mg/dl)
      - Trestbps (resting systolic blood pressure)
      - Fbs    (fasting blood sugar > 120 mg/dl, used as smoking proxy)
    """
    path = os.path.join(DATA_DIR, 'heart.csv')
    _check_file(path, 'heart')

    cols = ['age','sex','cp','trestbps','chol','fbs','restecg',
            'thalach','exang','oldpeak','slope','ca','thal','target']

    df = pd.read_csv(path, header=None, names=cols, na_values='?')
    df.dropna(inplace=True)

    # Binarize: 0 = no disease, 1 = disease (original has 0-4)
    df['target'] = (df['target'] > 0).astype(int)

    # 'fbs' (fasting blood sugar > 120) serves as smoking proxy in our UI
    features = ['age', 'thalach', 'trestbps', 'cp']
    X = df[features].values.astype(float)
    y = df['target'].values.astype(int)

    return _fit_and_report('heart', features, X, y,
        feature_mins=[1,   60,  80, 0],
        feature_maxs=[120, 220, 250, 3]
    )


# ══════════════════════════════════════════════════════════════════════════
# 3. ANEMIA
# ══════════════════════════════════════════════════════════════════════════
def train_anemia():
    """
    Dataset: Anemia Dataset (Kaggle)
    ~1421 samples, features: hemoglobin, RBC, MCV, MCH, MCHC, Result

    4 features used:
      - Hemoglobin
      - RBC
      - MCV
      - MCH
    """
    path = os.path.join(DATA_DIR, 'anemia.csv')
    _check_file(path, 'anemia')

    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

    # Try common column name variants
    col_map = {
        'hemoglobin': ['hemoglobin', 'hgb', 'hb'],
        'mch':        ['mch', 'mean_corpuscular_hemoglobin'],
        'mchc':       ['mchc', 'mean_corpuscular_hemoglobin_concentration'],
        'mcv':        ['mcv', 'mean_corpuscular_volume'],
        'target':     ['target', 'result', 'anemia', 'label', 'class'],
    }

    rename = {}
    for canonical, variants in col_map.items():
        for v in variants:
            if v in df.columns:
                rename[v] = canonical
                break

    df.rename(columns=rename, inplace=True)

    required = ['hemoglobin', 'mch', 'mchc', 'mcv', 'target']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"\n  Available columns: {list(df.columns)}")
        raise ValueError(
            f"Anemia CSV is missing columns: {missing}\n"
            f"Please check the dataset and map them in train_anemia()."
        )

    df.dropna(subset=required, inplace=True)

    # Binarize target if needed (some versions have 0/1, some have strings)
    if df['target'].dtype == object:
        df['target'] = df['target'].str.lower().map(
            {'yes': 1, 'no': 0, 'true': 1, 'false': 0,
             'anemic': 1, 'non-anemic': 0, '1': 1, '0': 0}
        )

    features = ['hemoglobin', 'mch', 'mchc', 'mcv']
    X = df[features].values.astype(float)
    y = df['target'].values.astype(int)

    return _fit_and_report('anemia', features, X, y,
                       feature_mins=[4,  15, 20,  50],
                       feature_maxs=[20, 40, 40, 120])


# ══════════════════════════════════════════════════════════════════════════
# Core training function
# ══════════════════════════════════════════════════════════════════════════
def _fit_and_report(name, feature_names, X, y,
                    feature_mins, feature_maxs):
    """
    Min-max normalize using fixed clinical ranges (not data-driven),
    train logistic regression, evaluate, extract weights.

    WHY fixed ranges instead of fit-on-train?
    Because at inference time the client normalizes features using
    the same fixed ranges. If we used data-driven scaler ranges,
    we'd need to store and ship them too. Fixed clinical ranges
    are self-contained and interpretable.
    """
    print(f"\n{'─'*50}")
    print(f"  Model: {name.upper()}")
    print(f"  Samples: {len(X)}  |  Features: {feature_names}")
    print(f"  Class balance: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Normalize using fixed clinical ranges
    mins = np.array(feature_mins, dtype=float)
    maxs = np.array(feature_maxs, dtype=float)
    X_norm = np.clip((X - mins) / (maxs - mins), 0, 1)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_norm, y, test_size=0.2, random_state=42, stratify=y
    )

    # Logistic Regression (linear model — required for HE compatibility)
    # C=1.0 is default regularization. Increase for less regularization.
    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42,
        solver='lbfgs',
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    auc       = roc_auc_score(y_test, y_prob)

    # 5-fold CV on full dataset
    cv_scores = cross_val_score(model, X_norm, y, cv=5, scoring='accuracy')

    print(f"\n  Test set metrics:")
    print(f"    Accuracy:  {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall:    {recall:.4f}")
    print(f"    F1:        {f1:.4f}")
    print(f"    AUC-ROC:   {auc:.4f}")
    print(f"  5-fold CV:   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    print(f"\n  Trained weights:")
    for fn, w in zip(feature_names, model.coef_[0]):
        print(f"    {fn:20s}: {w:+.6f}")
    print(f"    {'bias':20s}: {model.intercept_[0]:+.6f}")

    print(f"\n{classification_report(y_test, y_pred, target_names=['No Disease','Disease'])}")

    # TESTING
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    j_scores = tpr - fpr
    optimal_idx = j_scores.argmax()
    optimal_threshold = float(thresholds[optimal_idx])
    print(f"  Optimal threshold: {optimal_threshold:.4f} (vs default 0.5)")

    # Return weights for serialization
    return {
        "weights":      [round(float(w), 8) for w in model.coef_[0]],
        "bias":          round(float(model.intercept_[0]), 8),
        "threshold":     round(optimal_threshold, 4),
        "feature_names": feature_names,
        "feature_mins":  feature_mins,
        "feature_maxs":  feature_maxs,
        "metrics": {
            "accuracy":  round(accuracy, 4),
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
            "auc":       round(auc, 4),
            "cv_mean":   round(float(cv_scores.mean()), 4),
            "cv_std":    round(float(cv_scores.std()), 4),
            "n_samples": int(len(X)),
        }
    }


def _check_file(path, name):
    if not os.path.exists(path):
        print(f"\n  [ERROR] {name} dataset not found at: {path}")
        print(f"          Run: python scripts/download_data.py first.")
        raise SystemExit(1)


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("Heimdall — Model Training Pipeline")
    print("=" * 60)

    # Start from existing weights so models with missing datasets keep
    # the last successful training (avoids wiping anemia when its CSV
    # mirror is unreachable, etc.)
    results = {}
    if os.path.exists(OUT_FILE):
        with open(OUT_FILE) as f:
            results = json.load(f)
        print(f"Loaded existing weights for: {list(results.keys())}")

    trainers = [
        ('diabetes', train_diabetes),
        ('heart',    train_heart),
        ('anemia',   train_anemia),
    ]

    for model_id, fn in trainers:
        try:
            results[model_id] = fn()
        except SystemExit:
            print(f"  Skipping {model_id} — data file missing (preserving existing weights if any).")
        except Exception as e:
            print(f"\n  [ERROR] Training {model_id} failed: {e}")
            import traceback; traceback.print_exc()

    if not results:
        print("\n[FAIL] No models trained. Check data files.")
        sys.exit(1)

    # Save weights to JSON
    with open(OUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Trained weights saved to:")
    print(f"  {os.path.abspath(OUT_FILE)}")
    print("\nModels trained:", list(results.keys()))
    print("\nNext step: restart the backend server.")
    print("  uvicorn api.main:app --reload --port 8000")
    print("=" * 60)


if __name__ == '__main__':
    main()
