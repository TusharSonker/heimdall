"""
Heimdall - Dataset Downloader
Downloads all 3 real datasets used for training:
  1. Pima Indians Diabetes     — UCI ML Repository
  2. Heart Disease (Cleveland) — UCI ML Repository
  3. Anemia                    — Kaggle (manual download required)

Run this BEFORE train_models.py:
  cd heimdall/backend
  source venv/bin/activate
  python scripts/download_data.py
"""

import urllib.request
import os
import sys

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)


def download(url: str, dest: str, name: str):
    if os.path.exists(dest):
        print(f"  [SKIP] {name} already exists at {dest}")
        return True
    print(f"  [DOWN] Downloading {name}...")
    try:
        urllib.request.urlretrieve(url, dest)
        size = os.path.getsize(dest)
        print(f"  [OK]   {name} saved ({size:,} bytes)")
        return True
    except Exception as e:
        print(f"  [ERR]  Failed to download {name}: {e}")
        return False


def main():
    print("=" * 60)
    print("Heimdall — Dataset Downloader")
    print("=" * 60)

    # ── 1. Pima Indians Diabetes ──────────────────────────────────
    print("\n[1/3] Pima Indians Diabetes Dataset")
    diabetes_url = (
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
        "pima-indians-diabetes.data.csv"
    )
    diabetes_dest = os.path.join(DATA_DIR, "diabetes.csv")
    download(diabetes_url, diabetes_dest, "Diabetes")

    # ── 2. Heart Disease (Cleveland) ─────────────────────────────
    print("\n[2/3] Heart Disease Dataset (Cleveland)")
    heart_url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "heart-disease/processed.cleveland.data"
    )
    heart_dest = os.path.join(DATA_DIR, "heart.csv")
    download(heart_url, heart_dest, "Heart Disease")

    # ── 3. Anemia ─────────────────────────────────────────────────
    print("\n[3/3] Anemia Dataset")
    print("  [INFO] The Anemia dataset requires a free Kaggle account.")
    print("         Manual download steps:")
    print("         1. Go to: https://www.kaggle.com/datasets/biswaranjanrao/anemia-dataset")
    print("         2. Click 'Download' (sign in if asked)")
    print("         3. Unzip and copy 'anemia.csv' to:")
    anemia_dest = os.path.join(DATA_DIR, "anemia.csv")
    print(f"            {os.path.abspath(anemia_dest)}")

    anemia_fallback_url = (
        "https://raw.githubusercontent.com/dsrscientist/dataset1/master/anemia.csv"
    )
    print(f"\n  [TRY]  Attempting fallback mirror...")
    download(anemia_fallback_url, anemia_dest, "Anemia (fallback)")

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Summary:")
    for name, path in [
        ("Diabetes",     os.path.join(DATA_DIR, "diabetes.csv")),
        ("Heart Disease",os.path.join(DATA_DIR, "heart.csv")),
        ("Anemia",       os.path.join(DATA_DIR, "anemia.csv")),
    ]:
        status = "✓ Ready" if os.path.exists(path) else "✗ Missing"
        print(f"  {name:20s}  {status}  ({path})")

    print("\nNext step:  python scripts/train_models.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
