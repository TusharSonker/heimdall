# Heimdall — Data Directory

Place downloaded CSV files here before running `train_models.py`.

## Required Files

| File           | Dataset                     | Source                                                                 | Size     |
|----------------|-----------------------------|------------------------------------------------------------------------|----------|
| `diabetes.csv` | Pima Indians Diabetes       | Auto-downloaded by `download_data.py`                                 | ~24 KB   |
| `heart.csv`    | Heart Disease (Cleveland)   | Auto-downloaded by `download_data.py`                                 | ~7 KB    |
| `anemia.csv`   | Anemia Dataset              | Manual: https://www.kaggle.com/datasets/biswaranjanrao/anemia-dataset | ~60 KB   |

## How to Get Them

### Option A — Automatic (Diabetes + Heart)
```bash
cd heimdall/backend
source venv/bin/activate
python scripts/download_data.py
```

### Option B — Manual (all three)
Download each file and place it in this directory:

**Diabetes:**
https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv
→ Save as `diabetes.csv`

**Heart Disease:**
https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data
→ Save as `heart.csv`

**Anemia:**
https://www.kaggle.com/datasets/biswaranjanrao/anemia-dataset
→ Download ZIP, extract, rename CSV to `anemia.csv`

## Expected CSV Formats

### diabetes.csv
No header row. 9 columns:
```
Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
6,148,72,35,0,33.6,0.627,50,1
...
```

### heart.csv
No header row. 14 columns (? for missing values):
```
63,1,1,145,233,1,2,150,0,2.3,3,0,6,0
...
```

### anemia.csv
Has header row. Must include these columns (names may vary slightly):
```
Hemoglobin,RBC,MCV,MCH,...,Result
13.4,4.5,85,28,...,0
...
```

## After Placing Files

```bash
python scripts/train_models.py
```

This produces `backend/core/trained_weights.json` which the backend
loads automatically on startup.
