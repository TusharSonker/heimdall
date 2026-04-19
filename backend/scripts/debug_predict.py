import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import phe as paillier
from core.models import TRAINED_WEIGHTS

def sigmoid(x): return 1/(1+math.exp(-x))

w = TRAINED_WEIGHTS['diabetes']
print("=== DIABETES WEIGHTS ===")
print(f"weights: {w['weights']}")
print(f"bias:    {w['bias']}")
print(f"mins:    {w['feature_mins']}")
print(f"maxs:    {w['feature_maxs']}")

# Test case: healthy person, expect LOW
raw    = [85, 22, 28, 70]   # glucose, bmi, age, bp
mins   = w['feature_mins']
maxs   = w['feature_maxs']
norm   = [(raw[i]-mins[i])/(maxs[i]-mins[i]) for i in range(4)]
score  = sum(w['weights'][i]*norm[i] for i in range(4)) + w['bias']
prob   = sigmoid(score)

print(f"\n--- Healthy person (expect LOW) ---")
print(f"raw:        {raw}")
print(f"normalized: {[round(n,4) for n in norm]}")
print(f"score:      {score:.4f}")
print(f"prob:       {prob*100:.1f}%")
print(f"risk:       {'HIGH' if prob>0.5 else 'LOW'}")

# Test case: diabetic person, expect HIGH
raw2   = [180, 36, 55, 95]
norm2  = [(raw2[i]-mins[i])/(maxs[i]-mins[i]) for i in range(4)]
score2 = sum(w['weights'][i]*norm2[i] for i in range(4)) + w['bias']
prob2  = sigmoid(score2)

print(f"\n--- Diabetic person (expect HIGH) ---")
print(f"raw:        {raw2}")
print(f"normalized: {[round(n,4) for n in norm2]}")
print(f"score:      {score2:.4f}")
print(f"prob:       {prob2*100:.1f}%")
print(f"risk:       {'HIGH' if prob2>0.5 else 'LOW'}")