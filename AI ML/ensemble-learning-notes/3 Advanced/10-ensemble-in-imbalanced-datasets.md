# 10 — Ensemble in Imbalanced Datasets

Goal: Build ensembles that stay robust on rare classes using reweighting, resampling, and specialized algorithms.

## Metrics & evaluation
- Prefer AUC-PR, F1 (macro), balanced accuracy, recall at fixed precision; avoid accuracy when skewed.
- Use stratified CV; keep test set untouched. Calibrate probabilities if thresholds matter.

## Techniques
- Class weighting: `class_weight='balanced'` (RF/LogReg/SVM) or library-specific `scale_pos_weight`.
- Resampling:
  - Over-sampling: SMOTE/ADASYN; under-sampling: RandomUnderSampler.
  - Do resampling inside CV folds only (avoid leakage).
- Specialized ensembles: Balanced Random Forest, EasyEnsemble, BalancedBagging.

## Balanced Random Forest
```python
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X, y = make_classification(n_samples=8000, n_features=20, weights=[0.95, 0.05], random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

brf = BalancedRandomForestClassifier(n_estimators=600, max_features='sqrt', random_state=42, n_jobs=-1)
brf.fit(X_tr, y_tr)
print(classification_report(y_te, brf.predict(X_te)))
```

## EasyEnsemble & BalancedBagging
```python
from imblearn.ensemble import EasyEnsembleClassifier, BalancedBaggingClassifier
from sklearn.tree import DecisionTreeClassifier

easy = EasyEnsembleClassifier(n_estimators=10, random_state=42)
base = DecisionTreeClassifier(max_depth=None, random_state=42)
bbag = BalancedBaggingClassifier(base_estimator=base, n_estimators=50, sampling_strategy='auto', random_state=42, n_jobs=-1)
```

## SMOTE + Ensemble workflow (leakage-safe)
```python
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

pipe = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=500, class_weight=None, random_state=42, n_jobs=-1))
])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X, y, scoring='average_precision', cv=cv, n_jobs=-1)
print('CV AUC-PR:', scores.mean())
```

## Threshold tuning
- Optimize threshold on validation to maximize F1/recall@precision. Consider cost-sensitive metrics.

## Tips
- Start with class_weight and thresholding; move to BRF/EasyEnsemble if needed.
- Monitor calibration (Brier score); isotonic/sigmoid calibration can help.
