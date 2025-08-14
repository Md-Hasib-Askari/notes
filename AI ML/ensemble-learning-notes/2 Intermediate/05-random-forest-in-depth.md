# 05 — Random Forest in Depth

Goal: Confidently tune Random Forests, use OOB validation, and extract trustworthy feature importance.

## Essentials
- Why RF works: bagging + feature subsampling reduce variance; trees capture non-linear interactions.
- Strengths: robust to scaling/outliers, handles mixed feature types, minimal preprocessing.
- Weaknesses: may underperform on very sparse/high-dimensional linear problems; slower with very many trees.

## Key hyperparameters (what they do)
- n_estimators: number of trees; more → lower variance, higher compute. 300–1000 typical.
- max_depth: cap tree depth; smaller depth reduces overfitting. None = grow to purity.
- min_samples_split / min_samples_leaf: larger values regularize; start with leaf=1–5.
- max_features: features to consider at each split. Common: 'sqrt' (cls), 'log2', or a float fraction.
- class_weight: 'balanced' for class imbalance.
- bootstrap: usually True (bagging).
- oob_score: if True, computes OOB accuracy (or R^2 in regression).

## Practical defaults
- Classification: n_estimators=400, max_features='sqrt', min_samples_leaf=1–3, random_state=42.
- Regression: try max_features=1.0 (all features) or 0.7; monitor OOB R^2.

## OOB validation
- Out-of-bag samples act like a built-in validation set for bagging. Reliable when n_estimators is reasonably large.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(
    n_estimators=600,
    max_features='sqrt',
    oob_score=True,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_tr, y_tr)
print('OOB score:', getattr(rf, 'oob_score_', None))
print('Test accuracy:', rf.score(X_te, y_te))
```

## Feature importance
- Impurity-based (Gini) importance: `rf.feature_importances_` — fast, but biased toward high-cardinality/continuous features.
- Permutation importance: more reliable, model-agnostic; measures drop in score when shuffling a feature.

```python
from sklearn.inspection import permutation_importance
import numpy as np

imp = permutation_importance(rf, X_te, y_te, n_repeats=20, random_state=42, n_jobs=-1)
sorted_idx = np.argsort(imp.importances_mean)[::-1]
for i in sorted_idx[:10]:
    print(i, imp.importances_mean[i])
```

## Tuning playbook
1) Fix n_estimators large enough (e.g., 600) to stabilize.
2) Tune max_features: try ['sqrt', 'log2', 0.3, 0.5, 0.7].
3) Regularize tree size: tune max_depth, min_samples_leaf.
4) Use class_weight='balanced' if classes are skewed.
5) Compare OOB vs CV; they should broadly agree.

## Gotchas
- Correlated features split importance; permutation importance helps.
- Very wide/sparse data may favor linear models or boosted trees.
- For probability calibration, consider CalibratedClassifierCV.
