# 02 — Basic Bagging

Bootstrap Aggregating (bagging) trains many models on different random samples and averages them to reduce variance.

## Intuition
- Decision trees are high-variance: small changes in data can change the tree a lot.
- Train many trees on different bootstrapped samples → average their predictions → smoother, more stable model.

## Algorithm (Random Forest perspective)
1) For each tree t in 1..n_estimators:
   - Sample with replacement a bootstrap of the training data (same size as original by default).
   - Grow a decision tree; at each split, consider a random subset of features (max_features).
2) Aggregate predictions over trees (majority vote for classification; mean for regression).

## Key parameters
- n_estimators: more trees → lower variance; diminishing returns after a point.
- max_features: controls feature subsampling (e.g., 'sqrt' for classification, 'log2', or a float fraction).
- max_depth, min_samples_split, min_samples_leaf: control tree complexity.
- max_samples (BaggingClassifier): size of bootstrap sample per estimator.
- oob_score: if True, use out-of-bag samples for validation.
- n_jobs: parallelism; set to -1 to use all cores.

## Example — RandomForest on Iris with OOB
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(
    n_estimators=400,
    max_features='sqrt',
    oob_score=True,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_tr, y_tr)
print('OOB score:', getattr(rf, 'oob_score_', None))
print('Test accuracy:', rf.score(X_te, y_te))
```

## Practical heuristics
- Start: n_estimators=300–500; max_features='sqrt' (classification) or 1.0 (regression if few features).
- Use min_samples_leaf=1–5 to control overfitting; increase if noisy.
- Tune class_weight='balanced' when classes are imbalanced.
- Feature importance: `rf.feature_importances_` is useful but can be biased towards high-cardinality features.

## Troubleshooting
- Underfitting: increase tree depth or lower min_samples_leaf.
- Overfitting: decrease depth or increase min_samples_leaf; try smaller max_features.
- Slow training: reduce n_estimators temporarily during tuning; set n_jobs=-1.

## Exercise
- Train RandomForestClassifier on Iris; grid search n_estimators and max_features; compare OOB vs CV.
