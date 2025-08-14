# 07 — Advanced Stacking / Blending

Goal: Build leakage-safe stacks with nested CV and understand when simple blending is enough.

## Stacking vs Blending
- Stacking: meta-model trained on out-of-fold predictions from base models. Safer, uses CV to avoid leakage.
- Blending: hold out a validation split; train base models on train, get predictions on holdout, train meta on those. Simpler but uses less data for training.

## Nested cross-validation (to evaluate stacking fairly)
- Outer CV: measures generalization of the entire stacked pipeline.
- Inner CV: generates OOF predictions for meta-model training and tunes hyperparameters.

### Sketch (conceptual)
```
for outer_fold in OuterCV(K):
    X_tr_o, X_te_o = split_outer(outer_fold)
    # Inner loop for stacking features (OOF)
    OOF_preds = zeros_like(X_tr_o.shape[0], n_base * n_classes)
    for base in base_models:
        for inner_fold in InnerCV(k):
            fit(base, X_tr_inner, y_tr_inner)
            OOF_preds[val_idx] = predict_proba(base, X_val_inner)
        fit(base, X_tr_o, y_tr_o)  # refit on full outer-train
    meta.fit(OOF_preds, y_tr_o)
    preds = meta.predict_proba(stack_predict(base_models, X_te_o))
    score = metric(y_te_o, preds)
```

## Practical guidance
- Use probabilities for classification; standardize meta-features if models differ in scale.
- Keep meta simple (logistic/linear/ridge). Regularize (L2) to prevent overfitting.
- Limit base models to 2–4 diverse models; more isn’t always better.
- For small data, prefer stacking (CV); for large data, blending can be a good, faster baseline.

## Example — Blending with a holdout set
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
import numpy as np

X, y = make_classification(n_samples=5000, n_features=40, n_informative=10, random_state=42)
X_tr, X_hold, y_tr, y_hold = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=500, random_state=42)
svm = SVC(probability=True, kernel='rbf', C=1.0, random_state=42)
rf.fit(X_tr, y_tr)
svm.fit(X_tr, y_tr)

P_rf = rf.predict_proba(X_hold)[:, 1]
P_svm = svm.predict_proba(X_hold)[:, 1]
X_meta = np.vstack([P_rf, P_svm]).T

meta = LogisticRegression(max_iter=2000, random_state=42)
meta.fit(X_meta, y_hold)
print('Blending AUC (holdout):', roc_auc_score(y_hold, meta.predict_proba(X_meta)[:, 1]))
```

## Exercise
- Implement a stacking pipeline with RF + SVM (+ optional GBM) using 5-fold OOF predictions for meta training; compare to blending.
