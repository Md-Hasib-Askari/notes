# 04 — Simple Stacking

Stacking combines predictions from multiple base models using a meta-model trained on out-of-fold (OOF) predictions, reducing overfitting and leveraging complementary strengths.

## Why stacking works
- Different algorithms capture different structures (trees capture interactions, SVMs capture margins, linear models capture global trends).
- A meta-learner can learn when to trust each base model.

## Correct workflow (to avoid leakage)
1) Split data into K folds.
2) For each base model:
   - For fold k: train on K-1 folds, predict on the held-out fold → collect OOF predictions.
   - After OOF is complete, fit the base model on the full training data for final use.
3) Train the meta-model on the OOF predictions (X_meta) and true labels (y).
4) At inference time, get base model predictions on test data and feed them to the meta-model.

## What to feed the meta-model
- Classification: use predicted probabilities (soft) rather than hard class labels.
- Regression: use raw predictions.

## Example — StackingClassifier (probabilities to meta-model)
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

base_estimators = [
    ('rf', RandomForestClassifier(n_estimators=300, random_state=42)),
    ('svc', SVC(kernel='rbf', probability=True, random_state=42))
]
meta = LogisticRegression(max_iter=2000, random_state=42)

stack = StackingClassifier(
    estimators=base_estimators,
    final_estimator=meta,
    stack_method='auto',    # uses predict_proba when available
    passthrough=False,      # set True to include original features
    cv=5                    # safer OOF-like splitting inside
)
stack.fit(X_tr, y_tr)
print('Stacked accuracy:', stack.score(X_te, y_te))
```

## Tips
- Diversity matters: mix trees, linear/logistic, kernel methods, kNN, etc.
- Keep the meta-model simple (logistic/linear) with regularization.
- Use cross-validated stacking (cv>1) and probabilities for classification.
- Passthrough=True can help when base predictions alone are insufficient, but watch for overfitting.

## Troubleshooting
- If stacking underperforms best base model:
  - Check that cv>1 and probabilities are used; ensure no leakage.
  - Simplify meta-model or reduce number of base models.
  - Ensure base models are individually well-tuned and diverse.

## Exercise
- Build a stack with RandomForest + SVM (+ optionally kNN) and a logistic meta-model; compare against each base model with 5-fold CV.
