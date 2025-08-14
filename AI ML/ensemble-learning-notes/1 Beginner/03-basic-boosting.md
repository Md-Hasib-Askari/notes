# 03 — Basic Boosting

Boosting builds a strong model by adding many weak learners sequentially, each correcting the errors of the previous ensemble.

## Intuition
- Start with a simple model (e.g., decision stump). Identify where it errs. Add another model that focuses on those errors. Repeat.
- Think of it as gradient descent in function space (for Gradient Boosting): each stage fits the negative gradient (residuals) of the loss.

## AdaBoost (concept)
- Keeps per-sample weights. Misclassified samples get higher weights for the next learner.
- Final prediction is a weighted vote of weak learners.

## Gradient Boosting (concept)
- No sample weights; instead, each stage fits residuals/gradients of a differentiable loss (e.g., log loss).
- Learning rate shrinks each stage’s contribution to improve generalization.

## Key parameters
- n_estimators: number of boosting stages.
- learning_rate: contribution of each stage (smaller → need more stages, often better generalization).
- max_depth or max_leaf_nodes of weak learners: keep weak learners shallow (1–3 depth) to avoid overfitting.

## Example — AdaBoost (binary)
```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

X, y = make_classification(n_samples=1200, n_features=20, n_informative=6, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)

base = DecisionTreeClassifier(max_depth=1, random_state=42)
ada = AdaBoostClassifier(
    estimator=base,        # for older sklearn, use base_estimator=base
    n_estimators=300,
    learning_rate=0.2,
    random_state=42
)
ada.fit(X_tr, y_tr)
print('AdaBoost accuracy:', ada.score(X_te, y_te))
```

## Early signs of over/underfitting
- Underfitting: train and validation scores both low → increase n_estimators or weak learner depth slightly.
- Overfitting: train high, validation dropping → lower max_depth, reduce learning_rate, enable early stopping (for GBM/XGBoost/LightGBM), or increase regularization.

## Practical tips
- Lower learning_rate (0.05–0.2) with more estimators (200–1000) is a safe default.
- For class imbalance, try AdaBoost SAMME.R (default) and check AUC/PR, not only accuracy.
- Consider modern libraries (XGBoost/LightGBM/CatBoost) for speed and strong performance later.

## Exercise
- Train AdaBoost on a binary dataset; try learning_rate in [0.01, 0.05, 0.1, 0.2] and n_estimators in [50, 200, 500]; plot validation curves.
