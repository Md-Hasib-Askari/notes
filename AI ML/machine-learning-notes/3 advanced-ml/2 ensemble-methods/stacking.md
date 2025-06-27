# Stacking

## Overview
Stacking is an ensemble learning technique that combines multiple base models (weak learners) and a meta-model (strong learner) to improve predictive performance.

## Core Concepts

### What is Stacking?
- **Base Models**: Diverse models trained on the same dataset
- **Meta-Model**: Learns to combine predictions from base models
- **Training Process**:
  1. Train base models on the training data
  2. Use predictions from base models as features for the meta-model
  3. Train the meta-model on these features

### Advantages
- Combines strengths of multiple models
- Reduces bias and variance
- Works well with diverse base models

### Disadvantages
- Computationally expensive
- Risk of overfitting if meta-model is too complex

## Implementation

### Basic Stacking
```python
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Example: Classification
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Base models
base_models = [
    ('decision_tree', DecisionTreeClassifier(max_depth=5)),
    ('svm', SVC(probability=True))
]

# Meta-model
meta_model = LogisticRegression()

# Stacking classifier
stacking_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5
)

stacking_clf.fit(X_train, y_train)
print("Accuracy:", stacking_clf.score(X_test, y_test))

# Example: Regression
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

base_models = [
    ('decision_tree', DecisionTreeClassifier(max_depth=5)),
    ('svm', SVC(probability=True))
]

meta_model = LinearRegression()

stacking_reg = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5
)

stacking_reg.fit(X_train, y_train)
print("R² Score:", stacking_reg.score(X_test, y_test))
```

### Custom Stacking
```python
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

# Custom stacking implementation
class CustomStacking:
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    def fit(self, X, y):
        self.base_models_ = [list() for _ in self.base_models]
        self.meta_features_ = np.zeros((X.shape[0], len(self.base_models)))

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        for i, model in enumerate(self.base_models):
            for train_idx, val_idx in kf.split(X):
                clone_model = clone(model)
                clone_model.fit(X[train_idx], y[train_idx])
                self.base_models_[i].append(clone_model)
                self.meta_features_[val_idx, i] = clone_model.predict(X[val_idx])

        self.meta_model.fit(self.meta_features_, y)

    def predict(self, X):
        meta_features = np.zeros((X.shape[0], len(self.base_models)))

        for i, models in enumerate(self.base_models_):
            predictions = np.array([model.predict(X) for model in models])
            meta_features[:, i] = predictions.mean(axis=0)

        return self.meta_model.predict(meta_features)

# Example usage
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

base_models = [DecisionTreeClassifier(max_depth=5), LogisticRegression()]
meta_model = LogisticRegression()

stacking = CustomStacking(base_models, meta_model)
stacking.fit(X, y)
print("Accuracy:", accuracy_score(y, stacking.predict(X)))
```

## Advanced Techniques

### Blending
Blending is a simpler alternative to stacking where base model predictions are combined using a weighted average or another aggregation method.
```python
# Weighted average blending
base_model_preds = [model.predict(X_test) for model in base_models]
weights = [0.5, 0.5]  # Example weights
blended_preds = np.average(base_model_preds, axis=0, weights=weights)
```

### Multi-layer Stacking
Multi-layer stacking involves stacking multiple levels of base models and meta-models.
```python
# Example: Two-layer stacking
layer1_models = [
    ('decision_tree', DecisionTreeClassifier(max_depth=5)),
    ('svm', SVC(probability=True))
]

layer2_model = LogisticRegression()

stacking_clf = StackingClassifier(
    estimators=layer1_models,
    final_estimator=layer2_model,
    cv=5
)

stacking_clf.fit(X_train, y_train)
```

## Best Practices
1. Use diverse base models to maximize ensemble benefits.
2. Avoid overfitting by using cross-validation for meta-model training.
3. Experiment with different meta-models (e.g., linear models, neural networks).
4. Use feature selection to reduce dimensionality of meta-features.
5. Monitor computational cost, especially for large datasets.

## Common Pitfalls
- **Overfitting**: Use regularization and cross-validation.
- **Computational Cost**: Limit the number of base models and folds.
- **Model Compatibility**: Ensure base models and meta-models are compatible.

## Applications
- **Classification**: Fraud detection, sentiment analysis
- **Regression**: House price prediction, stock market forecasting
- **Multi-output Tasks**: Multi-label classification, multi-target regression

## Key Takeaways
- Stacking combines predictions from multiple models to improve accuracy.
- Meta-models play a crucial role in learning how to combine base model outputs.
- Blending and multi-layer stacking are advanced techniques for ensemble learning.

## Resources
- **Scikit-learn Documentation**: Official guide to stacking
- **Kaggle Competitions**: Many winning solutions use stacking
- **Papers**: Research papers on ensemble methods
