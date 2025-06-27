# Cross-Validation

## Overview
Cross-validation is a statistical method used to estimate the performance of machine learning models on unseen data by partitioning the original dataset into training and validation subsets.

## Types of Cross-Validation

### K-Fold Cross-Validation
- Divide data into k equal-sized folds
- Train on k-1 folds, validate on 1 fold
- Repeat k times with different validation fold
- Average the results

### Stratified K-Fold
- Maintains class distribution in each fold
- Important for imbalanced datasets
- Ensures representative samples

### Leave-One-Out (LOO)
- Special case where k = n (number of samples)
- Each sample is validation set once
- Computationally expensive but unbiased

### Time Series Cross-Validation
- Respects temporal order
- Walk-forward validation
- Expanding or sliding window

## Implementation

### Scikit-learn Implementation
```python
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

# Model
model = LogisticRegression()

# K-Fold CV
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

print(f"CV Scores: {cv_scores}")
print(f"Mean CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
```

### Custom Implementation
```python
import numpy as np
from sklearn.metrics import accuracy_score

def custom_cross_validation(model, X, y, k=5, shuffle=True, random_state=None):
    """
    Custom k-fold cross-validation implementation.
    """
    if random_state:
        np.random.seed(random_state)
    
    n_samples = len(X)
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    fold_size = n_samples // k
    scores = []
    
    for i in range(k):
        # Define validation indices
        start_val = i * fold_size
        end_val = start_val + fold_size if i < k-1 else n_samples
        val_indices = indices[start_val:end_val]
        
        # Define training indices
        train_indices = np.concatenate([indices[:start_val], indices[end_val:]])
        
        # Split data
        X_train, X_val = X[train_indices], X[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]
        
        # Train and evaluate
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        score = accuracy_score(y_val, y_pred)
        scores.append(score)
        
        print(f"Fold {i+1}: {score:.3f}")
    
    return np.array(scores)
```

## Best Practices

### Choosing K
- Common values: 5, 10
- Bias-variance tradeoff
- Computational considerations
- Dataset size constraints

### Stratification
- Use for classification problems
- Ensures balanced representation
- Particularly important for imbalanced data

### Nested Cross-Validation
- Outer loop: Model evaluation
- Inner loop: Hyperparameter tuning
- Prevents data leakage

## Learning Objectives
- [ ] Understand CV principles
- [ ] Implement different CV types
- [ ] Choose appropriate CV strategy
- [ ] Avoid common pitfalls

## Applications
- Model selection
- Hyperparameter tuning
- Performance estimation
- Feature selection validation
