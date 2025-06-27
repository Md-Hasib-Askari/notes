# Decision Trees

## Overview
Decision trees are intuitive supervised learning algorithms that make predictions by learning decision rules from data features. They create a tree-like model where internal nodes represent features, branches represent decision rules, and leaves represent outcomes.

## Tree Structure
```
Root Node (Best Feature)
├── Branch 1 (Feature ≤ threshold)
│   ├── Internal Node
│   └── Leaf (Prediction)
└── Branch 2 (Feature > threshold)
    └── Leaf (Prediction)
```

**Key Components:**
- **Root Node**: Starting point with best splitting feature
- **Internal Nodes**: Decision points based on feature values
- **Branches**: Paths representing decision outcomes
- **Leaf Nodes**: Final predictions (class or value)

## Splitting Criteria

### Information Gain (ID3, C4.5)
```python
import numpy as np

def entropy(y):
    """Calculate entropy of target variable"""
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities + 1e-10))

def information_gain(X, y, feature_idx, threshold):
    """Calculate information gain for a split"""
    parent_entropy = entropy(y)
    
    # Split data
    left_mask = X[:, feature_idx] <= threshold
    left_y, right_y = y[left_mask], y[~left_mask]
    
    # Weighted entropy after split
    n = len(y)
    weighted_entropy = (len(left_y)/n * entropy(left_y) + 
                       len(right_y)/n * entropy(right_y))
    
    return parent_entropy - weighted_entropy
```

### Gini Impurity (CART)
```python
def gini_impurity(y):
    """Calculate Gini impurity"""
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return 1 - np.sum(probabilities ** 2)

def gini_gain(X, y, feature_idx, threshold):
    """Calculate Gini gain for a split"""
    parent_gini = gini_impurity(y)
    
    left_mask = X[:, feature_idx] <= threshold
    left_y, right_y = y[left_mask], y[~left_mask]
    
    n = len(y)
    weighted_gini = (len(left_y)/n * gini_impurity(left_y) + 
                    len(right_y)/n * gini_impurity(right_y))
    
    return parent_gini - weighted_gini
```

## Implementation Example

### Simple Decision Tree Classifier
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load data
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train decision tree
dt = DecisionTreeClassifier(
    max_depth=3,           # Limit tree depth
    min_samples_split=2,   # Minimum samples to split
    min_samples_leaf=1,    # Minimum samples in leaf
    random_state=42
)

dt.fit(X_train, y_train)
predictions = dt.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
print(f"Feature Importance: {dt.feature_importances_}")
```

### Tree Visualization
```python
from sklearn.tree import plot_tree

plt.figure(figsize=(12, 8))
plot_tree(dt, 
          feature_names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
          class_names=['setosa', 'versicolor', 'virginica'],
          filled=True, 
          fontsize=10)
plt.title("Decision Tree for Iris Classification")
plt.show()
```

## Preventing Overfitting

### Pruning Techniques
```python
# Pre-pruning (during training)
dt_pruned = DecisionTreeClassifier(
    max_depth=5,           # Limit depth
    min_samples_split=10,  # Require more samples to split
    min_samples_leaf=5,    # Require more samples in leaves
    max_features=3,        # Limit features considered
    random_state=42
)

# Post-pruning using cost complexity
path = dt.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

# Train trees with different alpha values
train_scores, test_scores = [], []
for alpha in ccp_alphas:
    dt_alpha = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
    dt_alpha.fit(X_train, y_train)
    train_scores.append(dt_alpha.score(X_train, y_train))
    test_scores.append(dt_alpha.score(X_test, y_test))

# Plot pruning results
plt.figure(figsize=(10, 6))
plt.plot(ccp_alphas, train_scores, label='Training Accuracy')
plt.plot(ccp_alphas, test_scores, label='Test Accuracy')
plt.xlabel('Cost Complexity Parameter (alpha)')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Decision Tree Pruning')
plt.show()
```

## Advantages & Disadvantages

**Advantages:**
- Easy to understand and interpret
- No assumptions about data distribution
- Handles both numerical and categorical features
- Built-in feature selection
- Can capture non-linear relationships

**Disadvantages:**
- Prone to overfitting
- Unstable (small data changes = different trees)
- Biased toward features with many levels
- Poor performance on linear relationships

## Best Practices

1. **Control Tree Complexity:**
   ```python
   # Tune hyperparameters
   from sklearn.model_selection import GridSearchCV
   
   param_grid = {
       'max_depth': [3, 5, 7, 10],
       'min_samples_split': [2, 5, 10],
       'min_samples_leaf': [1, 2, 4]
   }
   
   grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
   grid_search.fit(X_train, y_train)
   ```

2. **Use Ensemble Methods:**
   - Random Forest
   - Gradient Boosting
   - Extra Trees

3. **Feature Engineering:**
   - Handle missing values
   - Scale features if needed
   - Create meaningful categorical bins

## Learning Objectives
- [x] Understand tree structure and splitting criteria
- [x] Implement decision trees from scratch
- [x] Apply pruning techniques to prevent overfitting
- [x] Visualize and interpret decision trees
- [x] Compare different splitting algorithms
- [x] Use decision trees for both classification and regression