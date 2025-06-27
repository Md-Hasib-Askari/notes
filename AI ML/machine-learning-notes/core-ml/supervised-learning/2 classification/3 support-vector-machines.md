# Support Vector Machines

## Key Concepts
**Support Vector Machines (SVM)** find the optimal hyperplane that maximizes the margin between classes, using support vectors (closest points to the decision boundary).

### Core Ideas
- **Maximum Margin**: Find the widest possible separation between classes
- **Support Vectors**: Data points closest to the decision boundary
- **Kernel Trick**: Transform data to higher dimensions for non-linear separation

## Types of SVM

### Linear SVM
For linearly separable data:
```python
from sklearn.svm import SVC
import numpy as np

# Linear SVM
svm_linear = SVC(kernel='linear', C=1.0)
svm_linear.fit(X_train, y_train)
predictions = svm_linear.predict(X_test)
```

### Non-linear SVM with Kernels
```python
# RBF (Radial Basis Function) kernel
svm_rbf = SVC(kernel='rbf', gamma='scale', C=1.0)

# Polynomial kernel
svm_poly = SVC(kernel='poly', degree=3, C=1.0)

# Custom kernel
def custom_kernel(X, Y):
    return np.dot(X, Y.T) ** 2

svm_custom = SVC(kernel=custom_kernel)
```

### Support Vector Regression (SVR)
```python
from sklearn.svm import SVR

# For regression problems
svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr.fit(X_train, y_train)
```

## Common Kernels

| Kernel | Formula | Use Case |
|--------|---------|----------|
| Linear | K(x,y) = x·y | Linearly separable data |
| RBF | K(x,y) = exp(-γ‖x-y‖²) | General purpose, non-linear |
| Polynomial | K(x,y) = (γx·y + r)^d | Feature interactions |
| Sigmoid | K(x,y) = tanh(γx·y + r) | Neural network-like |

## Hyperparameters

### C Parameter
- **High C**: Low bias, high variance (complex model)
- **Low C**: High bias, low variance (simple model)

### Gamma (for RBF kernel)
- **High gamma**: Close decision boundary (overfitting risk)
- **Low gamma**: Smooth decision boundary (underfitting risk)

```python
# Grid search for optimal parameters
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
}

grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
```

## Complete Example
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load data
iris = datasets.load_iris()
X = iris.data[:, :2]  # First two features
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train SVM
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model.fit(X_train, y_train)

# Predictions
y_pred = svm_model.predict(X_test)

# Results
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Support vectors: {svm_model.n_support_}")
print(classification_report(y_test, y_pred))
```

## Advantages & Disadvantages

### Advantages ✅
- Effective in high-dimensional spaces
- Memory efficient (uses support vectors only)
- Versatile (different kernels)
- Works well with small datasets

### Disadvantages ❌
- Slow on large datasets
- Sensitive to feature scaling
- No probabilistic output (unless probability=True)
- Choice of kernel and parameters crucial

## Best Practices
1. **Scale features** before training
2. **Start with RBF kernel** for most problems
3. **Use cross-validation** for parameter tuning
4. **Consider class imbalance** (use class_weight='balanced')

```python
# Feature scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Balanced classes
svm_balanced = SVC(kernel='rbf', class_weight='balanced')
```

## Learning Objectives
- [ ] Understand maximum margin principle
- [ ] Apply different kernel functions
- [ ] Tune hyperparameters effectively
- [ ] Handle non-linear classification problems
- [ ] Scale features appropriately