# Logistic Regression

## Overview
Logistic regression is a statistical method for binary classification that uses the logistic (sigmoid) function to model the probability of class membership. Despite its name, it's a classification algorithm that outputs probabilities between 0 and 1.

## Mathematical Foundation

### Sigmoid Function
The core of logistic regression is the sigmoid function:
```
σ(z) = 1 / (1 + e^(-z))
where z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```

**Properties:**
- Output range: (0, 1)
- S-shaped curve
- Smooth and differentiable
- Maps any real number to probability

### Log-Odds and Logits
```python
# Log-odds (logit) transformation
logit(p) = ln(p / (1-p)) = z

# Probability from logit
p = e^z / (1 + e^z) = σ(z)
```

### Cost Function (Log-Likelihood)
```
Cost(θ) = -1/m * Σ[y*log(h_θ(x)) + (1-y)*log(1-h_θ(x))]
```

## Implementation from Scratch

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None
        self.costs = []
    
    def sigmoid(self, z):
        """Sigmoid activation function"""
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        """Train the logistic regression model"""
        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.max_iterations):
            # Forward pass
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_pred)
            
            # Compute cost
            cost = self.compute_cost(y, predictions)
            self.costs.append(cost)
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def compute_cost(self, y_true, y_pred):
        """Compute logistic regression cost"""
        # Prevent log(0)
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        cost = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return cost
    
    def predict(self, X):
        """Make predictions"""
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_pred)
        return [1 if i > 0.5 else 0 for i in y_pred]
    
    def predict_proba(self, X):
        """Predict probabilities"""
        linear_pred = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_pred)

# Example usage
if __name__ == "__main__":
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                             n_informative=2, random_state=42, n_clusters_per_class=1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LogisticRegression(learning_rate=0.1, max_iterations=1000)
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    # Calculate accuracy
    accuracy = np.mean(predictions == y_test)
    print(f"Accuracy: {accuracy:.3f}")
    
    # Plot cost function
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(model.costs)
    plt.title('Cost Function During Training')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    
    # Plot decision boundary
    plt.subplot(1, 2, 2)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', alpha=0.7)
    plt.title('Test Data with True Labels')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    plt.tight_layout()
    plt.show()
```

## Scikit-learn Implementation

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Standardize features (important for logistic regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train model
sklearn_model = LogisticRegression(random_state=42, max_iter=1000)
sklearn_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = sklearn_model.predict(X_test_scaled)
y_pred_proba = sklearn_model.predict_proba(X_test_scaled)[:, 1]

# Evaluate model
print("Classification Report:")
print(classification_report(y_test, y_pred))

print(f"\nROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.3f}")

print(f"\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance (coefficients)
feature_importance = sklearn_model.coef_[0]
print(f"\nFeature Coefficients: {feature_importance}")
```

## Regularization

### L1 Regularization (Lasso)
```python
# L1 regularization encourages sparsity
l1_model = LogisticRegression(penalty='l1', C=0.1, solver='liblinear', random_state=42)
l1_model.fit(X_train_scaled, y_train)

print(f"L1 Coefficients: {l1_model.coef_[0]}")
```

### L2 Regularization (Ridge)
```python
# L2 regularization prevents overfitting
l2_model = LogisticRegression(penalty='l2', C=0.1, random_state=42)
l2_model.fit(X_train_scaled, y_train)

print(f"L2 Coefficients: {l2_model.coef_[0]}")
```

### Elastic Net
```python
# Combination of L1 and L2
elastic_model = LogisticRegression(penalty='elasticnet', C=0.1, l1_ratio=0.5, 
                                 solver='saga', random_state=42, max_iter=1000)
elastic_model.fit(X_train_scaled, y_train)
```

## Multiclass Classification

```python
from sklearn.datasets import load_iris

# Load multiclass dataset
iris = load_iris()
X_iris, y_iris = iris.data, iris.target

# Split data
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(
    X_iris, y_iris, test_size=0.3, random_state=42)

# Multiclass logistic regression
multiclass_model = LogisticRegression(multi_class='ovr', random_state=42)
multiclass_model.fit(X_train_iris, y_train_iris)

# Predictions
y_pred_iris = multiclass_model.predict(X_test_iris)
y_pred_proba_iris = multiclass_model.predict_proba(X_test_iris)

print("Multiclass Classification Report:")
print(classification_report(y_test_iris, y_pred_iris, target_names=iris.target_names))
```

## Model Interpretation

```python
import pandas as pd

def interpret_logistic_regression(model, feature_names):
    """Interpret logistic regression coefficients"""
    coefficients = model.coef_[0]
    odds_ratios = np.exp(coefficients)
    
    interpretation_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients,
        'Odds_Ratio': odds_ratios,
        'Impact': ['Positive' if coef > 0 else 'Negative' for coef in coefficients]
    })
    
    interpretation_df = interpretation_df.sort_values('Coefficient', key=abs, ascending=False)
    return interpretation_df

# Example interpretation
feature_names = ['Feature_1', 'Feature_2']
interpretation = interpret_logistic_regression(sklearn_model, feature_names)
print("Feature Interpretation:")
print(interpretation)
```

## Advanced Topics

### Handling Imbalanced Data
```python
from sklearn.utils.class_weight import compute_class_weight

# Calculate class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

# Train with balanced class weights
balanced_model = LogisticRegression(class_weight='balanced', random_state=42)
balanced_model.fit(X_train_scaled, y_train)
```

### Cross-Validation and Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV, cross_val_score

# Define parameter grid
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

# Grid search with cross-validation
grid_search = GridSearchCV(LogisticRegression(random_state=42), 
                          param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.3f}")

# Cross-validation scores
cv_scores = cross_val_score(grid_search.best_estimator_, X_train_scaled, y_train, cv=5)
print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
```

## When to Use Logistic Regression

**Advantages:**
- Simple and fast
- No assumptions about feature distributions
- Less prone to overfitting with low-dimensional data
- Provides probabilistic output
- Good baseline model

**Disadvantages:**
- Assumes linear relationship between features and log-odds
- Sensitive to outliers
- Requires large sample sizes for stable results

**Best Use Cases:**
- Binary classification problems
- When you need probability estimates
- Linear relationships between features and target
- As a baseline model
- When interpretability is important

## Learning Objectives
- [x] Understand the sigmoid function and its properties
- [x] Implement logistic regression from scratch
- [x] Use scikit-learn for logistic regression
- [x] Apply regularization techniques
- [x] Interpret model coefficients and odds ratios
- [x] Handle multiclass classification
- [x] Apply to real-world problems