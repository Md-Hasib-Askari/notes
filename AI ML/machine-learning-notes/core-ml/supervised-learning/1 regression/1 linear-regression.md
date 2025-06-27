# Linear Regression

## Overview

Linear regression is one of the fundamental algorithms in machine learning, used to model the relationship between a dependent variable and one or more independent variables by fitting a linear equation to the observed data.

## Mathematical Foundation

### Simple Linear Regression
For one feature:
```
y = β₀ + β₁x + ε
```
Where:
- y: dependent variable (target)
- x: independent variable (feature)
- β₀: intercept (bias)
- β₁: slope (weight)
- ε: error term

### Multiple Linear Regression
For multiple features:
```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
```
Matrix form:
```
y = Xβ + ε
```

## Key Assumptions

1. **Linearity**: Relationship between features and target is linear
2. **Independence**: Observations are independent
3. **Homoscedasticity**: Constant variance of residuals
4. **Normality**: Residuals are normally distributed
5. **No multicollinearity**: Features are not highly correlated

## Cost Function

### Mean Squared Error (MSE)
```
MSE = (1/n) Σ(yᵢ - ŷᵢ)²
```

### Sum of Squared Residuals (SSR)
```
SSR = Σ(yᵢ - ŷᵢ)²
```

## Solution Methods

### Normal Equation (Analytical Solution)
```
β = (XᵀX)⁻¹Xᵀy
```

Advantages:
- Exact solution
- No hyperparameters
- No iterations needed

Disadvantages:
- Requires matrix inversion O(n³)
- Numerical instability if XᵀX is singular
- Memory intensive for large datasets

### Gradient Descent (Iterative Solution)
```
β := β - α∇J(β)
```
Where ∇J(β) is the gradient of the cost function.

## Implementation

### From Scratch Implementation
```python
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def fit(self, X, y, method='gradient_descent'):
        """
        Fit the linear regression model.
        
        Parameters:
        X: feature matrix (n_samples, n_features)
        y: target vector (n_samples,)
        method: 'gradient_descent' or 'normal_equation'
        """
        n_samples, n_features = X.shape
        
        if method == 'normal_equation':
            self._fit_normal_equation(X, y)
        else:
            self._fit_gradient_descent(X, y)
    
    def _fit_normal_equation(self, X, y):
        """Fit using normal equation."""
        # Add bias column
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        
        # Calculate weights using normal equation
        try:
            theta = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
            self.bias = theta[0]
            self.weights = theta[1:]
        except np.linalg.LinAlgError:
            # Use pseudoinverse if matrix is singular
            theta = np.linalg.pinv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
            self.bias = theta[0]
            self.weights = theta[1:]
    
    def _fit_gradient_descent(self, X, y):
        """Fit using gradient descent."""
        # Initialize parameters
        self.weights = np.random.normal(0, 0.01, X.shape[1])
        self.bias = 0
        
        prev_cost = float('inf')
        
        for i in range(self.max_iterations):
            # Forward pass
            y_pred = self.predict(X)
            
            # Calculate cost
            cost = self._calculate_cost(y, y_pred)
            self.cost_history.append(cost)
            
            # Calculate gradients
            dw, db = self._calculate_gradients(X, y, y_pred)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Check for convergence
            if abs(prev_cost - cost) < self.tolerance:
                print(f"Converged after {i+1} iterations")
                break
            
            prev_cost = cost
    
    def _calculate_cost(self, y_true, y_pred):
        """Calculate mean squared error."""
        return np.mean((y_true - y_pred) ** 2)
    
    def _calculate_gradients(self, X, y_true, y_pred):
        """Calculate gradients for weights and bias."""
        n_samples = X.shape[0]
        
        # Gradient of weights
        dw = -(2/n_samples) * X.T @ (y_true - y_pred)
        
        # Gradient of bias
        db = -(2/n_samples) * np.sum(y_true - y_pred)
        
        return dw, db
    
    def predict(self, X):
        """Make predictions."""
        return X @ self.weights + self.bias
    
    def score(self, X, y):
        """Calculate R-squared score."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def plot_cost_history(self):
        """Plot cost function over iterations."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history)
        plt.title('Cost Function Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Cost (MSE)')
        plt.grid(True)
        plt.show()
```

### Scikit-learn Implementation
```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Generate sample data
def generate_data(n_samples=100, noise=0.1):
    np.random.seed(42)
    X = np.random.randn(n_samples, 2)
    true_weights = np.array([3, -2])
    y = X @ true_weights + 5 + noise * np.random.randn(n_samples)
    return X, y

# Create and train model
X, y = generate_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scikit-learn model
sklearn_model = LinearRegression()
sklearn_model.fit(X_train, y_train)

# Custom model
custom_model = LinearRegression()
custom_model.fit(X_train, y_train, method='gradient_descent')

# Make predictions
sklearn_pred = sklearn_model.predict(X_test)
custom_pred = custom_model.predict(X_test)

# Evaluate models
sklearn_r2 = r2_score(y_test, sklearn_pred)
custom_r2 = custom_model.score(X_test, y_test)

print(f"Scikit-learn R²: {sklearn_r2:.4f}")
print(f"Custom implementation R²: {custom_r2:.4f}")
```

## Model Evaluation

### Metrics
```python
def evaluate_regression(y_true, y_pred):
    """Comprehensive regression evaluation."""
    
    # Mean Squared Error
    mse = np.mean((y_true - y_pred) ** 2)
    
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(y_true - y_pred))
    
    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Adjusted R-squared (requires number of features)
    n = len(y_true)
    p = 1  # number of features (example)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'Adjusted R²': adj_r2
    }
```

### Residual Analysis
```python
def analyze_residuals(y_true, y_pred, X=None):
    """Analyze residuals for model diagnostics."""
    residuals = y_true - y_pred
    
    # Plot residuals
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Residuals vs fitted values
    axes[0, 0].scatter(y_pred, residuals, alpha=0.6)
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_xlabel('Fitted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Fitted Values')
    
    # Q-Q plot for normality
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot (Normality Check)')
    
    # Histogram of residuals
    axes[1, 0].hist(residuals, bins=20, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Residuals')
    
    # Residuals vs observation order
    axes[1, 1].plot(residuals, marker='o', linestyle='-', alpha=0.6)
    axes[1, 1].axhline(y=0, color='r', linestyle='--')
    axes[1, 1].set_xlabel('Observation Order')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].set_title('Residuals vs Order')
    
    plt.tight_layout()
    plt.show()
    
    # Statistical tests
    # Shapiro-Wilk test for normality
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    
    # Durbin-Watson test for autocorrelation (approximation)
    dw_stat = np.sum(np.diff(residuals)**2) / np.sum(residuals**2)
    
    print(f"Shapiro-Wilk test p-value: {shapiro_p:.4f}")
    print(f"Durbin-Watson statistic: {dw_stat:.4f}")
    
    if shapiro_p < 0.05:
        print("⚠️  Residuals may not be normally distributed")
    if dw_stat < 1.5 or dw_stat > 2.5:
        print("⚠️  Potential autocorrelation in residuals")
```

## Regularization

### Ridge Regression (L2 Regularization)
```python
class RidgeRegression(LinearRegression):
    def __init__(self, alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
    
    def _fit_normal_equation(self, X, y):
        """Ridge regression using normal equation."""
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        n_features = X_with_bias.shape[1]
        
        # Ridge penalty matrix (don't penalize bias term)
        penalty_matrix = self.alpha * np.eye(n_features)
        penalty_matrix[0, 0] = 0  # Don't penalize bias
        
        # Ridge solution
        theta = np.linalg.inv(X_with_bias.T @ X_with_bias + penalty_matrix) @ X_with_bias.T @ y
        self.bias = theta[0]
        self.weights = theta[1:]
    
    def _calculate_cost(self, y_true, y_pred):
        """Calculate cost with L2 penalty."""
        mse = np.mean((y_true - y_pred) ** 2)
        l2_penalty = self.alpha * np.sum(self.weights ** 2)
        return mse + l2_penalty
```

### Lasso Regression (L1 Regularization)
```python
from sklearn.linear_model import Lasso

# Lasso regression (L1 regularization)
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)

# Check feature selection (some coefficients become zero)
print("Lasso coefficients:", lasso_model.coef_)
print("Non-zero features:", np.sum(lasso_model.coef_ != 0))
```

## Practical Applications

### Feature Engineering for Linear Regression
```python
def polynomial_features(X, degree=2):
    """Create polynomial features."""
    n_samples, n_features = X.shape
    features = [X]
    
    for d in range(2, degree + 1):
        features.append(X ** d)
    
    return np.column_stack(features)

def interaction_features(X):
    """Create interaction features."""
    n_samples, n_features = X.shape
    interactions = []
    
    for i in range(n_features):
        for j in range(i + 1, n_features):
            interactions.append((X[:, i] * X[:, j]).reshape(-1, 1))
    
    if interactions:
        return np.column_stack([X] + interactions)
    return X
```

### Real-world Example: House Price Prediction
```python
# Simulate house price data
def create_house_data(n_samples=1000):
    np.random.seed(42)
    
    # Features: size, bedrooms, age, location_score
    size = np.random.normal(2000, 500, n_samples)
    bedrooms = np.random.choice([2, 3, 4, 5], n_samples, p=[0.2, 0.4, 0.3, 0.1])
    age = np.random.randint(0, 50, n_samples)
    location_score = np.random.uniform(1, 10, n_samples)
    
    # Price calculation with some realistic relationships
    price = (
        150 * size +           # $150 per sq ft
        10000 * bedrooms +     # $10k per bedroom
        -500 * age +           # Depreciation
        5000 * location_score + # Location premium
        50000 +                # Base price
        np.random.normal(0, 20000, n_samples)  # Noise
    )
    
    X = np.column_stack([size, bedrooms, age, location_score])
    return X, price

# Train and evaluate
X_house, y_house = create_house_data()
feature_names = ['Size (sq ft)', 'Bedrooms', 'Age (years)', 'Location Score']

X_train, X_test, y_train, y_test = train_test_split(
    X_house, y_house, test_size=0.2, random_state=42
)

# Standardize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
metrics = evaluate_regression(y_test, y_pred)

print("House Price Prediction Results:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.2f}")

# Feature importance
coefficients = model.weights if hasattr(model, 'weights') else model.coef_
for name, coef in zip(feature_names, coefficients):
    print(f"{name}: {coef:.2f}")
```

## Common Issues and Solutions

### Multicollinearity
```python
def detect_multicollinearity(X, feature_names):
    """Detect multicollinearity using correlation matrix and VIF."""
    import pandas as pd
    
    # Correlation matrix
    corr_matrix = np.corrcoef(X.T)
    
    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            if abs(corr_matrix[i, j]) > 0.8:
                high_corr_pairs.append((feature_names[i], feature_names[j], corr_matrix[i, j]))
    
    # Variance Inflation Factor (VIF)
    vif_values = []
    for i in range(X.shape[1]):
        # Regress feature i on all other features
        X_others = np.delete(X, i, axis=1)
        r_squared = LinearRegression().fit(X_others, X[:, i]).score(X_others, X[:, i])
        vif = 1 / (1 - r_squared) if r_squared < 0.999 else float('inf')
        vif_values.append(vif)
    
    return {
        'high_correlations': high_corr_pairs,
        'vif_values': list(zip(feature_names, vif_values))
    }
```

### Outlier Detection
```python
def detect_outliers(X, y, method='iqr'):
    """Detect outliers in features and target."""
    outliers = {}
    
    if method == 'iqr':
        # Interquartile Range method
        Q1 = np.percentile(y, 25)
        Q3 = np.percentile(y, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers['target'] = np.where((y < lower_bound) | (y > upper_bound))[0]
    
    elif method == 'zscore':
        # Z-score method
        z_scores = np.abs((y - np.mean(y)) / np.std(y))
        outliers['target'] = np.where(z_scores > 3)[0]
    
    return outliers
```

## Learning Objectives
- [ ] Understand linear regression theory
- [ ] Implement from scratch using gradient descent
- [ ] Apply normal equation method
- [ ] Evaluate model performance
- [ ] Handle assumptions violations
- [ ] Apply regularization techniques

## Practice Exercises
1. Implement polynomial regression
2. Create a cross-validation framework
3. Build automatic feature selection
4. Handle categorical variables
5. Compare different regularization methods

## Resources
- "The Elements of Statistical Learning" - Hastie, Tibshirani, Friedman
- Scikit-learn documentation
- Andrew Ng's Machine Learning Course
