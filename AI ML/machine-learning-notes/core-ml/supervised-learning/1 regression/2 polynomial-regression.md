# Polynomial Regression

## Overview
Polynomial regression extends linear regression to model non-linear relationships by adding polynomial terms (x², x³, etc.) as features. It's useful when the relationship between variables follows a curved pattern.

## Mathematical Foundation

### Polynomial Features
- **Linear**: y = β₀ + β₁x
- **Quadratic**: y = β₀ + β₁x + β₂x²
- **Cubic**: y = β₀ + β₁x + β₂x² + β₃x³
- **General**: y = β₀ + β₁x + β₂x² + ... + βₙxⁿ

### Key Concepts
- **Degree**: Highest power of the polynomial
- **Feature Engineering**: Creating polynomial features from original variables
- **Bias-Variance Tradeoff**: Higher degrees increase variance

## Implementation with Python

### Basic Polynomial Regression
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Generate sample data with non-linear relationship
np.random.seed(42)
X = np.linspace(0, 1, 100).reshape(-1, 1)
y = 1.5 * X.ravel() + 2 * X.ravel()**2 + 0.5 * np.random.randn(100)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create polynomial features
degree = 2
poly_features = PolynomialFeatures(degree=degree)
X_poly_train = poly_features.fit_transform(X_train)
X_poly_test = poly_features.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_poly_train, y_train)

# Make predictions
y_pred = model.predict(X_poly_test)

print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
```

### Using Pipeline for Cleaner Code
```python
# Create polynomial regression pipeline
poly_reg = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('linear', LinearRegression())
])

# Fit and predict
poly_reg.fit(X_train, y_train)
y_pred_pipeline = poly_reg.predict(X_test)

# Visualize results
X_plot = np.linspace(0, 1, 100).reshape(-1, 1)
y_plot = poly_reg.predict(X_plot)

plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, alpha=0.6, label='Training data')
plt.scatter(X_test, y_test, alpha=0.6, label='Test data')
plt.plot(X_plot, y_plot, 'r-', label=f'Polynomial (degree {degree})')
plt.legend()
plt.title('Polynomial Regression')
plt.show()
```

## Degree Selection and Validation

### Comparing Different Degrees
```python
from sklearn.model_selection import validation_curve

# Test different polynomial degrees
degrees = range(1, 10)
train_scores, val_scores = validation_curve(
    Pipeline([('poly', PolynomialFeatures()), ('linear', LinearRegression())]),
    X_train, y_train, param_name='poly__degree', param_range=degrees,
    cv=5, scoring='neg_mean_squared_error'
)

# Plot validation curve
plt.figure(figsize=(10, 6))
plt.plot(degrees, -train_scores.mean(axis=1), 'o-', label='Training Score')
plt.plot(degrees, -val_scores.mean(axis=1), 'o-', label='Validation Score')
plt.fill_between(degrees, -val_scores.mean(axis=1) - val_scores.std(axis=1),
                 -val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.3)
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error')
plt.title('Validation Curve for Polynomial Regression')
plt.legend()
plt.grid(True)
plt.show()
```

### Regularized Polynomial Regression
```python
from sklearn.linear_model import Ridge, Lasso

# Ridge regression with polynomial features
ridge_poly = Pipeline([
    ('poly', PolynomialFeatures(degree=5)),
    ('ridge', Ridge(alpha=0.1))
])

# Lasso regression with polynomial features
lasso_poly = Pipeline([
    ('poly', PolynomialFeatures(degree=5)),
    ('lasso', Lasso(alpha=0.01))
])

# Compare models
models = {
    'Linear': LinearRegression(),
    'Polynomial (deg=5)': Pipeline([('poly', PolynomialFeatures(5)), ('linear', LinearRegression())]),
    'Ridge Polynomial': ridge_poly,
    'Lasso Polynomial': lasso_poly
}

for name, model in models.items():
    if name == 'Linear':
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
    else:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
    
    print(f"{name}: R² = {r2_score(y_test, pred):.4f}, MSE = {mean_squared_error(y_test, pred):.4f}")
```

## Practical Example: Temperature vs Ice Cream Sales

```python
# Simulate realistic data
np.random.seed(42)
temperature = np.random.uniform(10, 35, 200)  # Temperature in Celsius
# Non-linear relationship: sales increase rapidly at higher temperatures
ice_cream_sales = (temperature - 10) * 2 + 0.5 * (temperature - 10)**2 + np.random.normal(0, 10, 200)

X_temp = temperature.reshape(-1, 1)
y_sales = ice_cream_sales

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_temp, y_sales, test_size=0.2, random_state=42)

# Compare linear vs polynomial
linear_model = LinearRegression()
poly_model = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('linear', LinearRegression())
])

# Fit models
linear_model.fit(X_train, y_train)
poly_model.fit(X_train, y_train)

# Evaluate
linear_pred = linear_model.predict(X_test)
poly_pred = poly_model.predict(X_test)

print("Linear Regression:")
print(f"R² Score: {r2_score(y_test, linear_pred):.4f}")
print(f"MSE: {mean_squared_error(y_test, linear_pred):.4f}")

print("\nPolynomial Regression:")
print(f"R² Score: {r2_score(y_test, poly_pred):.4f}")
print(f"MSE: {mean_squared_error(y_test, poly_pred):.4f}")
```

## Common Pitfalls and Best Practices

### Overfitting Prevention
```python
# Use cross-validation for degree selection
from sklearn.model_selection import GridSearchCV

param_grid = {'poly__degree': range(1, 8)}
grid_search = GridSearchCV(
    Pipeline([('poly', PolynomialFeatures()), ('linear', LinearRegression())]),
    param_grid, cv=5, scoring='neg_mean_squared_error'
)

grid_search.fit(X_train, y_train)
best_degree = grid_search.best_params_['poly__degree']
print(f"Best polynomial degree: {best_degree}")
```

### Feature Scaling
```python
from sklearn.preprocessing import StandardScaler

# Scale features before polynomial transformation
scaled_poly = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=3)),
    ('linear', LinearRegression())
])

scaled_poly.fit(X_train, y_train)
scaled_pred = scaled_poly.predict(X_test)
```

## When to Use Polynomial Regression

### Good Use Cases
- **Clear non-linear patterns** in scatter plots
- **Small to medium datasets** (polynomial features increase quickly)
- **Interpretable relationships** where domain knowledge suggests polynomial behavior
- **Baseline for comparison** with more complex models

### Limitations
- **Curse of dimensionality** with high degrees
- **Extrapolation issues** outside training range
- **Numerical instability** with very high degrees
- **Limited flexibility** compared to other non-linear methods

## Real-World Applications
1. **Physics**: Projectile motion, spring dynamics
2. **Economics**: Cost functions, revenue curves
3. **Biology**: Population growth models
4. **Engineering**: Stress-strain relationships

## Learning Objectives
- [x] Understand polynomial feature transformation
- [x] Implement polynomial regression with scikit-learn
- [x] Handle overfitting through regularization and validation
- [x] Select appropriate polynomial degree using cross-validation
- [x] Compare linear vs polynomial models
- [x] Apply feature scaling with polynomial features
- [x] Recognize when polynomial regression is appropriate

## Next Steps
- Explore spline regression for more flexible non-linear modeling
- Learn about kernel methods and Support Vector Regression
- Study ensemble methods for complex non-linear relationships