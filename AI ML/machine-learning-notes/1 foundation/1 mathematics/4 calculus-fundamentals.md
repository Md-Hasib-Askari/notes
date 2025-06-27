# Calculus Fundamentals

## Limits and Continuity

### Limits
- Definition of limits
- One-sided limits
- Limits at infinity
- Indeterminate forms

### Continuity
- Continuous functions
- Discontinuities
- Intermediate Value Theorem

## Derivatives

### Basic Derivatives
- Definition using limits
- Power rule
- Product rule
- Quotient rule
- Chain rule

### Common Derivatives
- Polynomial functions
- Exponential functions
- Logarithmic functions
- Trigonometric functions

### Applications
- Rate of change
- Optimization problems
- Curve sketching
- Related rates

## Integration

### Indefinite Integrals
- Antiderivatives
- Basic integration rules
- Substitution method
- Integration by parts

### Definite Integrals
- Fundamental Theorem of Calculus
- Area under curves
- Average value of functions

## Applications in Machine Learning

### Optimization
- Finding minima and maxima
- Gradient descent algorithm
- Loss function minimization
- Learning rate selection

### Probability Distributions
- Continuous probability distributions
- Cumulative distribution functions
- Expected values and moments

### Neural Networks
- Backpropagation algorithm
- Chain rule applications
- Activation function derivatives

## Key Concepts for ML

### Gradient
- Partial derivatives
- Gradient vector
- Directional derivatives
- Gradient descent

### Optimization
- Critical points
- First and second derivative tests
- Convex functions
- Global vs local optima

## Practice Problems

### Derivatives
1. Find derivative of f(x) = x² + 3x + 2
2. Use chain rule for f(x) = sin(x²)
3. Optimize f(x) = x³ - 6x² + 9x + 1

### Applications
1. Minimize mean squared error
2. Find optimal learning rate
3. Analyze convergence conditions

## Code Examples

```python
import numpy as np
import matplotlib.pyplot as plt

# Numerical derivatives
def numerical_derivative(f, x, h=1e-7):
    return (f(x + h) - f(x - h)) / (2 * h)

# Gradient descent example
def gradient_descent(f, df, x0, learning_rate=0.01, iterations=1000):
    x = x0
    for i in range(iterations):
        x = x - learning_rate * df(x)
    return x
```

## Learning Objectives
- [ ] Master derivative rules
- [ ] Understand optimization principles
- [ ] Apply calculus to ML algorithms
- [ ] Implement gradient descent
- [ ] Analyze function behavior

## Next Steps
- Study multivariable calculus
- Learn partial derivatives
- Explore optimization theory
