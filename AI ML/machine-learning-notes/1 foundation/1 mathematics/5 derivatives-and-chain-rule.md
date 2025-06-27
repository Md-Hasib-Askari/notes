# Derivatives and Chain Rule

## Multivariable Calculus

### Partial Derivatives
- Definition and notation
- Geometric interpretation
- Higher-order partial derivatives
- Mixed partial derivatives

### Gradient Vector
- Definition: ∇f = (∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ)
- Geometric meaning: direction of steepest ascent
- Magnitude: rate of steepest ascent

### Directional Derivatives
- Rate of change in any direction
- Relationship with gradient
- Maximum rate of change

## Chain Rule

### Single Variable Chain Rule
If y = f(g(x)), then:
dy/dx = f'(g(x)) · g'(x)

### Multivariable Chain Rule
For composite functions of multiple variables:
- Tree diagrams
- Partial derivative chains
- Total derivatives

### Applications in Neural Networks
- Backpropagation algorithm
- Error gradient computation
- Weight update calculations

## Optimization Theory

### Critical Points
- First-order conditions: ∇f = 0
- Stationary points
- Local vs global extrema

### Second-Order Conditions
- Hessian matrix
- Definiteness tests
- Saddle points

### Constrained Optimization
- Lagrange multipliers
- KKT conditions
- Penalty methods

## Machine Learning Applications

### Loss Function Optimization
- Mean Squared Error derivatives
- Cross-entropy derivatives
- Regularization terms

### Gradient Descent Variants
- Batch gradient descent
- Stochastic gradient descent
- Mini-batch gradient descent

### Backpropagation Algorithm
```
∂L/∂w = ∂L/∂y · ∂y/∂z · ∂z/∂w
```

## Practical Examples

### Example 1: Linear Regression
```python
import numpy as np

def mse_gradient(X, y, w):
    n = len(y)
    predictions = X.dot(w)
    gradient = (2/n) * X.T.dot(predictions - y)
    return gradient

# Gradient descent for linear regression
def train_linear_regression(X, y, learning_rate=0.01, epochs=1000):
    w = np.random.randn(X.shape[1])
    for epoch in range(epochs):
        gradient = mse_gradient(X, y, w)
        w -= learning_rate * gradient
    return w
```

### Example 2: Neural Network Backprop
```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Chain rule in action
def backprop_step(z, a, delta_next, W_next):
    delta = np.dot(W_next.T, delta_next) * sigmoid_derivative(z)
    return delta
```

## Common Derivative Patterns

### Activation Functions
- Sigmoid: σ'(x) = σ(x)(1 - σ(x))
- ReLU: f'(x) = 1 if x > 0, else 0
- Tanh: tanh'(x) = 1 - tanh²(x)

### Loss Functions
- MSE: ∂L/∂ŷ = 2(ŷ - y)
- Cross-entropy: ∂L/∂ŷ = -(y/ŷ - (1-y)/(1-ŷ))

## Practice Exercises

1. Compute gradient of f(x,y) = x²y + xy²
2. Apply chain rule to composite function
3. Derive backpropagation for 2-layer network
4. Implement gradient checking

## Learning Checklist
- [ ] Master partial derivatives
- [ ] Understand gradient computation
- [ ] Apply chain rule correctly
- [ ] Implement backpropagation
- [ ] Debug gradient calculations

## Debugging Tips
- Gradient checking with finite differences
- Numerical stability considerations
- Vanishing/exploding gradient problems
