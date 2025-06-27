# Neural Networks Basics

## Introduction to Neural Networks

### Biological Inspiration
- Neurons and synapses
- Information processing
- Learning and adaptation
- Pattern recognition

### Artificial Neurons
- Mathematical model
- Weighted inputs
- Activation function
- Output computation

## Perceptron Model

### Single Perceptron
- Linear classifier
- Binary classification
- Limitation to linearly separable data

### Mathematical Representation
```
output = activation(Σ(wᵢ × xᵢ) + bias)
```

### Implementation
```python
import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Training loop
        for _ in range(self.max_iterations):
            for i in range(n_samples):
                # Forward pass
                prediction = self.predict_single(X[i])
                
                # Update weights if prediction is wrong
                if prediction != y[i]:
                    self.weights += self.learning_rate * y[i] * X[i]
                    self.bias += self.learning_rate * y[i]
    
    def predict_single(self, x):
        return 1 if np.dot(x, self.weights) + self.bias > 0 else 0
    
    def predict(self, X):
        return np.array([self.predict_single(x) for x in X])
```

## Multi-Layer Perceptrons (MLPs)

### Architecture
- Input layer
- Hidden layers
- Output layer
- Fully connected structure

### Forward Propagation
1. Compute weighted sum for each neuron
2. Apply activation function
3. Pass output to next layer
4. Repeat until output layer

### Activation Functions

#### Common Functions
```python
# Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ReLU (Rectified Linear Unit)
def relu(x):
    return np.maximum(0, x)

# Tanh (Hyperbolic Tangent)
def tanh(x):
    return np.tanh(x)

# Softmax (for multi-class classification)
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)
```

#### Properties and Use Cases
- **Sigmoid**: Smooth, bounded (0,1), vanishing gradient problem
- **ReLU**: Simple, efficient, addresses vanishing gradients
- **Tanh**: Zero-centered, bounded (-1,1)
- **Softmax**: Probability distribution for multi-class

## Backpropagation Algorithm

### Chain Rule Application
- Compute gradients layer by layer
- Work backwards from output
- Update weights and biases

### Mathematical Foundation
```
∂Loss/∂w = ∂Loss/∂output × ∂output/∂activation × ∂activation/∂w
```

### Implementation
```python
class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.learning_rate = learning_rate
        
        # Initialize weights randomly
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def forward(self, X):
        # Forward propagation
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y, output):
        m = X.shape[0]
        
        # Backward propagation
        dz2 = output - y
        dW2 = (1/m) * np.dot(self.a1.T, dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * sigmoid_derivative(self.z1)
        dW1 = (1/m) * np.dot(X.T, dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        # Update parameters
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
    
    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Backward pass
            self.backward(X, y, output)
            
            if epoch % 100 == 0:
                loss = np.mean((output - y) ** 2)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))
```

## Loss Functions

### For Regression
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Huber Loss

### For Classification
- Binary Cross-Entropy
- Categorical Cross-Entropy
- Sparse Categorical Cross-Entropy

```python
def binary_cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def categorical_cross_entropy(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
```

## Training Considerations

### Weight Initialization
- Random initialization
- Xavier/Glorot initialization
- He initialization

### Learning Rate
- Too high: Overshooting
- Too low: Slow convergence
- Adaptive methods: Adam, RMSprop

### Regularization
- L1 and L2 penalties
- Dropout
- Early stopping

## Practical Example

### XOR Problem Solution
```python
# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Create and train network
nn = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1)
nn.train(X, y, epochs=5000)

# Test predictions
predictions = nn.forward(X)
print("XOR Predictions:")
for i in range(len(X)):
    print(f"{X[i]} -> {predictions[i][0]:.3f} (target: {y[i][0]})")
```

## Universal Approximation Theorem

### Key Insights
- Neural networks can approximate any continuous function
- Sufficient hidden units required
- Practical limitations exist

### Implications
- Theoretical foundation for deep learning
- Justification for using neural networks
- Importance of architecture design

## Common Challenges

### Vanishing Gradients
- Problem in deep networks
- Gradients become very small
- Solutions: ReLU, batch normalization, residual connections

### Overfitting
- Memorizing training data
- Poor generalization
- Solutions: regularization, dropout, more data

### Local Minima
- Non-convex optimization
- Gradient descent limitations
- Solutions: random restarts, momentum, advanced optimizers

## Learning Objectives
- [ ] Understand neural network fundamentals
- [ ] Implement forward and backward propagation
- [ ] Choose appropriate activation functions
- [ ] Apply to classification and regression
- [ ] Address common training challenges

## Resources
- "Deep Learning" by Ian Goodfellow
- Neural Networks and Deep Learning (online book)
- TensorFlow and PyTorch tutorials
