
## 🧠 6. Deep Learning – Notes

### 📌 Overview:

Deep Learning is a subset of machine learning that uses **artificial neural networks** with many layers to model complex patterns in data—especially for images, text, and speech.

---

### 🧩 6.1 Neural Networks Basics

#### ✅ Components:

* **Input Layer**: Accepts input features.
* **Hidden Layers**: Perform transformations via weighted connections.
* **Output Layer**: Generates prediction (regression/classification).

#### ✅ Terminology:

* **Weights**: Parameters learned from data
* **Biases**: Extra parameter to shift activation
* **Activation Function**: Adds non-linearity

---

### ⚙️ 6.2 Activation Functions

| Name    | Formula / Behavior          | Usage                        |
| ------- | --------------------------- | ---------------------------- |
| ReLU    | `max(0, x)`                 | Default for hidden layers    |
| Sigmoid | `1 / (1 + e^-x)`            | Binary classification output |
| Tanh    | `(e^x - e^-x)/(e^x + e^-x)` | Rare; better than sigmoid    |
| Softmax | `exp(xi)/sum(exp(xj))`      | Multi-class classification   |

---

### 🔁 6.3 Forward & Backward Propagation

#### ✅ Forward Propagation:

* Calculates output using weights and activations.

#### ✅ Backpropagation:

* Calculates gradients via chain rule
* Updates weights using gradient descent

#### ✅ Loss Functions:

* MSE (Regression)
* Cross-Entropy (Classification)

---

### 🧮 6.4 Optimization Algorithms

| Optimizer | Characteristics                      |
| --------- | ------------------------------------ |
| SGD       | Simple, slow convergence             |
| Momentum  | Helps SGD move faster                |
| Adam      | Adaptive learning rate, most popular |
| RMSprop   | Good for RNNs                        |

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

---

### 🧯 6.5 Regularization Techniques

| Method              | Purpose                                                       |   |   |   |      |
| ------------------- | ------------------------------------------------------------- | - | - | - | ---- |
| Dropout             | Randomly drops neurons during training to prevent overfitting |   |   |   |      |
| L2 Regularization   | Penalizes large weights (\`lambda \*                          |   | w |   | ²\`) |
| Batch Normalization | Normalizes layer inputs for faster convergence                |   |   |   |      |

---

### 🧰 6.6 Implementing with Frameworks

#### ✅ TensorFlow (Keras):

```python
from tensorflow.keras.models import Sequential
model = Sequential([
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

#### ✅ PyTorch:

```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
    def forward(self, x):
        return torch.sigmoid(self.fc1(x))
```

