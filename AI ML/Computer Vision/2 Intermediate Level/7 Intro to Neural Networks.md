## 🧠 Topic 7: **Intro to Neural Networks**

### 🎯 Goal:

Understand the core mechanics of how a basic neural network works — the building block of deep learning in computer vision.

---

### 🤖 What Is a Neural Network?

A **Neural Network** is a system of connected layers that learn patterns by adjusting internal weights.

> Input (features) → Hidden Layers → Output (predicted label)

---

### 🔧 Components of a Basic Neural Network:

| Component      | Description                               |
| -------------- | ----------------------------------------- |
| **Neuron**     | Computes weighted sum + activation        |
| **Layer**      | Group of neurons                          |
| **Weights**    | Parameters the network learns             |
| **Bias**       | Additional parameter to shift activation  |
| **Activation** | Non-linear function (ReLU, Sigmoid, etc.) |

---

### 🔁 Training Process

1. **Forward Propagation**
   Input → prediction

2. **Loss Calculation**
   Compare prediction vs. true label

3. **Backward Propagation**
   Compute gradients (how much to adjust weights)

4. **Weight Update**
   Use gradient descent to reduce error

---

### 🔣 Common Activation Functions

| Function | Range          | Use                            |
| -------- | -------------- | ------------------------------ |
| ReLU     | 0 to ∞         | Most common in hidden layers   |
| Sigmoid  | 0 to 1         | For binary classification      |
| Softmax  | 0 to 1 (sum=1) | For multi-class classification |

---

### 🛠️ Simple Neural Network in PyTorch (Image Classifier)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 classes

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

---

### 🧠 Tips:

* Flatten images into 1D vectors for basic NNs.
* Use ReLU for hidden layers, Softmax for output.
* Use **CrossEntropyLoss** for classification tasks.

---

### 📉 Loss Functions for CV:

| Task           | Loss Function     |
| -------------- | ----------------- |
| Classification | CrossEntropyLoss  |
| Regression     | MSELoss           |
| Binary Class   | BCEWithLogitsLoss |

---

### 📚 Summary:

* Neural networks are the **core engine** of deep learning.
* Start with simple feedforward NNs before going to CNNs.
* Next: Learn how to use **CNNs** for actual image data.
