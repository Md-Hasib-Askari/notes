## 📘 Topic 1.6: **Intro to Neural Networks**

### 🧠 What is a Neural Network?

A **neural network** is a series of mathematical functions that map input data to output predictions. It’s inspired (loosely) by how neurons work in the brain.

---

### 🧱 1. **Perceptron (Single-Layer)**

The **perceptron** is the simplest neural unit:

<p align="center">
  <img src="https://github.com/user-attachments/assets/48b7301a-8985-45d8-bdc1-2bfcd308a329" alt="Perceptron (Single-Layer)" />
</p>

![image](https://github.com/user-attachments/assets/130e9305-1c14-4955-b86b-e263a12d8d35)

---

### 🔁 2. **Feedforward Neural Network (Multi-Layer Perceptron)**

* **Input layer**: raw data
* **Hidden layers**: learn patterns
* **Output layer**: final prediction

Each layer performs:

<p align="center">
  <img src="https://github.com/user-attachments/assets/174a0654-829b-47b4-acb5-f768619f46e3" alt="Feedforward Neural Network Function" />
</p>

---

### 🔑 3. **Activation Functions**

These introduce **non-linearity** so the model can learn complex patterns.

#### 🔸 **Sigmoid**

<p align="center">
  <img src="https://github.com/user-attachments/assets/f339c1d9-8d14-43a2-a3c4-82657f731e62" alt="Sigmoid " />
</p>

#### 🔸 **Tanh**

<p align="center">
  <img src="https://github.com/user-attachments/assets/badfbdc6-337c-42c2-b86e-006dde06ded8" alt="Tanh " />
</p>

#### 🔸 **ReLU (Rectified Linear Unit)**

<p align="center">
  <img src="https://github.com/user-attachments/assets/c8c43c25-5b0d-49f1-aaa4-50b1d2f8efa3" alt="ReLU (Rectified Linear Unit)" />
</p>

---

### 🧮 4. **Cost (Loss) Functions**

Quantify how bad your model's prediction is.

#### For regression:  

![image](https://github.com/user-attachments/assets/16791035-cbd6-44a8-8309-d0c00df7cac0)


#### For classification:

* **Cross-entropy**:

<p align="center">
  <img src="https://github.com/user-attachments/assets/40a2fc8f-1049-4bd7-803e-1245f0e91c1f " alt="Cross-entropy " />
</p>

---

### ⚙️ 5. **Training a Neural Network**

1. **Forward pass** → predict
2. **Loss computation** → calculate error
3. **Backward pass** (backpropagation) → compute gradients
4. **Weight update** → using gradient descent or optimizers like Adam

---

### 🧪 Exercises

#### ✅ Conceptual

1. What happens if you remove the activation function?
2. Why is ReLU preferred in deep networks?
3. What’s the difference between MSE and cross-entropy?

---

#### ✅ Code (Simple NN using PyTorch)

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Simple dataset: XOR logic
X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
Y = torch.tensor([[0],[1],[1],[0]], dtype=torch.float32)

# Model definition
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

model = SimpleNN()
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training
for epoch in range(1000):
    out = model(X)
    loss = loss_fn(out, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Test
with torch.no_grad():
    print("Predictions:", model(X).round().squeeze())
```
