## 📘 Topic 2.1: **Feedforward Neural Networks (FNN)**

### 🧠 What Is It?

A **Feedforward Neural Network (FNN)** is the most basic type of deep neural network where the data moves in one direction—**from input to output**—with no cycles or loops.

---

### 🔁 **1. Backpropagation**

Backpropagation is the core learning algorithm that:

1. Performs a **forward pass** to compute predictions.
2. Calculates the **loss**.
3. Uses the **chain rule** to propagate gradients **backward**.
4. Updates weights using **gradient descent**.

> It’s how the model “learns” by adjusting weights to minimize the loss.

#### Math Sketch:

Given:  

![image](https://github.com/user-attachments/assets/f231c196-e441-4a25-89a8-7478a6c4bafe)


Then:

<p align="center">
  <img src="https://github.com/user-attachments/assets/65125cc8-2370-487f-ad36-ee22e8564989 " alt="Backpropagation" />
</p>

---

### 🧰 **2. Gradient Descent Variants**

#### 🔹 **SGD (Stochastic Gradient Descent)**

* Updates weights on each sample.
* Faster, but noisier.

#### 🔹 **Mini-Batch Gradient Descent**

* Update based on a small group of samples.
* Balances stability and speed.

#### 🔹 **Adam (Adaptive Moment Estimation)**

* Combines momentum + adaptive learning rates.
* Default go-to optimizer for many DL tasks.

#### 🔹 **RMSProp**

* Adapts learning rate per parameter using moving average of squared gradients.
* Useful for recurrent tasks.

---

### 🧪 Exercises

#### ✅ Conceptual

1. Why is backpropagation essential?
2. When should you use Adam over SGD?
3. How does mini-batch gradient descent differ from SGD?

---

#### ✅ Coding Exercise (FNN with Manual Backprop in NumPy)

```python
import numpy as np

# Simple FNN with 1 hidden layer

# Sigmoid
def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x): return sigmoid(x) * (1 - sigmoid(x))

# Data: XOR
X = np.array([[0,0],[0,1],[1,0],[1,1]])
Y = np.array([[0],[1],[1],[0]])

# Init weights
np.random.seed(1)
W1 = np.random.randn(2, 4)
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1)
b2 = np.zeros((1, 1))
lr = 0.1

for epoch in range(10000):
    # Forward
    Z1 = X.dot(W1) + b1
    A1 = sigmoid(Z1)
    Z2 = A1.dot(W2) + b2
    A2 = sigmoid(Z2)

    # Loss (MSE)
    loss = np.mean((Y - A2)**2)

    # Backward
    dA2 = A2 - Y
    dZ2 = dA2 * sigmoid_derivative(Z2)
    dW2 = A1.T.dot(dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = dZ2.dot(W2.T)
    dZ1 = dA1 * sigmoid_derivative(Z1)
    dW1 = X.T.dot(dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    # Update
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    if epoch % 1000 == 0:
        print(f"Epoch {epoch} Loss: {loss:.4f}")

# Final prediction
print("Predictions:", A2.round())
```
