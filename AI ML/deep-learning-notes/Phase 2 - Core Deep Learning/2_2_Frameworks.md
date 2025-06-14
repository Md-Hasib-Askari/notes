## 📘 Topic 2.2: **Frameworks (PyTorch + TensorFlow)**

### 🧠 Why Use Frameworks?

Manual matrix math (like NumPy) is fine to learn the internals, but real deep learning requires:

* **GPU acceleration**
* **Automatic differentiation**
* **Model checkpointing**
* **Data pipelines**

That’s where **PyTorch** and **TensorFlow** come in.

---

## 🔷 PyTorch (Preferred for Researchers & Beginners)

### ✅ Key Features:

* Pythonic & intuitive
* Dynamic computation graph (eager execution)
* Clean debugging (just use `print()` and `pdb`)
* Massive community and academic support

### 🧱 Key Components

| Concept           | PyTorch Class                       |
| ----------------- | ----------------------------------- |
| Tensor            | `torch.Tensor`                      |
| Neural net layers | `torch.nn.Module`                   |
| Optimizers        | `torch.optim`                       |
| Loss functions    | `torch.nn.functional` or `torch.nn` |

---

### 🔁 Basic PyTorch Workflow

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Dataset (XOR)
X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
Y = torch.tensor([[0],[1],[1],[0]], dtype=torch.float32)

# 2. Define model
class FNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

model = FNN()

# 3. Loss and optimizer
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 4. Train loop
for epoch in range(1000):
    pred = model(X)
    loss = loss_fn(pred, Y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 5. Prediction
with torch.no_grad():
    print(model(X).round())
```

---

## 🔶 TensorFlow + Keras (More Production-Oriented)

### ✅ Key Features:

* Static computation graph (optimized performance)
* Production-ready (used by Google)
* Keras API is beginner-friendly and high-level

---

### 🔁 Basic TensorFlow Workflow

```python
import tensorflow as tf
import numpy as np

# Dataset (XOR)
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
Y = np.array([[0],[1],[1],[0]], dtype=np.float32)

# Model definition
model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training
model.fit(X, Y, epochs=1000, verbose=0)

# Prediction
print("Predictions:", model.predict(X).round())
```

---

### ✅ Summary: PyTorch vs TensorFlow

| Feature            | PyTorch       | TensorFlow (Keras)      |
| ------------------ | ------------- | ----------------------- |
| Graph Type         | Dynamic       | Static (eager optional) |
| Syntax             | More Pythonic | More abstract           |
| Debugging          | Easier        | Harder (but improving)  |
| Research adoption  | Higher        | Growing                 |
| Production support | Okay          | Excellent (TF Serving)  |
| Learning curve     | Moderate      | Very easy with Keras    |

---

### 🧪 Exercises

#### ✅ Conceptual

1. Why is dynamic graph execution useful for research?
2. When should you choose TensorFlow over PyTorch?
3. What does `model.parameters()` return in PyTorch?
