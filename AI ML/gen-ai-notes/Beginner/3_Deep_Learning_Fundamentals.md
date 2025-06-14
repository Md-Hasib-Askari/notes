
## 🟢 **3. Deep Learning Fundamentals**

### 🔍 **Key Concepts**

#### ✅ Neural Networks

* Layers of neurons simulate the brain.
* Each layer performs:

  $$
  \text{output} = \text{activation}(W \cdot X + b)
  $$

#### 🔄 Backpropagation

* Computes gradients via the chain rule to update weights.
* Loss gradients are passed backward through the network.

#### ⚡ Activation Functions

* **ReLU**: `max(0, x)`
* **Sigmoid**: $\frac{1}{1+e^{-x}}$
* **Tanh**: $\frac{e^x - e^{-x}}{e^x + e^{-x}}$

#### 📉 Loss Functions

* **MSE (Mean Squared Error)**: For regression
* **Cross-Entropy**: For classification

#### 🛠️ Optimizers

* **SGD**: Simple gradient step
* **Adam**: Momentum + adaptive learning rate

#### 🔧 Frameworks

* Start with **PyTorch** or **TensorFlow** (PyTorch preferred for research/ease of debugging)

---

### 🧠 Why This Matters in Generative AI

* Every generative model (GAN, VAE, Transformer) is built using neural networks.
* Backpropagation + optimization = model learns to "generate" realistic outputs.
* Understanding activations & loss functions is essential for stability and quality.

---

### 🧪 Practice Exercises

1. Build a 2-layer neural network from scratch (using NumPy) for binary classification.
2. Implement ReLU and Sigmoid functions in Python.
3. Train a basic classifier using PyTorch on MNIST.

```python
# Sample PyTorch MNIST model
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Data
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('.', train=True, download=True,
        transform=transforms.ToTensor()), batch_size=64, shuffle=True)

# Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.fc(x)

model = Net()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(3):
    for x, y in train_loader:
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()

print("Training done!")
```

---

### 📌 Mini Project

> 🔨 **Task**: Build an image classifier for CIFAR-10 or MNIST using PyTorch.
> Start with a basic feedforward net, then try adding convolutional layers (CNN) to improve performance.

