
## 🟡 **4. Autoencoders (AEs)**

### 🔍 **Key Concepts**

#### ✅ What Is an Autoencoder?

* A **neural network** trained to **reconstruct** its input.
* Composed of two parts:

  * **Encoder**: Compresses input into latent space.
  * **Decoder**: Reconstructs original input from latent space.

#### 🔄 Training Objective

* Minimize **reconstruction loss** (usually Mean Squared Error):

  $$
  \text{Loss} = ||X - \hat{X}||^2
  $$

---

### 🧠 Why This Matters in Generative AI

* Autoencoders are the basis for:

  * **Variational Autoencoders (VAEs)**
  * **Denoising Autoencoders**
  * Image compression & generation

---

### ✍️ Types of Autoencoders

| Type                     | Purpose                                   |
| ------------------------ | ----------------------------------------- |
| **Vanilla AE**           | Learn compressed representation           |
| **Denoising AE**         | Learn to reconstruct from corrupted input |
| **Sparse AE**            | Enforce sparsity in latent space          |
| **Variational AE (VAE)** | Probabilistic encoding for generation     |

---

### 🧪 Practice Exercises

#### ✅ 1. Build a Vanilla Autoencoder in PyTorch (on MNIST)

```python
import torch
import torch.nn as nn
from torchvision import datasets, transforms

# Data loader
loader = torch.utils.data.DataLoader(
    datasets.MNIST('.', download=True, transform=transforms.ToTensor()),
    batch_size=128, shuffle=True)

# AE Model
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 64),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 28*28),
            nn.Sigmoid(),  # Outputs in [0,1]
            nn.Unflatten(1, (1, 28, 28))
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = Autoencoder()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(3):
    for x, _ in loader:
        optimizer.zero_grad()
        recon = model(x)
        loss = loss_fn(recon, x)
        loss.backward()
        optimizer.step()

print("Autoencoder trained.")
```

#### ✅ 2. Modify the above to:

* Add **noise to input images** and train it as a **denoising AE**.
* Use a smaller latent size (e.g., 8 or 16) to compress more.

---

### 📌 Mini Project Idea

> 🔨 **Task**: Use a denoising autoencoder to clean noisy MNIST digits.
>
> Bonus: Visualize original vs. noisy vs. reconstructed images side-by-side using `matplotlib`.


