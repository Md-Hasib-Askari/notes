
## 🟡 **6. Variational Autoencoders (VAEs)**

### 🔍 **Key Concepts**

#### ✅ What Is a VAE?

A **Variational Autoencoder** is a type of autoencoder that:

* Learns a **probability distribution** over the latent space.
* Uses **sampling** from this distribution to generate new data.

#### 📦 VAE Architecture

* **Encoder**: Outputs two vectors — mean $\mu$ and log-variance $\log(\sigma^2)$
* **Reparameterization trick**:

  $$
  z = \mu + \sigma \cdot \epsilon \quad \text{where} \quad \epsilon \sim \mathcal{N}(0, 1)
  $$
* **Decoder**: Reconstructs input from $z$

#### 🎯 Loss Function

VAE optimizes a combination of:

1. **Reconstruction Loss** (like MSE)
2. **KL Divergence**: Ensures latent space resembles a normal distribution

$$
\text{Loss} = \text{Reconstruction Loss} + \text{KL}(q(z|x) || p(z))
$$

---

### 🧠 Why This Matters in Generative AI

* More **stable** than GANs
* Enables **interpolation** and **controlled generation**
* Useful in **text**, **images**, **molecular design**

---

### ✍️ Comparison with GANs

| Feature         | VAE                       | GAN                     |
| --------------- | ------------------------- | ----------------------- |
| Training Stable | ✅                         | ❌ (can be unstable)     |
| Sample Quality  | Average (blurry)          | High (sharp)            |
| Latent Control  | ✅ Structured latent space | ❌ Poor latent structure |
| Probabilistic   | ✅                         | ❌                       |

---

### 🧪 Practice Exercise

#### ✅ VAE on MNIST (PyTorch)

```python
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Data
loader = DataLoader(
    datasets.MNIST('.', download=True, transform=transforms.ToTensor()),
    batch_size=128, shuffle=True)

# VAE Model
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc_mu = nn.Linear(400, 20)
        self.fc_logvar = nn.Linear(400, 20)
        self.fc_dec = nn.Sequential(
            nn.Linear(20, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.fc_dec(z)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

model = VAE()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(3):
    for x, _ in loader:
        optimizer.zero_grad()
        recon, mu, logvar = model(x)
        loss = loss_function(recon, x, mu, logvar)
        loss.backward()
        optimizer.step()

print("VAE trained.")
```

---

### 📌 Mini Project Idea

> 🔨 **Task**: Train a VAE on MNIST or Fashion-MNIST.
> Bonus: Visualize the **2D latent space** by reducing latent dimensions and plotting reconstructions.

