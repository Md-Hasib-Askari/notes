
## 🟡 **5. Generative Adversarial Networks (GANs)**

### 🔍 **Key Concepts**

#### ✅ What Is a GAN?

A GAN is made up of **two neural networks** that compete:

* **Generator (G):** Creates fake data from random noise.
* **Discriminator (D):** Tries to distinguish real data from fake data.

#### 🥊 Adversarial Training

* **Generator’s goal:** Fool the Discriminator.
* **Discriminator’s goal:** Correctly classify real vs. fake.
* **Objective (min-max game):**

  $$
  \min_G \max_D \; \mathbb{E}[\log D(x)] + \mathbb{E}[\log(1 - D(G(z)))]
  $$

---

### 🧠 Why This Matters in Generative AI

GANs are used to generate:

* Realistic **images**, **videos**, **music**, **text**
* Style transfer, super-resolution, deepfakes
* Foundation of **StyleGAN**, **CycleGAN**, etc.

---

### ✍️ GAN Variants

| Type                       | Use Case                            |
| -------------------------- | ----------------------------------- |
| **DCGAN**                  | Stable image generation             |
| **Conditional GAN (cGAN)** | Label-controlled generation         |
| **CycleGAN**               | Unpaired image-to-image translation |
| **StyleGAN**               | High-res, realistic face generation |

---

### 🧪 Practice Exercises

#### ✅ 1. Build a Basic GAN with MNIST (PyTorch)

```python
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Data
dataloader = DataLoader(
    datasets.MNIST('.', download=True, transform=transforms.ToTensor()),
    batch_size=128, shuffle=True)

# Generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Tanh()
        )

    def forward(self, z):
        return self.gen(z).view(-1, 1, 28, 28)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.dis = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.dis(x)

G = Generator()
D = Discriminator()

loss_fn = nn.BCELoss()
optim_G = torch.optim.Adam(G.parameters(), lr=2e-4)
optim_D = torch.optim.Adam(D.parameters(), lr=2e-4)

# Training loop (3 epochs for simplicity)
for epoch in range(3):
    for real, _ in dataloader:
        batch_size = real.size(0)

        # Labels
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        # Train Discriminator
        z = torch.randn(batch_size, 100)
        fake = G(z).detach()
        real_pred = D(real)
        fake_pred = D(fake)
        loss_D = loss_fn(real_pred, real_labels) + loss_fn(fake_pred, fake_labels)

        optim_D.zero_grad()
        loss_D.backward()
        optim_D.step()

        # Train Generator
        z = torch.randn(batch_size, 100)
        fake = G(z)
        pred = D(fake)
        loss_G = loss_fn(pred, real_labels)

        optim_G.zero_grad()
        loss_G.backward()
        optim_G.step()

print("GAN training done.")
```

---

### 📌 Mini Project Idea

> 🔨 **Task**: Train a DCGAN to generate MNIST digits.
> Bonus: Use `matplotlib` to visualize generated digits every 5 epochs and save GIFs of progress.

