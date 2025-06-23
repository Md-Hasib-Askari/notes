## 🎨 Topic 14: **Generative Models (Autoencoders, GANs, VAEs)**

### 🎯 Goal:

Understand how to train models that don’t just recognize images — they **create** them.

---

## 🧠 Key Generative Models:

### 🔹 1. **Autoencoders (AEs)**

* Learn to **compress** an image to a low-dimensional space (encoding) and **reconstruct** it (decoding).
* Useful for **denoising**, **compression**, **anomaly detection**.

```python
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 64),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 28*28),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x.view(-1, 1, 28, 28)
```

---

### 🔹 2. **Variational Autoencoders (VAEs)**

* A probabilistic extension of autoencoders.
* Encodes image into **distribution parameters (μ, σ)**.
* Samples from distribution → generates new images.
* Useful for **controlled generation**, **latent space math**.

---

### 🔹 3. **Generative Adversarial Networks (GANs)**

> “Two networks — a Generator and a Discriminator — play a game.”

* **Generator (G)**: Learns to generate fake but realistic images.
* **Discriminator (D)**: Learns to distinguish real vs fake.

They compete until G gets good at fooling D.

```python
# G(z) → fake image
# D(x) → probability image is real
```

---

### 🔥 GAN Variants:

| Model        | Purpose                                |
| ------------ | -------------------------------------- |
| **DCGAN**    | Basic image GAN                        |
| **Pix2Pix**  | Image-to-image translation             |
| **CycleGAN** | Style transfer (e.g., horse ↔ zebra)   |
| **StyleGAN** | State-of-the-art human face generation |

---

### 🧪 PyTorch Snippet: GAN Training Loop

```python
for real_imgs in dataloader:
    # 1. Train Discriminator
    z = torch.randn(batch_size, latent_dim)
    fake_imgs = generator(z)
    real_loss = criterion(discriminator(real_imgs), torch.ones(batch_size, 1))
    fake_loss = criterion(discriminator(fake_imgs.detach()), torch.zeros(batch_size, 1))
    D_loss = real_loss + fake_loss
    D_loss.backward()

    # 2. Train Generator
    output = discriminator(fake_imgs)
    G_loss = criterion(output, torch.ones(batch_size, 1))  # Fool the discriminator
    G_loss.backward()
```

---

### 🧪 Mini Project Ideas:

* Build an autoencoder to denoise MNIST digits.
* Train a GAN to generate fake MNIST or Fashion-MNIST images.
* Try CycleGAN for style transfer between photos and paintings.

---

### 🧠 Real-World Applications:

| Use Case                      | Model             |
| ----------------------------- | ----------------- |
| Face generation               | StyleGAN          |
| Super-resolution              | SRGAN             |
| Denoising / anomaly detection | Autoencoders      |
| Art and style transfer        | CycleGAN, Pix2Pix |

---

### 📚 Summary:

* AEs and VAEs learn to **reconstruct** and **understand** images.
* GANs learn to **generate entirely new** images — super useful for augmentation, art, and more.
* These are critical for modern **Generative AI** systems.
