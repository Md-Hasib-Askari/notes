
## 📘 3.4: **Generative Models** (VAEs + GANs)

### 🎯 Goal

Understand how models can learn to **generate new data** — like images, audio, or text — that resemble training samples.

---

## 🧠 What Are Generative Models?

Generative models learn to capture the **underlying distribution** of data and can **sample new data points** from it.

### 🔄 Contrast with Discriminative Models:

| Discriminative               | Generative                          |
| ---------------------------- | ----------------------------------- |
| Learn **P(y \| x)**          | Learn **P(x)** or **P(x \| y)**     |
| Focus on classification      | Focus on data generation            |
| Example: Logistic Regression | Example: GAN, VAE                   |

---

## 🔮 1. **Variational Autoencoders (VAEs)**

### 🔹 Structure

* **Encoder**: Maps input → latent space (mean + std)
* **Latent space**: A distribution (not a point)
* **Decoder**: Reconstructs data from latent vector

### 🧠 Why "Variational"?

It learns a probability **distribution** over latent variables rather than fixed encodings.

### 🔍 Key Loss Function:

```math
  Loss = Reconstruction  Loss + KL  Divergence
```


### 🔧 PyTorch Sketch:

```python
z = mu + sigma * torch.randn_like(sigma)  # Reparameterization trick
```

### ✅ Use Cases

* Denoising images
* Anomaly detection
* Generative sampling (less sharp than GANs)

---

## 🧨 2. **Generative Adversarial Networks (GANs)**

### 🔹 Core Idea: A two-player game

| Component         | Role                              |
| ----------------- | --------------------------------- |
| **Generator**     | Tries to generate fake data       |
| **Discriminator** | Tries to distinguish real vs fake |

They compete, and both improve.

### 🎯 Objective:

<p align="center">
  <img src="https://github.com/user-attachments/assets/91121f23-ea57-4b16-87b8-d0400e6ab0cf " alt="Objective " />
</p>

### 🔧 GAN Training Pitfalls:

* Mode collapse
* Training instability
* Needs careful tuning (learning rate, architecture)

### ✅ Use Cases

* Image synthesis (e.g., StyleGAN)
* Super-resolution
* Image-to-image translation (Pix2Pix, CycleGAN)

---

## 📊 Comparison: VAE vs GAN

| Feature        | VAE                       | GAN                      |
| -------------- | ------------------------- | ------------------------ |
| Stability      | More stable training      | Can be unstable          |
| Output Quality | Blurry                    | Sharp                    |
| Latent Space   | Structured, interpretable | Often less interpretable |
| Use Case       | Encoding + generation     | Pure generation          |

---

## 🧪 Exercises

### ✅ Conceptual

1. Why do GANs need a discriminator?
2. What does KL divergence enforce in VAEs?

### ✅ Practical

* Train a VAE on MNIST and visualize the latent space.
* Train a simple GAN on Fashion-MNIST.
* Modify a GAN to generate hand-written digits from noise.

