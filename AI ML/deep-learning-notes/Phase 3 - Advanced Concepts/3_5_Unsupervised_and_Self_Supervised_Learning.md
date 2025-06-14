

## 📘 3.5: **Unsupervised & Self-Supervised Learning**

### 🎯 Goal

Learn how deep learning models **learn useful representations** without explicit labels — crucial for real-world, unlabeled data scenarios.

---

## 🧠 1. What Is Unsupervised Learning?

**No labels provided.**
Models aim to:

* Discover hidden patterns or structure
* Learn data representations
* Reduce dimensionality or cluster similar data

---

## 📦 2. Key Techniques

### 🔹 **Autoencoders (AE)**

* **Goal**: Learn compact representation by reconstructing input
* **Structure**:

  * Encoder → Latent Vector → Decoder
* **Loss**: MSE between input and reconstruction

#### Types:

* Denoising AE (DAE)
* Sparse AE
* Contractive AE

---

### 🔹 **Clustering Methods**

* **K-Means**: Partition into `k` groups based on distance
* **Hierarchical Clustering**: Builds a hierarchy of clusters
* **DBSCAN**: Density-based clustering

---

## 🔐 3. What Is Self-Supervised Learning?

> A **pretext task** is designed to supervise learning without human labels.

**Examples:**

* Predict the next word (language models)
* Predict missing patches (vision models)
* Match different views (contrastive learning)

---

## 🧠 4. Contrastive Learning

Learn embeddings by **pulling similar samples close** and **pushing dissimilar samples apart**.

### 🔹 Examples:

| Method     | Strategy                                   |
| ---------- | ------------------------------------------ |
| **SimCLR** | Augmentations of the same image = positive |
| **MoCo**   | Momentum contrast encoder for stability    |
| **BYOL**   | Bootstrap learning with no negative pairs  |

**Loss Function** (Simplified):

$$
\mathcal{L}_{contrastive} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{N} \exp(\text{sim}(z_i, z_k)/\tau)}
$$

---

## 📚 Real-World Applications

* **Self-supervised BERT pretraining** (masked language modeling)
* **SimCLR for image representations**
* **Unsupervised clustering for recommendation systems**

---

## 🔧 Tooling Tips

* Use `sklearn` for unsupervised clustering
* Use `PyTorch Lightning Bolts` or `timm` for contrastive learning
* Use `Hugging Face Datasets` to explore unlabeled data

---

## 🧪 Exercises

### ✅ Conceptual

1. How is self-supervised learning different from unsupervised?
2. Why are contrastive losses powerful for representation learning?

### ✅ Practical

* Train a basic autoencoder on MNIST.
* Use SimCLR to learn embeddings on CIFAR-10 (via `lightly` or `solo-learn`).
* Visualize learned embeddings using t-SNE.
