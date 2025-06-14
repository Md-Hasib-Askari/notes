

## 📘 3.2: Attention Mechanism

### 🎯 Goal

Understand how attention enables models to dynamically focus on relevant input features — the basis of the **Transformer** architecture.

---

## 🧠 Core Concepts

### 1. **Why Attention?**

Traditional RNNs/CNNs treat input features with equal importance or rely on fixed-size windows.
⚠️ Problem: They struggle with **long-range dependencies**.

**Attention** solves this by letting the model learn *what to focus on*.

---

### 2. **Basic Idea**

Each output is a **weighted sum** of all input tokens — where weights represent **importance**.

<p align="center">
  <img src="https://github.com/user-attachments/assets/0478fc09-c9c2-4403-b316-c3eb6d9b0874" alt="Attention " />
</p>

* **Q (Query)**: What we want to find.
* **K (Key)**: What we compare with.
* **V (Value)**: The actual content.

---

### 3. **Types of Attention**

| Type               | Description                                               |
| ------------------ | --------------------------------------------------------- |
| **Additive**       | Used in early sequence models (e.g., Bahdanau)            |
| **Dot-Product**    | More efficient, used in Transformers (scaled dot product) |
| **Self-Attention** | A token attends to all others in the same sequence        |

---

## 🔄 Self-Attention (Used in Transformers)

Each token computes attention **with respect to every other token** in the sequence.

Example:

* In “The cat sat on the mat”, the word “sat” may pay more attention to “cat” than to “the”.

---

## 🔀 Multi-Head Attention

Instead of computing one attention distribution, do it **multiple times in parallel**, then concatenate outputs.

This allows the model to **attend to information at different positions and subspaces**.

<p align="center">
  <img src="https://github.com/user-attachments/assets/54edc68e-4768-4b11-85c3-42f408482025 " alt="Multi-Head Attention " />
</p>

---

## 🔧 PyTorch Example (Simplified)

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / d_k ** 0.5
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, V)
```

---

## 📚 Real-world Use

* **Transformers** (BERT, GPT): Use stacked self-attention layers.
* **Vision Transformers (ViT)**: Use attention over image patches.
* **Speech models**, **recommendation engines**, and more.

---

## 🧪 Exercises

### ✅ Conceptual

1. Why is self-attention superior to RNNs for long sequences?
2. Explain the role of queries, keys, and values.

### ✅ Practical

* Visualize attention scores from a pre-trained BERT model.
* Implement simple self-attention from scratch.
* Compare outputs of self-attention vs. RNN on a toy sequence.

