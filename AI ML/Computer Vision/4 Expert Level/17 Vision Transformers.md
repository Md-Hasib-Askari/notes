## 🧠 Topic 17: **Vision Transformers (ViT)**

### 🎯 Goal:

Learn how transformers — originally built for text — are now revolutionizing computer vision tasks like classification, segmentation, and detection.

---

## 🤔 Why Move Beyond CNNs?

CNNs:

* Great at **local feature extraction**
* But struggle with **global context** and **long-range dependencies**

Transformers:

* Capture **global attention** across an image
* Are **scalable**, **modular**, and align well with multi-modal tasks

---

## 📦 1. Vision Transformer (ViT)

### 🔹 Core Idea:

* Divide image into patches (e.g., 16×16)
* Flatten each patch → linear embedding
* Add position encoding
* Feed into standard transformer encoder (like in NLP)

```plaintext
Image → Patches → Linear Embedding + Positional Encoding → Transformer → Class Token → Output
```

### 🔧 Code Overview (using Hugging Face):

```python
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import torch

image = Image.open("cat.jpg")
extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
inputs = extractor(images=image, return_tensors="pt")

model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
outputs = model(**inputs)
logits = outputs.logits
pred = torch.argmax(logits, dim=1)
```

---

## 🌀 2. Swin Transformer (Shifted Windows)

* Introduces **hierarchical patching** like CNNs
* Uses **shifted window attention** to reduce complexity
* Better for **dense prediction** (segmentation, detection)

```python
from transformers import SwinModel
```

Used in: **Swin-Unet**, **YOLOS**, **Detectron2-Swin**

---

## 🔪 3. SAM – Segment Anything Model (Meta AI)

> A general model that can segment any object with zero-shot prompts.

### Features:

* Promptable with **points**, **boxes**, or **text**
* Works out-of-the-box on new objects/domains
* Used with CLIP, BLIP, etc. for multimodal CV

```bash
pip install segment-anything
```

---

## 🧠 Advantages of ViTs:

| Feature           | CNN      | ViT       |
| ----------------- | -------- | --------- |
| Local patterns    | ✅        | ✅         |
| Global attention  | ❌        | ✅         |
| Scalability       | Moderate | Excellent |
| Transfer learning | ✅        | ✅         |

---

## 🧪 Mini Project Ideas:

* Compare ViT vs ResNet on CIFAR or your own dataset.
* Use SAM to segment unknown objects interactively.
* Combine ViT with CLIP for zero-shot classification.

---

## 📚 Summary:

| Model    | Use Case                          |
| -------- | --------------------------------- |
| **ViT**  | Image classification              |
| **Swin** | Detection, segmentation           |
| **SAM**  | Anything segmentation (zero-shot) |

Vision Transformers are the **future of computer vision**, especially in multimodal and generalist AI systems.
