
## 📘 3.1: **Transfer Learning**

### 🎯 Goal

Leverage pre-trained models to solve new tasks with minimal data and computation.

---

## 🧠 Core Concepts

### 1. **What is Transfer Learning?**

> Reusing a pre-trained model’s learned knowledge (on a large dataset like ImageNet) for a new, often smaller, dataset.

### 2. **Two Approaches**

#### 🔹 **Feature Extraction**

* Freeze most layers of the pretrained model.
* Replace final classifier layer(s) and train only those.
* ✅ Faster, less data-hungry, ideal for small datasets.

#### 🔹 **Fine-Tuning**

* Unfreeze some top layers and retrain with a small learning rate.
* ✅ More flexible, better performance if you have more data.

---

## 🔧 Pretrained Model Examples

| Model            | Application                         |
| ---------------- | ----------------------------------- |
| **ResNet**       | Image classification, segmentation  |
| **EfficientNet** | High-accuracy + efficiency          |
| **BERT/GPT**     | NLP: classification, QA, generation |
| **CLIP**         | Multimodal vision + language        |

---

## ⚙️ Using Transfer Learning in PyTorch

```python
import torchvision.models as models
import torch.nn as nn

# Load pretrained ResNet
model = models.resnet50(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace final FC layer
model.fc = nn.Linear(model.fc.in_features, 2)  # Binary classification
```

To **fine-tune**, unfreeze some layers:

```python
for param in model.layer4.parameters():
    param.requires_grad = True
```

---

## ⚙️ Using Transfer Learning in TensorFlow/Keras

```python
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=x)
```

---

## 🧪 Exercises

### ✅ Conceptual

1. What are the pros/cons of feature extraction vs. fine-tuning?
2. When would you choose to fine-tune a model?

### ✅ Practical

* Load a pretrained CNN and classify a small custom dataset (e.g., cats vs. dogs).
* Compare results using feature extraction vs. fine-tuning.
* Visualize learned features with Grad-CAM.


