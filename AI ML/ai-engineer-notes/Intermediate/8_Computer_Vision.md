

## 👁️ 8. Computer Vision – Notes

### 📌 Overview:

Computer Vision enables machines to **understand visual content** from images and videos using deep learning and neural networks.

---

### 🧠 8.1 Image Basics

#### ✅ Key Concepts:

* **Pixel**: Smallest unit of an image (RGB values)
* **Channels**: 3 for RGB, 1 for grayscale
* **Shape**: `(height, width, channels)`

#### ✅ Image Preprocessing:

* Resize, crop
* Normalize (mean subtraction, divide by std)
* Augmentation (flip, rotate, color shift)

---

### 🧱 8.2 Convolutional Neural Networks (CNNs)

#### ✅ Architecture Components:

| Layer           | Purpose               |
| --------------- | --------------------- |
| Conv2D          | Feature extraction    |
| ReLU            | Non-linearity         |
| MaxPooling      | Downsampling          |
| Fully Connected | Classification output |

#### ✅ Example:

```python
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16 * 6 * 6, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = x.view(-1, 16 * 6 * 6)
        return self.fc(x)
```

---

### 🧠 8.3 Transfer Learning

#### ✅ Idea:

* Reuse pretrained CNNs (e.g., ResNet, EfficientNet) trained on large datasets like ImageNet.
* Fine-tune on smaller, task-specific datasets.

#### ✅ Tools:

```python
from torchvision import models
model = models.resnet18(pretrained=True)
```

---

### 📦 8.4 Object Detection & Segmentation

#### ✅ Object Detection:

* Outputs: Bounding boxes + class labels
* Popular Models: YOLO, SSD, Faster R-CNN

#### ✅ Image Segmentation:

* **Semantic**: Classifies each pixel (e.g., UNet, DeepLab)
* **Instance**: Detects individual objects + masks (e.g., Mask R-CNN)

---

### 🔧 8.5 Tools & Libraries

| Library              | Use Case                       |
| -------------------- | ------------------------------ |
| OpenCV               | Image/video processing         |
| `torchvision`        | Datasets & transforms          |
| `albumentations`     | Advanced augmentation          |
| Detectron2           | Facebook's detection framework |
| YOLOv8 (Ultralytics) | Real-time detection            |

