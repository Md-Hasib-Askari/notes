## 🧱 Topic 8: **Convolutional Neural Networks (CNNs)**

### 🎯 Goal:

Understand how CNNs extract **spatial features** from images using convolutional layers — far better than flattening pixel data.

---

### 🤔 Why Not Just Flatten Pixels?

Flattening an image loses spatial relationships (e.g., "cat ears are near the top").
CNNs preserve spatial structure and **learn local patterns** like edges, textures, shapes.

---

### 🧠 Key Components of CNNs:

| Layer Type                   | Function                                         |
| ---------------------------- | ------------------------------------------------ |
| **Convolutional (Conv2D)**   | Detects local patterns (edges, corners, etc.)    |
| **ReLU**                     | Non-linearity — allows learning complex patterns |
| **Pooling (Max/Avg)**        | Downsamples the image, keeps important info      |
| **Fully Connected (Linear)** | Final decision layer                             |
| **Softmax**                  | Converts scores to probabilities                 |

---

### 🔄 CNN Flow:

```
Input Image → Conv → ReLU → Pool → Conv → ReLU → Pool → FC → Output
```

---

### 🧪 PyTorch CNN Example for MNIST

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 28x28 → 14x14
        x = self.pool(F.relu(self.conv2(x)))  # 14x14 → 7x7
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

---

### 🔧 Key Parameters:

| Term            | Meaning                                               |
| --------------- | ----------------------------------------------------- |
| **Channels**    | Input/output depth (e.g., 1 for grayscale, 3 for RGB) |
| **Kernel Size** | Size of sliding window (e.g., 3x3)                    |
| **Stride**      | Step size of kernel movement                          |
| **Padding**     | Add border to maintain size (same padding)            |

---

### 📦 Pretrained CNN Architectures:

* **LeNet** – First CNN, great for digits
* **AlexNet** – Introduced CNN to the world
* **VGG** – Deep and simple
* **ResNet** – Solves vanishing gradients with skip connections

---

### 🧪 Mini Project Ideas:

* Build a CNN for MNIST or Fashion-MNIST.
* Visualize feature maps using hooks.

---

### 📚 Summary:

* CNNs extract spatial features directly from raw images.
* They are the **standard architecture** for image classification.
* From here, you’ll be ready for transfer learning and real-world image tasks.
