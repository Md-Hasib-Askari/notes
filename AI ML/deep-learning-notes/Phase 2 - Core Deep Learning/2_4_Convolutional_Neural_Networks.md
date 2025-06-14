## 📘 Topic 2.4: Convolutional Neural Networks (CNNs)

### 🎯 Goal

Understand how CNNs process visual data using spatial patterns and learn hierarchical features.

---

## 🧠 Core Concepts

### 1. **Convolution Operation**

* Slides a **filter/kernel** across the input image.
* Captures local patterns (e.g., edges, textures).
* Reduces number of parameters compared to fully connected networks.

```plaintext
Input (e.g. 28x28) → Kernel (e.g. 3x3) → Feature Map
```

---

### 2. **Filter/Kernel**

* A small matrix (e.g. 3x3) applied across input.
* Each filter learns to detect specific features (edges, curves, etc.).

```python
Conv2D(32, kernel_size=(3,3))  # 32 filters of size 3x3
```

---

### 3. **Stride & Padding**

* **Stride**: How far the filter moves.
* **Padding**: Adds border to keep output size constant.

| Type  | Effect                         |
| ----- | ------------------------------ |
| VALID | No padding → smaller output    |
| SAME  | Adds padding → keeps size same |

---

### 4. **Pooling (Subsampling)**

* Reduces spatial dimensions (downsampling).
* Common: **Max Pooling**, which selects max value in region.

```python
MaxPool2D(pool_size=(2, 2))
```

---

### 5. **Typical CNN Architecture**

```plaintext
Input Image
→ [Conv → ReLU → Pool] × N
→ Flatten
→ Fully Connected (Dense)
→ Softmax / Sigmoid
```

---

## 🔧 PyTorch Example

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 10)  # Assuming 28x28 input

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 28x28 → 14x14
        x = self.pool(F.relu(self.conv2(x)))  # 14x14 → 7x7
        x = x.view(-1, 32 * 7 * 7)
        return self.fc1(x)
```

---

## 🔧 TensorFlow Example

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10)  # No softmax if using logits
])
```

---

## 🧱 Famous Architectures to Study

| Model     | Key Feature                    |
| --------- | ------------------------------ |
| LeNet-5   | First CNN (handwritten digits) |
| AlexNet   | ReLU + GPU usage               |
| VGGNet    | Stacked small filters          |
| ResNet    | Skip connections               |
| Inception | Parallel filters (multi-scale) |

---

## 🧪 Exercises

### ✅ Conceptual

1. Why do we use small filters like 3×3?
2. How does padding affect the output size?
3. What’s the benefit of pooling layers?

### ✅ Practical

* Train a CNN on MNIST (use either PyTorch or TensorFlow).
* Visualize the learned filters of the first conv layer.
* Try removing pooling and see how performance changes.
