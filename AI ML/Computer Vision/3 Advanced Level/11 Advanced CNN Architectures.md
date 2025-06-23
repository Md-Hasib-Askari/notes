## 🧠 Topic 11: **Advanced CNN Architectures**

### 🎯 Goal:

Learn modern CNN architectures that improve accuracy, training speed, and generalization by using **deeper**, **wider**, and **smarter** networks.

---

### 📜 Why Go Beyond Basic CNNs?

Simple CNNs:

* Work for small datasets (like MNIST)
* Struggle with deeper learning due to **vanishing gradients**, overfitting, etc.

Advanced architectures solve this via:

* **Skip connections**
* **Parallel convolutions**
* **Depthwise convolutions**
* **Pretraining & transfer learning**

---

### 🧱 Core Architectures to Know:

#### 1️⃣ **LeNet-5 (1998)**

* Early CNN for digit recognition
* Architecture: Conv → Pool → Conv → Pool → FC → FC → Output
* Good starting point

---

#### 2️⃣ **AlexNet (2012)**

* Made CNNs famous by winning ImageNet
* Uses **ReLU**, **Dropout**, **LRN**, **GPU training**
* Architecture: 5 Conv + 3 FC

---

#### 3️⃣ **VGG-16/VGG-19 (2014)**

* Deep but simple (3x3 conv layers)
* Very large model
* Easy to implement

```python
from torchvision.models import vgg16
model = vgg16(pretrained=True)
```

---

#### 4️⃣ **ResNet (2015)** 🔥

* **Skip connections (Residual blocks)** solve vanishing gradient problem
* Can go **100+ layers** deep

```python
from torchvision.models import resnet50
model = resnet50(pretrained=True)
```

---

#### 5️⃣ **Inception (GoogLeNet)**

* Combines multiple kernel sizes in one layer
* Efficient at scale

---

#### 6️⃣ **MobileNet / EfficientNet**

* Designed for mobile/edge devices
* Use **depthwise separable convolutions** to reduce computation

```python
from torchvision.models import mobilenet_v2
model = mobilenet_v2(pretrained=True)
```

---

### ⚙️ When to Use What?

| Model            | Use Case                       |
| ---------------- | ------------------------------ |
| **LeNet**        | Educational                    |
| **AlexNet/VGG**  | Benchmarks, small-medium tasks |
| **ResNet**       | Default for most tasks         |
| **EfficientNet** | Mobile, speed-sensitive apps   |
| **MobileNet**    | Edge deployment                |

---

### 🧪 Transfer Learning with ResNet Example (PyTorch)

```python
from torchvision.models import resnet18
from torchvision import transforms
from torch import nn

model = resnet18(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace final FC layer for 3-class task
model.fc = nn.Linear(model.fc.in_features, 3)
```

---

### 🧪 Mini Project Ideas:

* Fine-tune ResNet on your own image dataset.
* Compare ResNet vs VGG vs MobileNet accuracy and size.
* Build a lightweight object classifier using MobileNet.

---

### 📚 Summary:

* These architectures power **ImageNet**, **COCO**, and real-world CV systems.
* Use pre-trained models and fine-tune for your task — don’t start from scratch!
* **ResNet** is the go-to architecture in most modern pipelines.
