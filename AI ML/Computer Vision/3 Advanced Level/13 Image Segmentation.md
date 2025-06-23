## 🧩 Topic 13: **Image Segmentation**

### 🎯 Goal:

Instead of just drawing boxes around objects (detection), segmentation **labels every pixel** in the image as part of an object or background.

---

### 🧠 Types of Segmentation:

| Type                      | Description                                                              | Example          |
| ------------------------- | ------------------------------------------------------------------------ | ---------------- |
| **Semantic Segmentation** | Labels each pixel with a class (no distinction between object instances) | All cars = “car” |
| **Instance Segmentation** | Labels each object **instance** separately                               | Car #1 ≠ Car #2  |
| **Panoptic Segmentation** | Combines semantic + instance (newest)                                    | Unified view     |

---

### 📦 Popular Models

| Model          | Type     | Description                              |
| -------------- | -------- | ---------------------------------------- |
| **U-Net**      | Semantic | Great for medical images, pixel-accurate |
| **DeepLabV3+** | Semantic | State-of-the-art accuracy                |
| **Mask R-CNN** | Instance | Adds a segmentation head to Faster R-CNN |

---

### 🛠️ U-Net Architecture (Simplified)

```
Image → Conv → Pool ↓
      → Conv → Pool ↓
         ...         → Bottleneck
      ↑ UpConv ← Skip Connections ← Earlier Layers
```

> Skip connections help preserve spatial details lost in downsampling.

---

### 🧪 PyTorch Example: Semantic Segmentation with torchvision

```python
from torchvision.models.segmentation import deeplabv3_resnet50
import torch

model = deeplabv3_resnet50(pretrained=True).eval()
input_tensor = preprocess(image).unsqueeze(0)  # shape: [1, 3, H, W]
output = model(input_tensor)['out']
```

---

### 📍 Output:

* For semantic segmentation: tensor of shape `[batch, classes, H, W]`
* Use `argmax(dim=1)` to get the predicted class for each pixel.

---

### 🎨 Visualizing Segmentation:

Use color maps (e.g., `matplotlib.cm`, OpenCV) to turn label maps into colored masks.

```python
import matplotlib.pyplot as plt
plt.imshow(segmentation_map, cmap='jet')
```

---

### 🧪 Mini Project Ideas:

* Use DeepLabV3 to segment road, sky, people from an image.
* Train U-Net on medical dataset (e.g., lung scans, cell segmentation).
* Combine segmentation + object detection for hybrid systems.

---

### 🧰 Tools for Annotation:

* [LabelMe](http://labelme.csail.mit.edu)
* [CVAT](https://cvat.org)
* [Supervisely](https://supervise.ly)

---

### 📚 Summary:

* Segmentation gives **pixel-perfect understanding** of images.
* Use **U-Net** for pixel-level tasks, **Mask R-CNN** for instance-level tasks.
* Powerful for **medical imaging**, **autonomous vehicles**, and **AR** apps.
