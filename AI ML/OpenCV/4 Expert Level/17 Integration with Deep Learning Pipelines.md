## ✅ Topic 17: Integration with Deep Learning Pipelines

This is where OpenCV becomes a **glue layer** between data, models, and visualization in PyTorch/TensorFlow pipelines.

---

### 🔹 1. Preprocessing with OpenCV for DL Models

#### Example: Prepare an image for a PyTorch classifier

```python
import cv2
import torch
import torchvision.transforms as transforms

img = cv2.imread("image.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

# Resize, normalize, and convert to tensor
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

tensor = transform(img).unsqueeze(0)  # Add batch dimension
```

---

### 🔹 2. Run Inference (PyTorch)

```python
from torchvision import models

model = models.resnet18(pretrained=True)
model.eval()

with torch.no_grad():
    output = model(tensor)
    predicted = torch.argmax(output, 1).item()
```

---

### 🔹 3. Use OpenCV for Postprocessing

#### Draw predicted class on the image:

```python
cv2.putText(img, f"Class: {predicted}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

cv2.imshow("Prediction", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
```

---

### 🔹 4. OpenCV + TensorFlow Example (ONNX)

```python
net = cv2.dnn.readNetFromONNX("model.onnx")

blob = cv2.dnn.blobFromImage(img, 1/255.0, (224, 224), swapRB=True)
net.setInput(blob)
output = net.forward()
```

---

### 🔹 5. OpenCV to Visualize DL Outputs (e.g., Object Detection)

```python
# Let's say `boxes`, `scores`, `classes` are outputs from a model

for box, score, label in zip(boxes, scores, classes):
    if score > 0.5:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, str(label), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
```

---

### 🔹 6. Use OpenCV for NMS (Non-Maximum Suppression)

```python
indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.5, nms_threshold=0.4)
```

---

### 🧪 Mini Exercise

* Use OpenCV to load an image
* Preprocess for PyTorch ResNet
* Predict and visualize the output label using OpenCV
* Bonus: Load a YOLOv5 ONNX model and run it using OpenCV
