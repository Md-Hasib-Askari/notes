## 🎯 Topic 12: **Object Detection**

### 🧠 Goal:

Object detection not only tells you **what** is in the image, but also **where** it is (bounding boxes + class labels).

---

## 📦 Two Main Categories:

| Type                    | Description               | Examples                 |
| ----------------------- | ------------------------- | ------------------------ |
| **Traditional**         | Uses handcrafted features | Haar Cascades, HOG + SVM |
| **Deep Learning-based** | Uses CNNs or transformers | YOLO, SSD, Faster R-CNN  |

---

## 🧰 Traditional Methods (Outdated but Educational)

### 🔹 Haar Cascades (OpenCV):

* Uses pre-trained XML files to detect faces, eyes, etc.
* Good for quick demos, poor accuracy.

```python
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
```

---

### 🔹 HOG + SVM (Pedestrian Detection):

* Extract HOG features, classify using SVM.

```python
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
boxes, _ = hog.detectMultiScale(img)
```

---

## 🔥 Deep Learning-Based Object Detectors

### 🔹 1. YOLO (You Only Look Once)

* Real-time object detection.
* Treats detection as regression.
* Latest version: **YOLOv8** (Ultralytics, PyTorch)

```bash
pip install ultralytics
```

```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
results = model("image.jpg")
results.show()
```

---

### 🔹 2. SSD (Single Shot Detector)

* Faster but less accurate than Faster R-CNN.
* Uses multiple feature maps to detect objects at different scales.

---

### 🔹 3. Faster R-CNN

* Two-stage detector (region proposal + classification).
* Slower but more accurate.

```python
from torchvision.models.detection import fasterrcnn_resnet50_fpn
model = fasterrcnn_resnet50_fpn(pretrained=True)
```

---

### 📐 Bounding Box Format:

* Usually `[x_min, y_min, x_max, y_max]`
* Sometimes `[center_x, center_y, width, height]`

---

### 🔁 Object Detection Pipeline:

1. Input image
2. Forward pass through model
3. Get bounding boxes + class probabilities
4. Apply **Non-Max Suppression (NMS)** to filter overlapping boxes
5. Draw boxes on the image

---

### 🧪 Mini Project Ideas:

* Use YOLOv8 to detect objects in your webcam in real-time.
* Annotate and train your own model using Roboflow or LabelImg.
* Build a people counter with bounding boxes.

---

### 📚 Summary:

* Object detection is essential for **autonomous driving**, **surveillance**, **retail**, and more.
* Use **YOLO** for real-time needs, **Faster R-CNN** for accuracy-first tasks.
* Next stop: **Semantic and Instance Segmentation** — pixel-level detection.
