## ✅ Topic 15: Deep Learning with OpenCV (`cv2.dnn` Module)

The `cv2.dnn` module allows you to **load and run deep learning models** (like YOLO, MobileNet, SSD) without needing TensorFlow or PyTorch directly.

---

### 🔹 1. Load a Pretrained Model (e.g., Caffe / ONNX / TensorFlow)

#### Example: Load a MobileNet-SSD (Caffe model)

```python
import cv2

net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "mobilenet_iter_73000.caffemodel")
```

---

### 🔹 2. Create a Blob from Input Image

A **blob** is a preprocessed input (resized, normalized, and channel-swapped).

```python
img = cv2.imread("image.jpg")
blob = cv2.dnn.blobFromImage(img, scalefactor=0.007843, size=(300, 300), mean=(127.5, 127.5, 127.5), swapRB=True)

net.setInput(blob)
detections = net.forward()
```

---

### 🔹 3. Parse Output

Loop through predictions and draw boxes.

```python
h, w = img.shape[:2]
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.5:
        box = detections[0, 0, i, 3:7] * [w, h, w, h]
        (x1, y1, x2, y2) = box.astype("int")
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
```

---

### 🔹 4. Load YOLOv3/v4/v5 (Darknet Weights)

```python
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
```

#### Inference:

```python
blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
outputs = net.forward(output_layers)
```

---

### 🔹 5. Load ONNX Models (e.g., Pose Estimation, Face Detection)

```python
net = cv2.dnn.readNetFromONNX("model.onnx")
net.setInput(blob)
outputs = net.forward()
```

📌 OpenCV >= 4.3 has strong ONNX support.

---

### 🔹 6. Use with CUDA (if supported)

```python
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
```

---

### 🧪 Mini Exercise

* Download MobileNet SSD Caffe model + prototxt
* Load an image, detect objects, and draw bounding boxes
