## 🛠️ Topic 19: **Computer Vision in Production**

### 🎯 Goal:

Learn how to **optimize**, **deploy**, and **run** CV models efficiently on real-world devices — from servers to smartphones and Raspberry Pis.

---

## ⚙️ 1. **Model Optimization Techniques**

### 🔹 Convert to ONNX (Open Neural Network Exchange)

* Platform-agnostic format
* Useful for TensorRT, OpenCV-DNN, etc.

```python
torch.onnx.export(model, dummy_input, "model.onnx")
```

---

### 🔹 Use TensorRT (NVIDIA GPUs)

* Highly optimized inference on NVIDIA GPUs
* Up to **10x faster** inference than PyTorch

```bash
# Convert ONNX to TensorRT
trtexec --onnx=model.onnx --saveEngine=model.trt
```

---

### 🔹 Quantization

* Reduce model size and speed up inference by using **int8** or **float16** instead of float32

```python
from torch.quantization import quantize_dynamic
quantized_model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
```

---

### 🔹 Pruning

* Remove redundant weights to make models lighter

---

## 🕓 2. **Real-Time Inference**

### Tools:

| Tool             | Use                             |
| ---------------- | ------------------------------- |
| **OpenCV DNN**   | Use models directly in OpenCV   |
| **ONNX Runtime** | Lightweight, multi-platform     |
| **TensorRT**     | GPU-only, ultra fast            |
| **TFLite**       | TensorFlow Lite for mobile/edge |

---

### 🧪 Webcam Inference Example (YOLOv8 + OpenCV):

```python
from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    result = model(frame)
    result[0].plot()
    cv2.imshow("YOLO", frame)
    if cv2.waitKey(1) == ord("q"):
        break
```

---

## 🌍 3. **Edge Deployment**

| Platform         | Use                                   |
| ---------------- | ------------------------------------- |
| **Raspberry Pi** | Lightweight object detection / OCR    |
| **Jetson Nano**  | Edge AI accelerator with CUDA support |
| **Coral TPU**    | Google’s USB stick for inference      |

### 🧪 Tools:

* TensorFlow Lite
* ONNX Runtime Mobile
* PyTorch Mobile
* OpenVINO (Intel devices)

---

## 📦 Serving Models (Cloud/API)

| Tool                        | Description                                   |
| --------------------------- | --------------------------------------------- |
| **FastAPI / Flask**         | Simple REST API around your model             |
| **TorchServe**              | Serve PyTorch models at scale                 |
| **Triton Inference Server** | NVIDIA enterprise-grade inference server      |
| **Docker**                  | Containerize your app for deployment anywhere |

---

### Example: Serving with FastAPI

```python
from fastapi import FastAPI, UploadFile
from PIL import Image
import torch

app = FastAPI()
model = torch.load("model.pt")

@app.post("/predict")
async def predict(file: UploadFile):
    img = Image.open(file.file)
    # preprocess → predict → return result
    return {"label": "cat"}
```

---

## 🧪 Mini Project Ideas:

* Deploy YOLOv8 on Jetson Nano for real-time detection.
* Convert your PyTorch OCR model to ONNX + host on a REST API.
* Quantize and deploy a FaceNet model on a mobile app.

---

## 🧠 Summary:

| Step           | Tool                           |
| -------------- | ------------------------------ |
| Optimize model | ONNX, quantization, pruning    |
| Inference      | TensorRT, OpenCV DNN, TFLite   |
| Serve          | FastAPI, Docker, TorchServe    |
| Deploy         | Raspberry Pi, Jetson, Edge TPU |

This is where computer vision meets **real-world engineering**. 💥
