## 🌐 Topic 20: **APIs, Frameworks & Tools**

### 🎯 Goal:

Master the tools and libraries used for building, training, evaluating, visualizing, labeling, and deploying computer vision systems — both traditional and deep learning-based.

---

## 🧰 1. **Core CV Libraries**

| Tool             | Purpose                                     |
| ---------------- | ------------------------------------------- |
| **OpenCV**       | Image processing, video I/O, traditional CV |
| **Pillow (PIL)** | Image manipulation (resize, crop, rotate)   |
| **Scikit-Image** | Classical image processing + filters        |
| **ImageIO**      | Read/write images (TIFF, PNG, etc.)         |
| **Matplotlib**   | Visualization (heatmaps, masks)             |

---

### 🔹 OpenCV Must-Know Features:

```python
import cv2

img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 200)
cv2.imshow("Edges", edges)
```

* Face detection
* Drawing bounding boxes
* Image filtering
* Reading from camera/video

---

## 🤖 2. **Model Frameworks**

| Tool                           | Purpose                                              |
| ------------------------------ | ---------------------------------------------------- |
| **PyTorch / TorchVision**      | Deep learning training, transfer learning            |
| **TensorFlow / Keras**         | Popular alt for training + deployment                |
| **Transformers (HuggingFace)** | ViT, CLIP, BLIP, TrOCR, etc.                         |
| **Ultralytics YOLOv8**         | Plug-and-play detection/segmentation                 |
| **Detectron2**                 | Facebook’s library for object detection/segmentation |
| **Segment Anything (SAM)**     | Meta’s SOTA segmentation tool                        |

---

### 🔹 YOLOv8 Inference Example:

```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
result = model("image.jpg")
result.show()
```

---

### 🔹 Detectron2 Highlights:

* Faster R-CNN, Mask R-CNN, RetinaNet
* Built-in evaluation tools
* High accuracy models out-of-the-box

---

## 🖍️ 3. **Dataset Annotation Tools**

| Tool             | Purpose                                          |
| ---------------- | ------------------------------------------------ |
| **CVAT**         | Web UI to label images, bounding boxes, masks    |
| **LabelImg**     | Simple local tool for object detection labels    |
| **Roboflow**     | Online platform to label, host, convert datasets |
| **Label Studio** | Multimodal annotation (text, images, audio)      |

---

## 📦 4. **Deployment Tools**

| Tool                             | Purpose                       |
| -------------------------------- | ----------------------------- |
| **ONNX**                         | Universal model format        |
| **TorchServe**                   | Serve PyTorch models          |
| **FastAPI**                      | Wrap models in web APIs       |
| **Docker**                       | Containerize and ship CV apps |
| **OpenVINO / TensorRT / TFLite** | Optimize for edge devices     |

---

## 🧪 Mini Projects Using Tools:

* Use **CVAT + YOLOv8** to label and detect trash types.
* Use **Detectron2** to segment vehicles from a dashcam feed.
* Use **FastAPI + Docker** to deploy an OCR API with Tesseract.

---

## 🧠 Summary:

| Tool Type        | Libraries                    |
| ---------------- | ---------------------------- |
| Image Processing | OpenCV, Pillow, Scikit-Image |
| Model Training   | PyTorch, TF, HuggingFace     |
| Object Detection | YOLOv8, Detectron2           |
| Segmentation     | SAM, DeepLab, Mask R-CNN     |
| Annotation       | CVAT, Roboflow               |
| Deployment       | FastAPI, TorchServe, Docker  |
