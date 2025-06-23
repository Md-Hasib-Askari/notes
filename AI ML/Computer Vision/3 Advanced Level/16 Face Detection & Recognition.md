## 🧑‍🤝‍🧑 Topic 16: **Face Detection & Recognition**

### 🎯 Goal:

Detect faces in images and video, then recognize **who** the person is.

---

## 🧭 Two Main Tasks:

| Task                 | Description                                                |
| -------------------- | ---------------------------------------------------------- |
| **Face Detection**   | Locate faces in an image (bounding boxes)                  |
| **Face Recognition** | Identify or verify a person based on their facial features |

---

## 👁️‍🗨️ 1. **Face Detection Methods**

### 🔹 Haar Cascades (Classical)

* Fast, works offline
* Weak on small or rotated faces

```python
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray_img)
```

---

### 🔹 DNN-based Detectors (Better Accuracy)

**OpenCV DNN Face Detector (ResNet10-based):**

```python
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')
```

* Input: image blob
* Output: confidence scores + bounding boxes

---

## 🧠 2. **Face Recognition**

### 🔹 How it works:

1. **Face embedding**: Convert face image → 128D vector (feature)
2. **Compare vectors**: Use cosine similarity or Euclidean distance

---

### 🔥 Popular Libraries:

| Library         | Description                                              |
| --------------- | -------------------------------------------------------- |
| **FaceNet**     | Embedding-based recognition, highly accurate             |
| **DeepFace**    | Wrapper over multiple models (VGGFace, Facenet, ArcFace) |
| **Dlib**        | Lightweight but powerful                                 |
| **InsightFace** | State-of-the-art, real-world scale                       |

---

### 🧪 DeepFace Example:

```python
from deepface import DeepFace

result = DeepFace.verify("img1.jpg", "img2.jpg")
print("Same person?" , result["verified"])
```

### 🧪 FaceNet Embedding Flow:

```python
# Step 1: Detect face
# Step 2: Align face (crop + resize)
# Step 3: Pass through FaceNet → get 128D embedding
# Step 4: Store or compare embeddings for recognition
```

---

## 🔍 3. **Liveness Detection**

Prevents spoofing using photos or videos.

| Technique                 | Description                       |
| ------------------------- | --------------------------------- |
| **Blink detection**       | Detects eye movement              |
| **Texture analysis**      | Real skin vs printed paper        |
| **3D depth check**        | RGB-D or stereo cameras           |
| **CNN-based classifiers** | Deep learning for spoof detection |

---

## 🧪 Mini Project Ideas:

* Build a webcam-based face login system.
* Compare multiple people’s faces using embeddings.
* Add liveness detection to block printed-photo attacks.
* Recognize faces from your CCTV videos (Imou IPC compatible).

---

## 🧠 Summary:

| Component         | Tool                       |
| ----------------- | -------------------------- |
| Face detection    | Haar, Dlib, OpenCV-DNN     |
| Face embedding    | FaceNet, DeepFace, ArcFace |
| Recognition logic | Vector comparison          |
| Anti-spoofing     | Liveness detection or CNNs |
