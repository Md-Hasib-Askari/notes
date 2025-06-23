## 🧩 OpenCV Structure Overview

OpenCV is organized into **modular modules** (C++ backend with Python bindings), covering everything from core image operations to deep learning and 3D vision.

---

### 🔹 1. **Core Module (`cv2.core`)**

Low-level operations, matrix math, basic data structures.

* `cv2.Mat`: Core image container (in C++)
* Data types: `cv2.CV_8UC1`, `CV_32FC3`, etc.
* Basic operations: `cv2.add()`, `cv2.subtract()`, `cv2.multiply()`

---

### 🔹 2. **Image Processing Module (`cv2.imgproc`)**

Functions for filtering, edge detection, transformations.

* Filters: `cv2.GaussianBlur()`, `cv2.medianBlur()`
* Thresholding: `cv2.threshold()`, `cv2.adaptiveThreshold()`
* Morphology: `cv2.erode()`, `cv2.dilate()`
* Geometric transforms: `cv2.resize()`, `cv2.warpAffine()`

---

### 🔹 3. **High-Level GUI Module (`cv2.highgui`)**

Functions for windowing, displaying, and interacting with images.

* `cv2.imshow()`, `cv2.waitKey()`, `cv2.destroyAllWindows()`
* GUI sliders, mouse events

---

### 🔹 4. **Image I/O Module (`cv2.imgcodecs`)**

Image read/write support.

* `cv2.imread()`, `cv2.imwrite()`

---

### 🔹 5. **Video I/O Module (`cv2.videoio`)**

Capture and save video streams.

* `cv2.VideoCapture()`, `cv2.VideoWriter()`

---

### 🔹 6. **Video Analysis (`cv2.video`)**

Motion tracking, background subtraction, optical flow.

* `cv2.calcOpticalFlowPyrLK()`, `cv2.createBackgroundSubtractorMOG2()`

---

### 🔹 7. **Feature Detection and Description (`cv2.features2d`)**

Keypoints, descriptors, matching.

* `cv2.ORB_create()`, `cv2.SIFT_create()`, `cv2.BFMatcher()`

---

### 🔹 8. **Object Detection Module (`cv2.objdetect`)**

* Face detection with Haar cascades
* QR code detection: `cv2.QRCodeDetector`

---

### 🔹 9. **Machine Learning Module (`cv2.ml`)**

Built-in classical ML algorithms.

* SVM, kNN, Decision Trees, Random Forest
* `cv2.ml.SVM_create()`

---

### 🔹 10. **Deep Learning (`cv2.dnn`)**

Load and run pre-trained deep models (Caffe, ONNX, TensorFlow).

* `cv2.dnn.readNetFromONNX()`, `net.forward()`, `blobFromImage()`

---

### 🔹 11. **3D Vision / Calibration (`cv2.calib3d`)**

Stereo vision, camera calibration, depth maps.

* `cv2.calibrateCamera()`, `cv2.findChessboardCorners()`, `cv2.stereoBM_create()`

---

### 🔹 12. **GPU Acceleration (OpenCL) (`cv2.ocl`, `cv2.UMat`)**

OpenCL backend for speed.

* `cv2.UMat`, `cv2.ocl.setUseOpenCL(True)`

---

### 🔹 13. **Utility Modules**

* `cv2.utils`: Logging, error handling
* `cv2.cuda`: NVIDIA-specific GPU ops (C++ only, limited Python)

---

## 🗂️ Summary Tree

```
cv2
├── core        ← Math, data structures
├── imgproc     ← Filtering, edges, transforms
├── highgui     ← GUI display
├── imgcodecs   ← Read/write images
├── videoio     ← Video read/write
├── video       ← Motion, background subtraction
├── features2d  ← Keypoints, matching
├── objdetect   ← Haar, QR detection
├── ml          ← Classical ML models
├── dnn         ← Deep learning inference
├── calib3d     ← Camera calibration, stereo vision
├── ocl / UMat  ← OpenCL acceleration
├── cuda        ← GPU operations (limited in Python)
```
