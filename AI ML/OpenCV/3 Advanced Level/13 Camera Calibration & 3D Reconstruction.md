## ✅ Topic 13: Camera Calibration & 3D Reconstruction

This topic is key for **removing lens distortion**, understanding **camera perspective**, and enabling **depth estimation** via stereo vision.

---

### 🔹 1. Why Calibrate?

Real-world cameras distort images (barrel/pincushion effects). Calibration helps:

* Correct distortion
* Get intrinsic/extrinsic camera parameters
* Enable 3D reconstruction

---

### 🔹 2. Capture Calibration Images

Use a **chessboard pattern** (e.g., 9×6 squares) and capture 10–15 images from different angles.

---

### 🔹 3. Find Chessboard Corners

```python
import cv2
import numpy as np

objp = np.zeros((6*9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)  # 9x6 grid

objpoints = []  # 3D points in real world
imgpoints = []  # 2D points in image plane

images = [...]  # list of calibration images

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)
        cv2.drawChessboardCorners(img, (9, 6), corners, ret)
```

---

### 🔹 4. Calibrate Camera

```python
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# mtx = camera matrix
# dist = distortion coefficients
```

---

### 🔹 5. Undistort Images

```python
dst = cv2.undistort(img, mtx, dist, None, mtx)
```

---

### 🔹 6. Stereo Vision & Depth Maps

Use two cameras/images to estimate depth via disparity:

```python
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(left_gray, right_gray)
cv2.imshow("Depth Map", disparity)
```

📌 `left_gray` and `right_gray` are stereo pair grayscale images

---

### 🔹 7. Re-projection to 3D

```python
Q = cv2.stereoRectify(...)  # Get the reprojection matrix
points_3D = cv2.reprojectImageTo3D(disparity, Q)
```

---

### 🧪 Mini Exercise

* Download sample chessboard images
* Use them to calibrate a camera and undistort one image
* Try creating a depth map using a stereo image pair
