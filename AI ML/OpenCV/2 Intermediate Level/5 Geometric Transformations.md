## ✅ Topic 5: Geometric Transformations

These operations change the **geometry** of an image: position, size, and orientation.

---

### 🔹 1. Translation (Shifting)

Move the image left/right or up/down.

```python
import cv2
import numpy as np

img = cv2.imread("image.jpg")
height, width = img.shape[:2]

# Shift right 100px and down 50px
M = np.float32([[1, 0, 100], [0, 1, 50]])
translated = cv2.warpAffine(img, M, (width, height))
```

---

### 🔹 2. Rotation

Rotate around the center (or any point).

```python
# Rotate by 45 degrees clockwise around center
center = (width // 2, height // 2)
M = cv2.getRotationMatrix2D(center, 45, 1.0)  # angle, scale
rotated = cv2.warpAffine(img, M, (width, height))
```

---

### 🔹 3. Scaling (Resizing)

Resize using interpolation.

```python
scaled_up = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
scaled_down = cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA)
```

---

### 🔹 4. Affine Transformation

Requires 3 points from input and output.

```python
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])

M = cv2.getAffineTransform(pts1, pts2)
affine = cv2.warpAffine(img, M, (width, height))
```

---

### 🔹 5. Perspective Transformation (Warp)

Needs 4 source points → 4 destination points.

```python
pts1 = np.float32([[56,65], [368,52], [28,387], [389,390]])
pts2 = np.float32([[0,0], [300,0], [0,300], [300,300]])

M = cv2.getPerspectiveTransform(pts1, pts2)
perspective = cv2.warpPerspective(img, M, (300, 300))
```

---

### 🧪 Mini Exercise

* Load an image.
* Shift it to the right and down.
* Rotate it 45°.
* Apply perspective transformation.
