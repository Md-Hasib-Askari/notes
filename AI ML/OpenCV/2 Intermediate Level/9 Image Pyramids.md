## ✅ Topic 9: Image Pyramids

**Image pyramids** are a way to represent images at multiple resolutions. Useful for object detection at different scales, seamless image blending, and more.

There are two types:

* **Gaussian Pyramid**: Downsampling (reducing size)
* **Laplacian Pyramid**: Captures edges/details lost between Gaussian levels

---

### 🔹 1. Gaussian Pyramid

#### a. **Reduce (Downsample)**

```python
import cv2

img = cv2.imread("image.jpg")

lower_reso = cv2.pyrDown(img)  # 0.5x smaller
```

#### b. **Expand (Upsample)**

```python
higher_reso = cv2.pyrUp(lower_reso)  # 2x larger
```

#### c. **Multiple Levels**

```python
layer = img.copy()
for i in range(3):
    layer = cv2.pyrDown(layer)
    cv2.imshow(f'Level {i+1}', layer)
```

---

### 🔹 2. Laplacian Pyramid

The Laplacian pyramid highlights **differences between pyramid levels** (image details lost when downsampling).

```python
import numpy as np

img = cv2.imread("image.jpg")
lower = cv2.pyrDown(img)
higher = cv2.pyrUp(lower)

laplacian = cv2.subtract(img, higher)
cv2.imshow("Laplacian", laplacian)
```

---

### 🔹 3. Image Blending Using Pyramids

**Smoothly blend two images** (e.g., apple and orange) using Laplacian pyramid:

#### a. Steps:

1. Build Gaussian pyramids for both images
2. Build Laplacian pyramids
3. Join left half of A and right half of B at each level
4. Reconstruct the image

**Full pyramid blending example can be provided on request**

---

### 🧪 Mini Exercise

* Build a 3-level Gaussian pyramid
* Reconstruct the original image using `pyrUp`
* Subtract the original from upsampled version to visualize Laplacian
