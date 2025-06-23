## 🧱 Topic 1: **Basics of Image Processing**

### 📌 What is an Image?

* An image is a matrix of pixels.

  * **Grayscale image**: 2D array → shape = (height, width)
  * **RGB image**: 3D array → shape = (height, width, 3)
  * Each pixel holds intensity values: 0–255 for 8-bit images.

---

### 🗂️ Common Image Formats:

| Format | Description                 | Compression |
| ------ | --------------------------- | ----------- |
| JPEG   | Lossy, good for photography | ✅           |
| PNG    | Lossless, supports alpha    | ❌           |
| BMP    | Uncompressed, large size    | ❌           |

---

### 🔧 Basic Operations:

* **Resizing**: `cv2.resize(img, (w, h))`
* **Cropping**: `img[y1:y2, x1:x2]`
* **Flipping**: `cv2.flip(img, 0)` (vert), `cv2.flip(img, 1)` (horiz)
* **Rotating**:

  ```python
  center = (w//2, h//2)
  M = cv2.getRotationMatrix2D(center, angle, scale)
  rotated = cv2.warpAffine(img, M, (w, h))
  ```

---

### 🧪 Practical Example:

```python
import cv2

# Read image
img = cv2.imread('image.jpg')

# Resize
resized = cv2.resize(img, (300, 300))

# Crop
cropped = img[100:300, 200:400]

# Flip
flipped = cv2.flip(img, 1)

# Show
cv2.imshow("Result", flipped)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

### 📚 Summary:

* An image = matrix of pixel values.
* Understand different image file types and operations like resize, crop, flip, rotate.
* These are **must-know tools** before diving into detection, classification, or segmentation.
