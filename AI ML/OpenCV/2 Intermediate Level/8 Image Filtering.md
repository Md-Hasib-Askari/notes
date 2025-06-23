## ✅ Topic 8: Image Filtering

Image filtering involves applying **kernels** to smooth, sharpen, or detect features like edges. These operations rely on **convolution** or **correlation** of a kernel (matrix) over an image.

---

### 🔹 1. Convolution Basics

A **kernel** (or filter) is a small matrix used to transform the image.

```python
import cv2
import numpy as np

img = cv2.imread("image.jpg")

# Define a 3x3 sharpening kernel
kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])

filtered = cv2.filter2D(img, -1, kernel)
```

---

### 🔹 2. Built-in Filters in OpenCV

#### a. **Box Filter (Averaging)**

```python
blur = cv2.blur(img, (5, 5))  # Simple mean
```

#### b. **Gaussian Filter**

```python
gaussian = cv2.GaussianBlur(img, (5, 5), 0)
```

#### c. **Median Filter**

```python
median = cv2.medianBlur(img, 5)  # Best for salt-and-pepper noise
```

---

### 🔹 3. Edge Detection Filters

#### a. **Sobel Filter** (detects vertical/horizontal edges)

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # X direction
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # Y direction
```

#### b. **Laplacian Filter** (second-order derivative)

```python
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
```

#### c. **Scharr Filter** (more accurate than Sobel)

```python
scharrx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
scharry = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
```

---

### 🔹 4. Custom Kernels

You can define your own filters using any 2D NumPy array:

```python
# Custom emboss kernel
emboss_kernel = np.array([[ -2, -1, 0],
                          [ -1,  1, 1],
                          [  0,  1, 2]])

embossed = cv2.filter2D(img, -1, emboss_kernel)
```

---

### 🧪 Mini Exercise

* Convert an image to grayscale.
* Apply:

  * Sobel X and Y
  * Laplacian
  * A custom sharpen kernel
