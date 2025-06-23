## ✅ Topic 16: OpenCV with NumPy for Custom Operations

We'll dive into:

* Direct pixel manipulation
* Replacing slow loops with vectorized NumPy
* Creating custom filters efficiently

---

### 🔹 1. Access & Modify Pixels with NumPy

```python
import cv2
import numpy as np

img = cv2.imread("image.jpg")

# Access pixel at row=50, col=100 (BGR)
b, g, r = img[50, 100]
img[50, 100] = [0, 255, 0]  # Set to green
```

You can modify full regions too:

```python
img[100:150, 200:250] = [255, 0, 0]  # Blue block
```

---

### 🔹 2. Convert to Grayscale with NumPy

Instead of using OpenCV’s `cvtColor`, do it manually:

```python
gray = np.dot(img[..., :3], [0.114, 0.587, 0.299]).astype(np.uint8)
```

---

### 🔹 3. Vectorized Custom Filter

Custom 3×3 average filter using NumPy:

```python
kernel = np.ones((3, 3)) / 9.0

def convolve2d(img, kernel):
    from scipy.signal import convolve2d
    return convolve2d(img, kernel, mode='same', boundary='symm').astype(np.uint8)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = convolve2d(gray, kernel)
```

---

### 🔹 4. Replace Slow Loops with Vectorization

❌ Slow:

```python
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        img[i, j] = 255 - img[i, j]
```

✅ Fast:

```python
inverted = 255 - img  # Vectorized
```

---

### 🔹 5. Thresholding with NumPy

```python
binary = np.where(gray > 127, 255, 0).astype(np.uint8)
```

---

### 🔹 6. Masking with NumPy

```python
mask = (gray > 100) & (gray < 150)
result = img.copy()
result[~mask] = 0  # Black out all other areas
```

---

### 🧪 Mini Exercise

* Load an image and invert its colors using NumPy
* Apply your own edge detector (e.g., Sobel kernel using `convolve2d`)
* Mask only pixels in a certain brightness range
