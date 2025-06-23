## ✅ Topic 3: Basic Image Processing

This stage covers essential operations used in almost every computer vision pipeline.

---

### 🔹 1. Blurring (Smoothing)

Helps reduce noise and details.

#### a. **Averaging Blur**

```python
blur = cv2.blur(img, (5, 5))  # Kernel size 5x5
```

#### b. **Gaussian Blur**

```python
gaussian = cv2.GaussianBlur(img, (5, 5), 0)
```

#### c. **Median Blur**

```python
median = cv2.medianBlur(img, 5)  # Kernel size must be odd
```

---

### 🔹 2. Thresholding

Converts grayscale images to binary.

#### a. **Simple Threshold**

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
```

#### b. **Inverse Binary Threshold**

```python
_, thresh_inv = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
```

#### c. **Adaptive Threshold**

```python
adaptive = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
)
```

---

### 🔹 3. Edge Detection

Detect object boundaries.

#### a. **Canny Edge Detection**

```python
edges = cv2.Canny(img, 100, 200)  # Threshold1, Threshold2
```

#### b. **Sobel Filter**

```python
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
```

---

### 🔹 4. Morphological Operations

Used to refine binary images.

```python
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
```

#### a. **Erosion**

```python
eroded = cv2.erode(thresh, kernel, iterations=1)
```

#### b. **Dilation**

```python
dilated = cv2.dilate(thresh, kernel, iterations=1)
```

#### c. **Opening (Erosion → Dilation)**

```python
opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
```

#### d. **Closing (Dilation → Erosion)**

```python
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
```

---

### 🧪 Mini Exercise

1. Read an image.
2. Convert to grayscale and apply:

   * Canny edge detection
   * Adaptive thresholding
3. Try erosion and dilation using a (3,3) kernel.
