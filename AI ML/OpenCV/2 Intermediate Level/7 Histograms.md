## ✅ Topic 7: Histograms

Histograms in OpenCV help visualize the **distribution of pixel intensities**, and they're key for tasks like contrast enhancement, thresholding, and image comparison.

---

### 🔹 1. Grayscale Histogram

```python
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)

hist = cv2.calcHist([img], [0], None, [256], [0, 256])

plt.plot(hist)
plt.title("Grayscale Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.show()
```

---

### 🔹 2. Color Histogram

```python
img = cv2.imread("image.jpg")
colors = ('b', 'g', 'r')

for i, col in enumerate(colors):
    hist = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)

plt.title("Color Histogram")
plt.show()
```

---

### 🔹 3. Histogram Equalization (Grayscale Only)

Improves contrast by spreading out intensity values.

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
equalized = cv2.equalizeHist(gray)

cv2.imshow("Original", gray)
cv2.imshow("Equalized", equalized)
```

---

### 🔹 4. CLAHE (Contrast Limited Adaptive Histogram Equalization)

Local contrast enhancement with limit control.

```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cl1 = clahe.apply(gray)
```

---

### 🔹 5. Histogram Backprojection

Finds regions that match a histogram (used in object tracking).

```python
roi = img[100:200, 100:200]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

cv2.imshow("BackProjection", dst)
```

---

### 🧪 Mini Exercise

* Plot the grayscale and color histogram of any image.
* Apply histogram equalization to improve contrast.
* Try CLAHE on a low-light image.
