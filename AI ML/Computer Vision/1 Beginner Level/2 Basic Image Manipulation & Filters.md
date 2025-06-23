## 🔎 Topic 2: **Basic Image Manipulation & Filters**

### 🎯 Goal:

Learn how filters (aka kernels) modify images by performing operations like blurring, edge detection, and sharpening.

---

### 🧠 Key Concepts:

#### 🔹 1. **Convolution (Kernel Operation)**

* Convolution is applying a small matrix (kernel) to an image.
* It slides over the image and calculates a **dot product** at each location.

**Example kernel: 3x3 blur**

```
1/9 * [[1, 1, 1],
       [1, 1, 1],
       [1, 1, 1]]
```

---

#### 🔹 2. **Blurring (Smoothing)**

Used to reduce noise or detail.

| Type         | Function                                  |
| ------------ | ----------------------------------------- |
| Average Blur | Uniform smoothing                         |
| Gaussian     | Weighted smoothing using Gaussian formula |
| Median       | Good for salt-and-pepper noise            |

```python
cv2.blur(img, (5,5))
cv2.GaussianBlur(img, (5,5), 0)
cv2.medianBlur(img, 5)
```

---

#### 🔹 3. **Edge Detection**

Highlights edges (sudden intensity changes).

| Method        | Description                               |
| ------------- | ----------------------------------------- |
| **Sobel**     | Computes gradient in X/Y directions       |
| **Laplacian** | Detects edges in all directions           |
| **Canny**     | Advanced edge detection (uses thresholds) |

```python
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1)
laplacian = cv2.Laplacian(img, cv2.CV_64F)
canny = cv2.Canny(img, 100, 200)
```

---

#### 🔹 4. **Thresholding**

Converts grayscale to binary (black/white) image.

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
```

---

#### 🔹 5. **Dilation & Erosion**

* **Dilation** adds pixels to object boundaries.
* **Erosion** removes pixels from boundaries.

Used in noise removal, object separation.

```python
kernel = np.ones((5,5), np.uint8)
dilated = cv2.dilate(binary, kernel, iterations=1)
eroded = cv2.erode(binary, kernel, iterations=1)
```

---

### 🧪 Mini Project Idea:

* Load a grayscale image.
* Apply edge detection (Canny).
* Dilate the edges.
* Overlay on the original image to create a "highlighted edges" effect.

---

### 📚 Summary:

* Filters are the foundation of many computer vision tasks.
* Learn to **blur**, **sharpen**, **detect edges**, and **clean up** images.
* Use OpenCV for hands-on practice.
