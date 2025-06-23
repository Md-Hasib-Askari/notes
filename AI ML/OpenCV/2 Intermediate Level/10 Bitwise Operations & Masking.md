## ✅ Topic 10: Bitwise Operations & Masking

**Bitwise operations** allow you to manipulate specific parts of an image using masks. They're essential for background removal, compositing, and working with regions of interest (ROI).

---

### 🔹 1. Bitwise Operations in OpenCV

These operations apply logical operations between two images (or an image and a mask):

```python
import cv2
import numpy as np

img1 = cv2.imread("img1.jpg")
img2 = cv2.imread("img2.jpg")
```

#### a. **Bitwise AND**

```python
bit_and = cv2.bitwise_and(img1, img2)
```

#### b. **Bitwise OR**

```python
bit_or = cv2.bitwise_or(img1, img2)
```

#### c. **Bitwise XOR**

```python
bit_xor = cv2.bitwise_xor(img1, img2)
```

#### d. **Bitwise NOT**

```python
bit_not = cv2.bitwise_not(img1)
```

---

### 🔹 2. Creating a Mask

A **mask** is a grayscale image where white (255) represents the area of interest.

```python
img = cv2.imread("image.jpg")
mask = np.zeros(img.shape[:2], dtype="uint8")  # Black mask

# Draw a white circle on the mask
cv2.circle(mask, (150, 150), 100, 255, -1)
```

---

### 🔹 3. Applying a Mask

Use bitwise operations to extract regions defined by the mask:

```python
masked_img = cv2.bitwise_and(img, img, mask=mask)
```

---

### 🔹 4. Isolating ROI from Complex Background

```python
# Convert to HSV and create mask for blue regions
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_blue = np.array([100, 150, 0])
upper_blue = np.array([140, 255, 255])
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Apply mask
blue_regions = cv2.bitwise_and(img, img, mask=mask)
```

---

### 🧪 Mini Exercise

* Create a black mask with a white rectangle or circle.
* Use it to extract that shape from the image using `bitwise_and`.
* Try `bitwise_not` on the image and observe the results.
