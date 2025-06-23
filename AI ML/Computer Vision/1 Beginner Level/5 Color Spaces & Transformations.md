## 📸 Topic 5: **Color Spaces & Transformations**

### 🎯 Goal:

Understand how to convert between different color spaces and use them for color filtering, segmentation, and analysis.

---

### 🧠 What is a Color Space?

A **color space** is a way to represent color values in an image.

| Color Space   | Description                                                     |
| ------------- | --------------------------------------------------------------- |
| **RGB**       | Red, Green, Blue (default for screens)                          |
| **Grayscale** | Single intensity channel                                        |
| **HSV**       | Hue, Saturation, Value (great for filtering colors)             |
| **YCrCb**     | Luminance (Y) + Chrominance (Cr, Cb), used in compression       |
| **LAB**       | Lightness + color components (better for perceptual similarity) |

---

### 🔄 Color Conversion in OpenCV

```python
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
```

📌 OpenCV uses **BGR** by default, not RGB.

---

### 🎯 HSV in Action: Color Filtering

HSV makes it easier to filter colors like red, blue, green, etc., based on hue ranges.

```python
# Convert to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define blue color range
lower_blue = np.array([100, 150, 0])
upper_blue = np.array([140, 255, 255])

# Mask the blue regions
mask = cv2.inRange(hsv, lower_blue, upper_blue)
result = cv2.bitwise_and(img, img, mask=mask)
```

---

### 🧪 Real-World Use Cases:

| Task                    | Color Space Used  |
| ----------------------- | ----------------- |
| Skin detection          | HSV / YCrCb       |
| Shadow removal          | HSV / LAB         |
| Background subtraction  | Grayscale / YCrCb |
| Traffic light detection | HSV               |

---

### 📸 Visual Example:

Create sliders to dynamically filter colors using HSV with OpenCV’s `cv2.createTrackbar`.

---

### 📚 Summary:

* Different color spaces give different advantages.
* HSV is highly useful for robust **color segmentation**.
* Practice converting and masking images with different color spaces.
