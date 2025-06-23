## ✅ Topic 1: Introduction to OpenCV

### 🔹 What is OpenCV?

OpenCV (Open Source Computer Vision Library) is an open-source library of programming functions mainly aimed at real-time computer vision and image processing.

### 🔹 Installation

Use pip to install OpenCV:

```bash
pip install opencv-python
# Optional: Headless version (no GUI support)
pip install opencv-python-headless
```

---

### 🔹 Basic Image Operations

#### 1. **Reading an image**

```python
import cv2

img = cv2.imread("path/to/image.jpg")
cv2.imshow("Image", img)
cv2.waitKey(0)  # Wait until any key is pressed
cv2.destroyAllWindows()
```

#### 2. **Writing/Saving an image**

```python
cv2.imwrite("output.jpg", img)
```

#### 3. **Resizing an image**

```python
resized = cv2.resize(img, (300, 300))  # Width x Height
```

#### 4. **Cropping an image**

```python
cropped = img[50:200, 100:300]  # img[y1:y2, x1:x2]
```

#### 5. **Rotating and flipping**

```python
rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
flipped = cv2.flip(img, 1)  # 0: vertical, 1: horizontal
```

#### 6. **Drawing shapes and text**

```python
# Draw a rectangle
cv2.rectangle(img, (50, 50), (200, 200), (0, 255, 0), 3)

# Draw a circle
cv2.circle(img, (150, 150), 40, (255, 0, 0), -1)

# Draw text
cv2.putText(img, "OpenCV", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
```

---

### 🧪 Mini Exercise

Load an image, resize it to 300x300, draw a red rectangle in the center, and save it.
