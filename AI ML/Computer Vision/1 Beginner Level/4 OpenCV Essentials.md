## 🧰 Topic 4: **OpenCV Essentials**

### 🎯 Goal:

Get hands-on with OpenCV's core functionalities — reading, displaying, drawing, and handling video/image input.

---

### 📥 1. Reading & Writing Images

```python
import cv2

# Read image (color)
img = cv2.imread('image.jpg')

# Show image
cv2.imshow('My Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save image
cv2.imwrite('output.jpg', img)
```

---

### 🎥 2. Reading Video & Webcam Input

```python
cap = cv2.VideoCapture(0)  # Use 0 for webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('Webcam Feed', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

🔧 You can also read video files:

```python
cap = cv2.VideoCapture('video.mp4')
```

---

### ✏️ 3. Drawing Shapes and Text

Useful for annotations, bounding boxes, etc.

```python
# Draw line
cv2.line(img, (50, 50), (200, 50), (0, 255, 0), 3)

# Rectangle
cv2.rectangle(img, (100, 100), (300, 300), (255, 0, 0), 2)

# Circle
cv2.circle(img, (150, 150), 50, (0, 0, 255), -1)

# Put text
cv2.putText(img, 'Hello', (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
```

---

### 🔄 4. Resize, Rotate, Flip, Crop

```python
resized = cv2.resize(img, (300, 300))
flipped = cv2.flip(img, 1)  # Horizontal flip
cropped = img[100:400, 200:500]
```

---

### ⏱️ 5. Wait & Exit

```python
cv2.waitKey(0)  # Wait indefinitely until a key is pressed
cv2.destroyAllWindows()  # Close all OpenCV windows
```

---

### 🧪 Mini Project Idea:

* Capture webcam video.
* Overlay real-time text and shapes on the frames.
* Save the video using `cv2.VideoWriter`.

---

### 📚 Summary:

* This is your **toolbox** — OpenCV handles reading images/videos, drawing, and displaying data.
* Master these essentials before moving into detection/classification tasks.
