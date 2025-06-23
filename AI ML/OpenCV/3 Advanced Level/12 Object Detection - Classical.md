## ✅ Topic 12: Object Detection (Classical Methods)

Before deep learning took over, classical techniques were widely used for detecting and localizing objects. They still have relevance in lightweight or real-time systems.

---

### 🔹 1. Template Matching

Slides a small image (template) over a larger one to find matches.

```python
import cv2
import numpy as np

img = cv2.imread("scene.jpg")
template = cv2.imread("template.jpg")
w, h = template.shape[1], template.shape[0]

result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)

cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
cv2.imshow("Matched", img)
```

📌 Use cases: Logo detection, exact shape finding

---

### 🔹 2. HOG (Histogram of Oriented Gradients) + SVM

Used for human and pedestrian detection.

```python
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

img = cv2.imread("people.jpg")
boxes, weights = hog.detectMultiScale(img, winStride=(8,8))

for (x, y, w, h) in boxes:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("HOG Detection", img)
```

📌 Use cases: Pedestrian detection in surveillance and automotive

---

### 🔹 3. Background Subtraction (Motion Detection)

Useful in static camera setups to detect moving objects.

```python
cap = cv2.VideoCapture("video.mp4")
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    mask = fgbg.apply(frame)
    cv2.imshow("Foreground", mask)

    if cv2.waitKey(30) & 0xFF == ord("q"):
        break
```

📌 Use cases: Anomaly detection, tracking, surveillance

---

### 🔹 4. Contour-Based Object Detection

Combine background subtraction or thresholding with contour detection:

```python
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    if cv2.contourArea(cnt) > 500:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
```

---

### 🧪 Mini Exercise

* Load a static image and a small template, then locate it using `matchTemplate`
* Try human detection using `HOGDescriptor` on a street scene
