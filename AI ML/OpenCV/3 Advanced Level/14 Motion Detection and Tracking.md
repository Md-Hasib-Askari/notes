## ✅ Topic 14: Motion Detection and Tracking

Motion detection helps identify **moving objects** in video, while tracking follows them across frames.

---

### 🔹 1. Optical Flow

Estimates **motion vectors** of pixels between frames.

#### a. **Dense Optical Flow (Farneback)**

Tracks **every pixel's motion**.

```python
import cv2
import numpy as np

cap = cv2.VideoCapture("video.mp4")
ret, frame1 = cap.read()
prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

hsv_mask = np.zeros_like(frame1)
hsv_mask[..., 1] = 255  # Full saturation for color

while True:
    ret, frame2 = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                         0.5, 3, 15, 3, 5, 1.2, 0)

    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv_mask[..., 0] = angle * 180 / np.pi / 2
    hsv_mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    motion = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)
    cv2.imshow("Dense Optical Flow", motion)

    prev_gray = gray

    if cv2.waitKey(30) & 0xFF == ord("q"):
        break
```

---

#### b. **Sparse Optical Flow (Lucas-Kanade)**

Tracks specific **corner/keypoints**.

```python
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7)
lk_params = dict(winSize=(15, 15), maxLevel=2)

prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

next_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)
```

---

### 🔹 2. Background Subtraction (Simple Motion Detection)

```python
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    cv2.imshow("Motion Mask", fgmask)
```

---

### 🔹 3. Object Tracking APIs (cv2.Tracker)

Track moving objects using bounding boxes.

```python
tracker = cv2.TrackerCSRT_create()  # or KCF, MIL, MOSSE, etc.
bbox = cv2.selectROI("Frame", frame, False)
tracker.init(frame, bbox)

while True:
    ret, frame = cap.read()
    success, box = tracker.update(frame)
    if success:
        x, y, w, h = map(int, box)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Tracking", frame)
```

📌 Use `cv2.legacy.TrackerCSRT_create()` for newer OpenCV versions

---

### 🧪 Mini Exercise

* Use MOG2 to detect motion from your webcam
* Track a selected object in a video using `cv2.TrackerCSRT`
