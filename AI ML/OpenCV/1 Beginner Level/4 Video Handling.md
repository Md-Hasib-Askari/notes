## ✅ Topic 4: Video Handling

This section is all about **reading, processing, and writing video** files or webcam streams using OpenCV.

---

### 🔹 1. Reading from a Webcam

```python
import cv2

cap = cv2.VideoCapture(0)  # 0 = default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
```

---

### 🔹 2. Reading from a Video File

```python
cap = cv2.VideoCapture("video.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Video", frame)

    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```

---

### 🔹 3. Writing/Saving a Video

```python
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("output.avi", fourcc, 20.0, (640, 480))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    out.write(frame)  # Save frame
    cv2.imshow("Recording", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
```

---

### 🔹 4. Processing Video Frame-by-Frame

Apply processing inside the loop:

```python
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    cv2.imshow("Edges", edges)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
```

---

### 🧪 Mini Exercise

* Capture live webcam feed.
* Convert each frame to grayscale.
* Save the grayscale video to a file.
