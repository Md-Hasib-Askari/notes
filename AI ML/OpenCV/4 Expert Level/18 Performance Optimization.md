## ✅ Topic 18: Performance Optimization

When working with real-time systems (e.g., cameras, DL inference, robotics), speed is everything. Here’s how to **optimize OpenCV performance**:

---

### 🔹 1. Multi-threaded Video Capture

OpenCV’s `VideoCapture` is **blocking** by default. Use threading to **read frames asynchronously**:

```python
import cv2
from threading import Thread

class VideoStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = self.cap.read()
        self.running = True
        Thread(target=self.update, daemon=True).start()

    def update(self):
        while self.running:
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.frame

    def stop(self):
        self.running = False
        self.cap.release()

stream = VideoStream()
while True:
    frame = stream.read()
    cv2.imshow("Threaded Camera", frame)
    if cv2.waitKey(1) == ord('q'):
        stream.stop()
        break
```

📌 Boosts FPS by \~30–60% on many systems

---

### 🔹 2. Enable OpenCL Acceleration (T-API)

OpenCV can use OpenCL **GPU acceleration** automatically.

```python
cv2.ocl.setUseOpenCL(True)
print("OpenCL Enabled:", cv2.ocl.useOpenCL())
```

Use `UMat` instead of `ndarray`:

```python
img = cv2.imread("image.jpg")
uimg = cv2.UMat(img)

blurred = cv2.GaussianBlur(uimg, (15, 15), 0)
cv2.imshow("Blurred (GPU)", blurred)
```

---

### 🔹 3. Memory-Efficient Pipelines

Avoid copies and reuse buffers:

```python
gray = np.empty_like(img[:, :, 0])
cv2.cvtColor(img, cv2.COLOR_BGR2GRAY, dst=gray)  # No extra allocation
```

Avoid unnecessary conversions:

```python
# Instead of converting BGR→RGB→PIL→Tensor each time,
# handle it once, vectorized, and in-place if possible
```

---

### 🔹 4. Compile OpenCV with Optimizations (CMake Flags)

For advanced setups:

```bash
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D WITH_OPENCL=ON \
      -D WITH_TBB=ON \
      -D ENABLE_FAST_MATH=1 \
      -D BUILD_opencv_python3=ON ..
```

* `WITH_TBB` = threading backend
* `ENABLE_FAST_MATH` = optimized math libs

---

### 🔹 5. Use `cv2.dnn` Efficiently

For DNN inference on GPU:

```python
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
```

Or for CPU multithreading:

```python
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
```

---

### 🧪 Mini Exercise

* Capture video using the threaded class
* Apply a Gaussian blur using `UMat`
* Check FPS improvements with and without threading
