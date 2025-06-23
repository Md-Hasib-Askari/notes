## ✅ Topic 19: Building Applications with OpenCV

Time to **deploy OpenCV in real-world apps** using GUI tools, packaging tools, and edge devices like Raspberry Pi & NVIDIA Jetson.

---

### 🔹 1. GUI Apps with OpenCV + Tkinter

OpenCV’s `imshow()` is limited. Use **Tkinter** or **PyQt** for real applications.

#### Example: OpenCV + Tkinter

```python
import cv2
import tkinter as tk
from PIL import Image, ImageTk

cap = cv2.VideoCapture(0)

def update_frame():
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    lbl.imgtk = imgtk
    lbl.configure(image=imgtk)
    lbl.after(10, update_frame)

root = tk.Tk()
lbl = tk.Label(root)
lbl.pack()
update_frame()
root.mainloop()
```

📌 Use PyQt5/PySide2 for advanced UIs.

---

### 🔹 2. Deploy with PyInstaller

Bundle OpenCV apps into a `.exe` or standalone binary:

```bash
pip install pyinstaller

pyinstaller --onefile app.py
```

> Include additional files (like model weights, images) with `--add-data`:

```bash
pyinstaller app.py --add-data "model.onnx:."
```

---

### 🔹 3. Run OpenCV on Raspberry Pi

Steps:

1. Install OpenCV via `apt`, `pip`, or build from source
2. Optimize build with:

   * `-DWITH_TBB=ON`
   * `-DNEON=ON` for ARM acceleration
3. Use light models (e.g., MobileNet, Tiny-YOLO)

📌 Install PiCamera driver or use USB camera with `cv2.VideoCapture(0)`

---

### 🔹 4. Run OpenCV on NVIDIA Jetson (Nano, Xavier, Orin)

Steps:

1. Flash JetPack SDK (includes OpenCV + CUDA + cuDNN)
2. Use Jetson-specific optimizations (e.g., TensorRT)
3. Run OpenCV with CUDA backend:

```python
cv2.ocl.setUseOpenCL(True)
cv2.dnn.readNet(...).setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
```

📌 You can run YOLOv5/YOLOv8 ONNX on Jetson using OpenCV or TensorRT

---

### 🔹 5. Build with Docker (for reproducibility)

Create a `Dockerfile`:

```dockerfile
FROM python:3.10-slim

RUN apt update && apt install -y python3-opencv libgl1-mesa-glx
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt

CMD ["python", "app.py"]
```

Then build and run:

```bash
docker build -t opencv-app .
docker run --rm -it opencv-app
```

---

### 🧪 Mini Exercise

* Build a basic OpenCV + Tkinter camera app
* Package it into an executable with PyInstaller
* Bonus: Run your code on a Raspberry Pi (if available)
