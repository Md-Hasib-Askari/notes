# 📘 Topic 13: Integrating Bengali OCR into Applications (APIs & GUI)

Once you’ve built and fine-tuned your OCR pipeline, the next step is **deploying it**—so others can use it easily. You can integrate your Bengali OCR engine into:

* Backend APIs (Flask, FastAPI, Django)
* Desktop GUIs (Tkinter, PyQt)
* Web frontends
* Mobile apps (via REST APIs)

---

## ✅ 1. **API-Based Integration (Recommended for Scalability)**

You can build a REST API to receive images and return Bengali OCR text using:

### 📦 Flask Example

```bash
pip install flask pytesseract opencv-python
```

```python
from flask import Flask, request, jsonify
import pytesseract, cv2
import numpy as np

app = Flask(__name__)

@app.route('/ocr', methods=['POST'])
def ocr():
    file = request.files['image']
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    text = pytesseract.image_to_string(img, lang='ben', config='--psm 6 --oem 1')
    return jsonify({"text": text})

app.run(debug=True)
```

#### 🔄 Usage:

Send a POST request with the image:

```bash
curl -X POST -F image=@sample.png http://localhost:5000/ocr
```

---

## ✅ 2. **GUI-Based Integration (For Desktop Apps)**

Use `Tkinter` or `PyQt` to build a drag-and-drop GUI.

### 🧩 Tkinter + Tesseract Example

```python
import tkinter as tk
from tkinter import filedialog
from PIL import Image
import pytesseract

def select_image():
    path = filedialog.askopenfilename()
    if path:
        img = Image.open(path)
        text = pytesseract.image_to_string(img, lang='ben')
        text_box.delete("1.0", tk.END)
        text_box.insert(tk.END, text)

root = tk.Tk()
root.title("Bengali OCR")

btn = tk.Button(root, text="Select Image", command=select_image)
btn.pack()

text_box = tk.Text(root, wrap=tk.WORD)
text_box.pack(expand=True, fill=tk.BOTH)

root.mainloop()
```

---

## ✅ 3. **Command-Line Interface (CLI)**

Use `argparse` to build a CLI tool:

```python
import argparse, pytesseract, cv2

parser = argparse.ArgumentParser()
parser.add_argument("image")
args = parser.parse_args()

img = cv2.imread(args.image)
text = pytesseract.image_to_string(img, lang='ben', config='--psm 6 --oem 1')
print(text)
```

Run:

```bash
python ocr_cli.py bangla_text.png
```

---

## ✅ 4. **Web App (with Frontend Upload)**

* Backend: Flask or FastAPI
* Frontend: HTML + JavaScript form
* Upload image → Server runs Bengali OCR → Return JSON/text

---

## ✅ 5. **Mobile Integration**

* Backend API (Flask/FastAPI) handles OCR
* Mobile app (Flutter, React Native, Android) captures images and calls API

➡️ This is ideal for **on-the-go document scanning apps**.

---

## 🔐 Tips for Production

* Limit image size and format for uploads
* Sanitize file input
* Use job queues (e.g. Celery) for processing large files
* Cache previously OCR’d images using a hash
* Store extracted text in a database if needed

---

## 🧠 Use Cases

| Use Case                | Integration Type      |
| ----------------------- | --------------------- |
| Personal OCR tool       | Desktop GUI (Tkinter) |
| Enterprise document OCR | Flask API + database  |
| Mobile scanning app     | API + mobile frontend |
| Bulk processing         | CLI + batch script    |

---

## 🔚 Summary

You can deploy your Bengali OCR pipeline in many forms:

* 🔁 Flask/FastAPI REST APIs for scalable backends
* 🖥️ GUI tools for user-friendly desktop applications
* 📱 Mobile apps via backend integration
* 🖥️ CLI for scripts and automation
