## 🔠 Topic 15: **Optical Character Recognition (OCR)**

### 🎯 Goal:

Extract readable **text** from images — used in scanning documents, license plates, receipts, etc.

---

## 🧠 Core OCR Pipeline:

```
Image → Text Detection → Text Region Cropping → Text Recognition → Final Output
```

---

## 🧰 1. **Tesseract Basics** (Open-Source OCR Engine)

### 🔧 Installation:

```bash
sudo apt install tesseract-ocr
pip install pytesseract
```

### 🧪 Basic Usage:

```python
import pytesseract
import cv2

img = cv2.imread("text_image.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
text = pytesseract.image_to_string(gray)
print(text)
```

### ✅ Supported Languages:

Download language packs (`ben`, `eng`, `ara`, etc.).
You can specify with:

```python
pytesseract.image_to_string(img, lang='ben+eng')
```

---

## 🧩 2. **Text Detection Models** (Tesseract struggles here)

### 🔹 EAST Detector (Efficient and Accurate Scene Text)

Detects rotated and multi-scale text in natural scenes.

```python
net = cv2.dnn.readNet("frozen_east_text_detection.pb")
```

* Input: image
* Output: bounding boxes
* Combine with Tesseract for full OCR

---

## 🧠 3. **Text Recognition Models**

| Model                                             | Use                                  |
| ------------------------------------------------- | ------------------------------------ |
| **Tesseract**                                     | Simple, OCR ready                    |
| **CRNN** (Convolutional Recurrent Neural Network) | Deep learning-based text recognition |
| **TrOCR (by Microsoft)**                          | Transformer-based OCR, high accuracy |
| **PaddleOCR**                                     | Very accurate, end-to-end system     |
| **EasyOCR**                                       | Deep learning-based, easy to use     |

### 🔹 EasyOCR Example:

```python
import easyocr
reader = easyocr.Reader(['en'])
result = reader.readtext('text_image.jpg')
```

---

## ✍️ 4. Handwriting Recognition

| Model                       | Use                                 |
| --------------------------- | ----------------------------------- |
| **IAM Dataset + CRNN/LSTM** | English handwriting                 |
| **BN-HTRd (Bangla)**        | Bangla handwritten recognition      |
| **TrOCR**                   | Can handle both print & handwriting |

---

## 📚 Applications:

| Use Case                  | Examples                       |
| ------------------------- | ------------------------------ |
| Document digitization     | Invoices, forms, scanned books |
| License plate recognition | Traffic monitoring             |
| Handwritten note parsing  | OCR notebooks                  |
| Bank cheques              | Recognize amounts, signatures  |
| Passport/ID scanning      | MRZ recognition                |

---

## 🧪 Mini Project Ideas:

* Build an OCR system for reading electricity bills or ID cards.
* Combine **EAST + Tesseract** to detect and extract text from street photos.
* Try **Bangla handwriting OCR** using pretrained models + fine-tuning.

---

## 📦 Summary:

* Use **Tesseract** for simple, language-rich OCR.
* Use **EAST + Tesseract** or **EasyOCR/PaddleOCR** for natural scene images.
* For handwriting or advanced needs, go with **CRNN** or **transformer-based models** like TrOCR.
