# 📘 Topic 6: Python Integration with `pytesseract`

## 🧩 What is `pytesseract`?

**`pytesseract`** is a Python wrapper for the Tesseract OCR engine. It allows you to use Tesseract directly from Python scripts, enabling you to:

* Automate OCR tasks
* Combine OCR with image processing (e.g., using OpenCV)
* Work with Bengali and other languages easily

---

## 🧰 Installation

```bash
pip install pytesseract
```

Also ensure:

* Tesseract is installed on your system (CLI must work)
* The `ben.traineddata` file for Bengali is in your `tessdata` directory

---

## 🔗 Link Tesseract Executable (Windows Only)

If `pytesseract` can't find Tesseract, set the path manually:

```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

---

## 🧪 Basic OCR with `pytesseract`

```python
from PIL import Image
import pytesseract

img = Image.open("bangla_sample.png")
text = pytesseract.image_to_string(img, lang='ben')
print(text)
```

* `lang='ben'`: Specifies Bengali language
* Returns UTF-8 encoded string

---

## 🖼️ OCR with OpenCV Preprocessing

```python
import cv2
import pytesseract

img = cv2.imread("bangla_sample.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

text = pytesseract.image_to_string(thresh, lang='ben')
print(text)
```

---

## 📦 Output in Other Formats

### Get bounding box data (dictionary form):

```python
from pytesseract import image_to_data

data = pytesseract.image_to_data(img, lang='ben', output_type=pytesseract.Output.DICT)
print(data['text'])  # List of words
```

### Get TSV format:

```python
tsv = pytesseract.image_to_data(img, lang='ben', output_type=pytesseract.Output.STRING)
```

---

## ⚙️ Custom OCR Settings

You can specify `--psm`, `--oem`, or config variables:

```python
custom_config = r'--oem 1 --psm 6'
text = pytesseract.image_to_string(img, lang='ben', config=custom_config)
```

---

## 📂 Save Output to File

```python
with open("output.txt", "w", encoding="utf-8") as f:
    f.write(text)
```

---

## 🧠 Tips

* Combine OpenCV + `pytesseract` for best results
* Use `pytesseract.image_to_boxes()` to get character-level bounding boxes
* Save processed images for debugging OCR errors

---

## 🔚 Summary:

`pytesseract` is the bridge between Tesseract OCR and Python, allowing seamless integration for document analysis, automation, and preprocessing workflows—especially useful when working with Bengali text and custom pipelines.
