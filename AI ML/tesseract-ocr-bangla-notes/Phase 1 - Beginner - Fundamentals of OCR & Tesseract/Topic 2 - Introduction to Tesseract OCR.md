# 📘 Topic 2: Introduction to Tesseract OCR

## 🔍 What is Tesseract?

**Tesseract** is an open-source OCR engine originally developed by Hewlett-Packard and later maintained by Google. It supports over 100 languages, including complex scripts like **Bengali**.

---

## 📜 History and Background:

| Year | Event                                                               |
| ---- | ------------------------------------------------------------------- |
| 1985 | Originally developed by HP Labs                                     |
| 2005 | Released as open-source by HP and UNLV                              |
| 2006 | Google took over development                                        |
| 2017 | Tesseract 4.0 introduced LSTM-based recognition for better accuracy |

---

## 🧠 How Tesseract Works (Simplified Pipeline):

1. **Image Input**

   * Accepts formats like PNG, JPG, TIFF
2. **Preprocessing**

   * Internal binarization and noise removal
3. **Page Layout Analysis**

   * Detects text blocks, lines, and words
4. **Character Recognition**

   * Uses LSTM (Long Short-Term Memory) models in newer versions
5. **Text Output**

   * Can be plain text, PDF with text layer, or searchable HOCR/ALTO XML

---

## ⚙️ Key Features:

* **Multilingual Support** – 100+ languages including `ben` (Bengali)
* **LSTM-based Models** – High accuracy on printed and cursive text
* **Custom Training Support** – Can be trained on your own dataset
* **Unicode Output** – Handles scripts like Bengali correctly
* **CLI and API** – Available as a command-line tool and as a library in Python (`pytesseract`)

---

## 📦 Supported Output Formats:

* `.txt` – Plain UTF-8 text
* `.pdf` – Searchable PDF with recognized text layer
* `.hocr` – HTML with OCR bounding boxes
* `.tsv` – Tab-separated value table with coordinates

---

## 📚 Language Packs:

Tesseract uses `.traineddata` files located in the `tessdata` directory. For Bengali:

* Language code: `ben`
* Install via package manager or manually download from [Tesseract GitHub](https://github.com/tesseract-ocr/tessdata)

---

## 📌 When to Use Tesseract:

✅ Ideal for:

* Clean, printed documents
* Multilingual OCR tasks
* Projects that require open-source solutions

❌ Not ideal for:

* Highly complex handwritten documents (use ICR or deep learning alternatives)
* Real-time OCR in videos (needs custom optimization)

---

## 🔚 Summary:

Tesseract OCR is a powerful, free tool for converting images to text across a wide range of languages. For Bengali OCR, it provides pre-trained models and supports fine-tuning for more specialized tasks.

---
