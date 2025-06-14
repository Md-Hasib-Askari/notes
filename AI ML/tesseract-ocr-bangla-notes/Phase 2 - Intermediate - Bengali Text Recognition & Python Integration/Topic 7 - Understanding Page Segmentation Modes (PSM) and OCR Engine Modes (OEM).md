# 📘 Topic 7: Understanding Page Segmentation Modes (PSM) and OCR Engine Modes (OEM)

These two flags (`--psm` and `--oem`) control how Tesseract:

* Interprets the **layout** of the image (PSM)
* Chooses the **recognition engine** (OEM)

---

## 🔢 Page Segmentation Modes (`--psm`)

Page Segmentation Mode tells Tesseract **how the text is arranged** in the image.

| Mode | Description                                                     |
| ---- | --------------------------------------------------------------- |
| 0    | Orientation and script detection (OSD) only                     |
| 1    | Automatic page segmentation with OSD                            |
| 2    | Automatic page segmentation, no OSD                             |
| 3    | Fully automatic page segmentation, no OSD (Default)             |
| 4    | Assume a single column of text                                  |
| 5    | Assume a single uniform block of vertically aligned text        |
| 6    | Assume a single uniform block of text (useful for text regions) |
| 7    | Treat the image as a single line                                |
| 8    | Treat the image as a single word                                |
| 9    | Treat the image as a single word in a circle                    |
| 10   | Treat the image as a single character                           |
| 11   | Sparse text                                                     |
| 12   | Sparse text with OSD                                            |
| 13   | Raw line (no layout detection)                                  |

### ✅ Common Usage:

* `--psm 6` — Good for scanned paragraphs (default for many OCR tasks)
* `--psm 11` — Good for receipts, scattered lines
* `--psm 7` — When you know it's one line (e.g. CAPTCHA)
* `--psm 3` — Default mode for automatic layout detection

---

### 🧪 Example Command:

```bash
tesseract input.png output -l ben --psm 6
```

---

## 🧠 OCR Engine Modes (`--oem`)

OCR Engine Mode tells Tesseract **which engine to use** for character recognition.

| Mode | Description                                          |
| ---- | ---------------------------------------------------- |
| 0    | Legacy engine only                                   |
| 1    | LSTM neural nets only (best accuracy for most tasks) |
| 2    | Legacy + LSTM (combination)                          |
| 3    | Default (choose best available)                      |

> ✅ **Use `--oem 1`** with `--psm 6` for best Bengali results using LSTM models.

---

### 🧪 Example with OEM and PSM:

```bash
tesseract input.png output -l ben --oem 1 --psm 6
```

---

## 📘 Python Equivalent

```python
custom_config = r'--oem 1 --psm 6'
text = pytesseract.image_to_string(image, lang='ben', config=custom_config)
```

---

## 🧠 Tips for Bengali OCR:

* Complex ligatures and matras benefit from **LSTM-only mode** (`--oem 1`)
* Use **`--psm 6`** for clean paragraphs
* Use **`--psm 11`** for multi-line forms and noisy documents

---

## 🔚 Summary:

Choosing the right `--psm` and `--oem` modes can drastically improve OCR accuracy. For Bengali text:

* Use `--oem 1` (LSTM)
* Tune `--psm` based on layout type
