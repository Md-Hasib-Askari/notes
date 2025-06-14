# 📘 Topic 9: Fine-Tuning Tesseract for Better Bengali OCR Accuracy

Bengali (বাংলা) OCR presents unique challenges due to:

* Complex ligatures (যুক্তাক্ষর)
* Multiple diacritics (matras)
* Vertical and horizontal stacking of glyphs
* Variations in fonts and print quality

This topic covers **key strategies to fine-tune Tesseract** for higher Bengali text recognition accuracy.

---

## ✅ 1. Use the Best Language Model

Always start with the **LSTM-based model** from [`tessdata_best`](https://github.com/tesseract-ocr/tessdata_best):

```bash
# Use this traineddata for best results
tesseract image.png output -l ben --oem 1 --psm 6
```

* `--oem 1`: LSTM only
* `--psm 6`: Single block of text (adjust as needed)

---

## ✅ 2. Optimize Preprocessing

Quality preprocessing can **double or triple** recognition accuracy.

**Key Preprocessing Steps:**

* Binarization using **adaptive thresholding**
* **Denoising** using morphological operations
* **Skew correction**
* Resize small text using `cv2.resize()`
* Sharpen text using filters like `cv2.GaussianBlur()` or `cv2.filter2D()`

➡️ Combine with `pytesseract.image_to_string()` using `lang='ben'`

---

## ✅ 3. Experiment with PSM and OEM

Different documents work better with different **page segmentation modes**:

| Scenario            | Suggested PSM |
| ------------------- | ------------- |
| Paragraphs          | `--psm 6`     |
| Receipts            | `--psm 11`    |
| Single line         | `--psm 7`     |
| Multi-column layout | `--psm 1`     |

> Always pair `--psm` with `--oem 1` for LSTM models.

---

## ✅ 4. Remove Non-Text Noise

Remove lines, tables, or graphical elements that can confuse OCR:

* Use OpenCV to detect and remove horizontal/vertical lines
* Mask out logos or stamps using contours or masks

---

## ✅ 5. Apply Custom Whitelist (Optional)

Restrict OCR to only desired characters (e.g., Bengali Unicode range):

```bash
tesseract image.png output -l ben -c tessedit_char_whitelist="অআইঈউঊএঐওঔকখগঘঙচছজঝঞ..."
```

Or do this in Python:

```python
custom_config = r'--psm 6 --oem 1 -c tessedit_char_whitelist="..."'
text = pytesseract.image_to_string(img, lang='ben', config=custom_config)
```

---

## ✅ 6. Use Custom Fonts for Training (Advanced)

If OCR fails for specific fonts or handwriting styles, you can:

* Generate synthetic training data
* Fine-tune the `ben.traineddata` (covered in future topics)

---

## ✅ 7. Correct Common Errors with Post-Processing

After OCR, apply post-processing:

* **Unicode normalization (NFC)**: to standardize Bengali characters
* **Regex-based correction**: fix predictable errors (e.g., replace “জ” misread as “ঙ”)
* **Language models/spell-check**: validate words using a Bengali dictionary

```python
import unicodedata
text = unicodedata.normalize("NFC", text)
```

---

## 🔚 Summary:

Improving Bengali OCR with Tesseract is a mix of:

1. Choosing the best model and modes (ben.traineddata, PSM, OEM)
2. Smart image preprocessing
3. Post-processing cleanup
4. (Optionally) Training with custom fonts

Even without custom training, these steps can push your accuracy well above 90% on clean inputs.
