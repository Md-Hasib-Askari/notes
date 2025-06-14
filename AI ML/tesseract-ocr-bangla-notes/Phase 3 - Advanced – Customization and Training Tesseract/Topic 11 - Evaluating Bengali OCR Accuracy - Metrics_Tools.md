# 📘 Topic 11: Evaluating Bengali OCR Accuracy (Metrics & Tools)

Evaluating OCR output is essential, especially for Bengali, where ligatures and matras can be easily misrecognized. This topic focuses on **how to measure OCR quality using metrics and tools**.

---

## 🎯 Why Evaluate?

* To **quantify recognition quality**
* To compare preprocessing or Tesseract config options
* To determine if custom training is necessary

---

## 📏 Common OCR Evaluation Metrics

### ✅ 1. **Character Error Rate (CER)**

CER measures the number of incorrect characters:

```
CER = (Insertions + Deletions + Substitutions) / Total characters in ground truth
```

✅ **Preferred for Bengali**, since character complexity is high.

---

### ✅ 2. **Word Error Rate (WER)**

WER compares entire words:

```
WER = (Insertions + Deletions + Substitutions) / Total words in ground truth
```

* Use when word-level accuracy matters (e.g. document indexing)
* Less sensitive than CER for fine ligature-level mistakes

---

## 🧪 Tools for Evaluation

### 1. **jiwer (Python package)**

Great for quick WER/CER evaluations.

```bash
pip install jiwer
```

```python
from jiwer import wer, cer

ground_truth = "বাংলা ভাষা একটি সমৃদ্ধ ভাষা।"
prediction = "বাংল ভাষা একটি সম্রদ্ধ ভাষ।"

print("WER:", wer(ground_truth, prediction))
print("CER:", cer(ground_truth, prediction))
```

---

### 2. **Levenshtein Distance**

For measuring how many edits it takes to turn prediction into the correct string.

```bash
pip install python-Levenshtein
```

```python
import Levenshtein
distance = Levenshtein.distance(prediction, ground_truth)
```

---

### 3. **Custom Script for Batch Evaluation**

If evaluating many images:

```python
from jiwer import cer

with open("gt.txt", "r", encoding="utf-8") as f1, open("ocr_output.txt", "r", encoding="utf-8") as f2:
    gt_text = f1.read()
    ocr_text = f2.read()
    print("CER:", cer(gt_text, ocr_text))
```

---

## 🧠 Tips for Bengali OCR Evaluation

* Normalize Unicode before comparing (`unicodedata.normalize('NFC', text)`)
* Ignore punctuation if your task doesn’t need it
* Align OCR output and ground truth line-by-line for better comparison
* Run evaluations per document type (print, scan, handwritten)

---

## 📊 Example Evaluation Report

| Sample | WER  | CER  |
| ------ | ---- | ---- |
| Doc A  | 0.11 | 0.07 |
| Doc B  | 0.22 | 0.14 |
| Doc C  | 0.05 | 0.03 |

---

## 🔚 Summary

To evaluate Bengali OCR:

* Use **CER** for character-level precision
* Use **WER** for full-text comprehension
* Use `jiwer` and `Levenshtein` for fast metrics
* Normalize text before comparison
