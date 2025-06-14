# 📘 Topic 12: Training Tesseract on Custom Bengali Fonts or Handwriting

Tesseract’s default Bengali model (`ben.traineddata`) works well for printed text, but **custom training is essential** when dealing with:

* Uncommon Bengali fonts
* Historical documents
* Scanned books
* Handwriting samples
* Low-quality print images

This topic walks through the **full custom training pipeline** using **Tesseract 4+ (LSTM)**.

---

## ⚙️ Prerequisites

* Tesseract built from source with training tools (≥ 4.0)
* Python, ImageMagick, and training fonts installed
* Bengali Unicode knowledge (to prepare `unicharset`)
* Ubuntu/Debian Linux preferred (training is CLI-heavy)

---

## 📁 Directory Setup

```bash
custom-ocr/
├── data/
│   ├── ground-truth/
│   │   ├── font1.exp0.tif
│   │   ├── font1.exp0.gt.txt
│   ├── langdata/
│   │   ├── ben/
│   └── tessdata/
```

---

## 🧾 Step-by-Step Training Pipeline

---

### ✅ Step 1: Generate Training Images

Use `text2image` to generate `.tif` image and `.box` file:

```bash
text2image \
  --text=ben.training_text.txt \
  --outputbase=font1.exp0 \
  --font="Likhan" \
  --ptsize 32 \
  --writing_mode=horizontal \
  --language=ben
```

Make sure Bengali font is installed and rendering properly.

---

### ✅ Step 2: Create Ground Truth `.gt.txt` Files

Each `.tif` must be accompanied by a `.gt.txt`:

```text
আমার সোনার বাংলা আমি তোমায় ভালোবাসি
```

Name must match the image:
📄 `font1.exp0.tif` → 📄 `font1.exp0.gt.txt`

---

### ✅ Step 3: Extract `.lstmf` Files

```bash
tesseract font1.exp0.tif font1.exp0 --psm 6 lstm.train
```

This creates `font1.exp0.lstmf`, used in LSTM training.

---

### ✅ Step 4: Download Base Model

Use the base Bengali model as a starting point:

```bash
wget https://github.com/tesseract-ocr/tessdata_best/raw/main/ben.traineddata
```

Extract the LSTM model:

```bash
combine_tessdata -e ben.traineddata ben.lstm
```

---

### ✅ Step 5: Start LSTM Training

```bash
lstmtraining \
  --model_output output/ben_custom \
  --continue_from ben.lstm \
  --traineddata tessdata/ben/ben.traineddata \
  --train_listfile ben.training_files.txt \
  --max_iterations 5000
```

* `train_listfile` contains paths to `.lstmf` files
* You can stop training when CER stops improving

---

### ✅ Step 6: Finalize Trained Model

Combine everything into a `.traineddata` file:

```bash
lstmtraining \
  --stop_training \
  --continue_from output/ben_custom_checkpoint \
  --traineddata tessdata/ben/ben.traineddata \
  --model_output output/ben.traineddata
```

---

### ✅ Step 7: Test the Model

```bash
tesseract test_image.png output --tessdata-dir output/ -l ben
```

---

## 💡 Tips

* Use diverse fonts and sizes for better generalization
* Normalize Unicode using NFC in training text
* Evaluate model with `cer()` after each 1000 iterations

---

## 📊 When Should You Train?

| Scenario                      | Retrain?                  |
| ----------------------------- | ------------------------- |
| Poor results with known fonts | ✅ Yes                     |
| Handwritten OCR               | ✅ Yes                     |
| Non-standard layouts only     | ❌ Maybe not               |
| Scan quality low              | ❌ Try preprocessing first |

---

## 🔚 Summary

Training Tesseract on Bengali fonts or handwriting unlocks much better OCR accuracy for non-standard inputs. It requires:

* Generating `.tif` and `.gt.txt` pairs
* Extracting `.lstmf`
* Fine-tuning with `lstmtraining`
* Producing a new `.traineddata` file
