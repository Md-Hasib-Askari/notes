# 📘 Topic 4: First Steps with Tesseract CLI

The command-line interface (CLI) is the simplest and most powerful way to use Tesseract for OCR tasks, especially when starting out.

---

## 🧾 Basic Syntax

```bash
tesseract input_image output_base [options]
```

* `input_image`: Path to the image file (e.g., `text.png`)
* `output_base`: Path to the output file (without extension)
* `[options]`: Additional parameters like language or output format

---

## 🖋️ Example with Bengali Language

```bash
tesseract bangla_text.png output -l ben
```

* `-l ben`: Use Bengali language model (`ben.traineddata`)
* `output.txt`: Tesseract saves output as a `.txt` file by default

---

## 📂 Input Image Formats

* Supported: `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`
* For best results:

  * Use **high-resolution**, **grayscale or black & white** images
  * Avoid blurry or skewed images

---

## 🧪 Output Options

| Format     | Command                         | Description                        |
| ---------- | ------------------------------- | ---------------------------------- |
| Plain Text | `tesseract img out`             | Default format                     |
| PDF        | `tesseract img out -l ben pdf`  | Outputs searchable PDF             |
| HOCR       | `tesseract img out -l ben hocr` | HTML output with OCR layout data   |
| TSV        | `tesseract img out -l ben tsv`  | Tab-separated values with box info |

---

## 🔎 Example Commands

### ▶️ Bengali text as plain text:

```bash
tesseract page.png output -l ben
```

### ▶️ Bengali text to searchable PDF:

```bash
tesseract page.png output -l ben pdf
```

### ▶️ Get OCR with bounding box data (TSV):

```bash
tesseract page.png output -l ben tsv
```

---

## ⚙️ Helpful Options

| Option   | Purpose                                |
| -------- | -------------------------------------- |
| `-l ben` | Specify language (Bengali)             |
| `--psm`  | Page segmentation mode (layout engine) |
| `--oem`  | OCR Engine Mode                        |
| `-c`     | Set config variable                    |

---

### 🔢 Common Page Segmentation Modes (`--psm`)

* `3`: Default, fully automatic page layout
* `6`: Assume a single uniform block of text
* `11`: Sparse text (ideal for scattered text)

```bash
tesseract img.png out -l ben --psm 6
```

---

### 🧠 OCR Engine Modes (`--oem`)

* `0`: Legacy only
* `1`: LSTM only (Tesseract 4+)
* `2`: Legacy + LSTM
* `3`: Default (uses best model available)

---

## 🧼 Pro Tips

* Crop the image before OCR to remove unwanted regions
* Use `--psm 6` for simple one-column documents
* Use `--psm 11` for receipts or scattered words

---

## 🔚 Summary:

The Tesseract CLI provides a flexible and quick way to run OCR on images with just a few commands. By specifying the `ben` language model and using proper `psm`/`oem` values, you can significantly improve Bengali OCR accuracy.
