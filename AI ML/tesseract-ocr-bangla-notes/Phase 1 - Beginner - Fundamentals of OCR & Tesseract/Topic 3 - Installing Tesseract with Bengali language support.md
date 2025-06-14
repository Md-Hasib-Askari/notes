# 📘 Topic 3: Installing Tesseract (with Bengali language support)

## 🛠️ Installation Overview:

Tesseract can be installed on all major platforms. Bengali support (`ben`) comes from its language data file (`ben.traineddata`), which can be added during or after installation.

---

## 🖥️ 1. **Linux Installation**

### ▶️ Ubuntu/Debian:

```bash
sudo apt update
sudo apt install tesseract-ocr
sudo apt install tesseract-ocr-ben
```

This installs both the core Tesseract engine and the Bengali language pack.

---

## 🪟 2. **Windows Installation**

### ▶️ Steps:

1. Download the installer from:
   [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)

   *(UB Mannheim builds are more frequently updated than official releases.)*

2. During installation:

   * Check “Additional language data”
   * Select **Bengali** or all Indic scripts

3. After installation, set the system PATH:

   ```cmd
   setx PATH "%PATH%;C:\Program Files\Tesseract-OCR"
   ```

4. Verify installation:

   ```cmd
   tesseract --version
   ```

---

## 🍏 3. **macOS Installation**

### ▶️ Using Homebrew:

```bash
brew install tesseract
brew install tesseract-lang  # Optional for all languages
```

To install only Bengali:

```bash
brew install tesseract-lang && \
cp /usr/local/Cellar/tesseract-lang/*/share/tessdata/ben.traineddata /usr/local/share/tessdata/
```

---

## 📁 4. **Manual Bengali Language Pack Installation**

If not installed during setup, you can manually add the Bengali language pack:

### ▶️ Steps:

1. Go to:
   [https://github.com/tesseract-ocr/tessdata](https://github.com/tesseract-ocr/tessdata)

2. Download `ben.traineddata`

3. Place it in your `tessdata` directory:

   * Linux/macOS: `/usr/share/tesseract-ocr/4.00/tessdata/`
   * Windows: `C:\Program Files\Tesseract-OCR\tessdata\`

---

## ✅ 5. Verifying Bengali OCR Support

```bash
tesseract image.png output -l ben
```

If Tesseract runs without an error, `ben` is installed correctly.

---

## 🔚 Summary:

Tesseract is easy to install on any system, and adding Bengali OCR support requires either installing the right language pack (`ben.traineddata`) or enabling it during setup. Once installed, you're ready to start OCR on Bengali text.
