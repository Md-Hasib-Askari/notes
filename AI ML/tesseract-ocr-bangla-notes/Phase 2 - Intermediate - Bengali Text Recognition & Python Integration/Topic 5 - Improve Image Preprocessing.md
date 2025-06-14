# 📘 Topic 5: Improve Image Preprocessing

## 🎯 Why Preprocessing?

OCR accuracy heavily depends on the quality of the input image. Bengali text, especially with complex ligatures and matras, is sensitive to:

* Low resolution
* Noise
* Uneven lighting
* Skew

**Preprocessing** prepares your image to make it “OCR-friendly.”

---

## 🔧 Common Preprocessing Techniques

### 1. **Grayscale Conversion**

Reduces image to shades of gray, making it easier to binarize.

```python
import cv2
img = cv2.imread('bangla.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

---

### 2. **Thresholding (Binarization)**

Converts the grayscale image to pure black & white.

```python
# Otsu's thresholding
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

### Adaptive Threshold (for uneven lighting):

```python
thresh = cv2.adaptiveThreshold(
    gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11, 2
)
```

---

### 3. **Noise Removal**

Use morphological operations to remove small dots and noise.

```python
# Remove noise using morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
```

---

### 4. **Resize (Upscaling)**

OCR engines perform better on larger text sizes.

```python
resized = cv2.resize(clean, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
```

---

### 5. **Skew Correction**

Use image moments or Hough transforms to deskew the image.

```python
# Example with image rotation if angle is known
(h, w) = gray.shape
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle=-1.5, scale=1.0)
deskewed = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR)
```

---

## 🔄 Full Preprocessing Pipeline (Python + OpenCV)

```python
import cv2
from PIL import Image
import pytesseract

img = cv2.imread('bangla.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 11, 2
)
resized = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
cv2.imwrite('processed.png', resized)

text = pytesseract.image_to_string(Image.open('processed.png'), lang='ben')
print(text)
```

---

## 🧠 Tips for Bengali Text

* Use **high-contrast**, black-and-white inputs.
* Avoid touching characters or broken glyphs.
* Dealing with ligatures? Ensure font rendering is not distorted.
* Be extra cautious with low-resolution scans of newspapers or books.

---

## 🔚 Summary:

Image preprocessing is essential for enhancing OCR performance. Steps like grayscale conversion, thresholding, noise removal, and resizing can drastically boost accuracy, especially for complex scripts like Bengali.
