## ✅ Topic 2: Color Spaces and Channels

### 🔹 What is a Color Space?

A **color space** is a specific organization of colors. OpenCV loads images in **BGR** by default, not RGB.

---

### 🔹 Converting Between Color Spaces

#### 1. **BGR to Grayscale**

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

#### 2. **BGR to RGB**

```python
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```

#### 3. **BGR to HSV**

```python
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
```

#### 4. **HSV to BGR**

```python
bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
```

#### 5. **BGR to LAB**

```python
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
```

---

### 🔹 Splitting and Merging Channels

#### 1. **Split BGR channels**

```python
b, g, r = cv2.split(img)
```

#### 2. **View individual channels**

```python
cv2.imshow("Blue", b)
cv2.imshow("Green", g)
cv2.imshow("Red", r)
```

#### 3. **Merge back to BGR**

```python
merged = cv2.merge([b, g, r])
```

---

### 🔹 Example: Isolate Red Color Using HSV

```python
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define red color range in HSV
lower_red = (160, 100, 100)
upper_red = (179, 255, 255)

# Create mask
mask = cv2.inRange(hsv, lower_red, upper_red)

# Apply mask to original image
red_only = cv2.bitwise_and(img, img, mask=mask)

cv2.imshow("Red Only", red_only)
```

---

### 🧪 Mini Exercise

* Convert an image to grayscale and HSV
* Split the BGR channels and display each one
* Use `inRange()` to extract blue color from an image
