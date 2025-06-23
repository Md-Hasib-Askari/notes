## ✅ Topic 6: Contours and Shape Analysis

**Contours** are curves joining all continuous points along a boundary with the same color or intensity. They're crucial for shape detection, counting objects, etc.

---

### 🔹 1. Finding Contours

```python
import cv2

img = cv2.imread("shapes.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on original image
contour_img = img.copy()
cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
```

---

### 🔹 2. Contour Approximation

Reduce the number of points in a contour.

```python
cnt = contours[0]
epsilon = 0.02 * cv2.arcLength(cnt, True)
approx = cv2.approxPolyDP(cnt, epsilon, True)
cv2.drawContours(img, [approx], -1, (255, 0, 0), 2)
```

---

### 🔹 3. Convex Hull

Draw the smallest convex shape that encloses the contour.

```python
hull = cv2.convexHull(cnt)
cv2.drawContours(img, [hull], -1, (0, 0, 255), 2)
```

---

### 🔹 4. Bounding Shapes

#### a. **Bounding Rectangle**

```python
x, y, w, h = cv2.boundingRect(cnt)
cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
```

#### b. **Minimum Area Rectangle**

```python
rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = box.astype(int)
cv2.drawContours(img, [box], 0, (0, 255, 255), 2)
```

#### c. **Enclosing Circle**

```python
(x, y), radius = cv2.minEnclosingCircle(cnt)
center = (int(x), int(y))
radius = int(radius)
cv2.circle(img, center, radius, (255, 255, 0), 2)
```

---

### 🔹 5. Shape Properties

```python
area = cv2.contourArea(cnt)
perimeter = cv2.arcLength(cnt, True)
```

---

### 🧪 Mini Exercise

* Load an image with multiple shapes.
* Detect and draw:

  * All contours
  * Bounding rectangles
  * Convex hulls
