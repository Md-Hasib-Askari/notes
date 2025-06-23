## ✅ Topic 11: Feature Detection & Matching

This topic helps you **identify key points and descriptors** in images and match them across frames, useful for:

* Object recognition
* Panorama stitching
* Structure-from-motion

---

### 🔹 1. Key Concepts

* **Keypoints**: Points of interest (corners, blobs)
* **Descriptors**: Feature vectors describing each keypoint
* **Matching**: Compare descriptors to find similarities

---

### 🔹 2. ORB (Fast + Free)

ORB is fast and works without extra libraries.

```python
import cv2

img = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)

orb = cv2.ORB_create()
keypoints, descriptors = orb.detectAndCompute(img, None)

img_out = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0), flags=0)
cv2.imshow("ORB Keypoints", img_out)
```

---

### 🔹 3. SIFT (Accurate but needs `opencv-contrib-python`)

```bash
pip install opencv-contrib-python
```

```python
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(img, None)
```

---

### 🔹 4. Feature Matching with BFMatcher

Match features between two images:

```python
img1 = cv2.imread("img1.jpg", 0)
img2 = cv2.imread("img2.jpg", 0)

orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

matched_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)
cv2.imshow("Matches", matched_img)
```

---

### 🔹 5. FLANN Matcher (for large datasets)

For SIFT/SURF (floating-point descriptors):

```python
index_params = dict(algorithm=1, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Apply Lowe's ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)
```

---

### 🔹 6. Homography (Stitch Images)

Estimate transformation to align matched keypoints:

```python
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
```

---

### 🧪 Mini Exercise

* Detect ORB keypoints in two images.
* Match features using BFMatcher.
* Draw top 10 matches.
