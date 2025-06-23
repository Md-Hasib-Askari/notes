## 🧠 Topic 3: **Image Features & Descriptors**

### 🎯 Goal:

Understand how to detect and describe key points in an image so they can be matched or tracked — the foundation for tasks like object detection, recognition, and SLAM.

---

### 🔍 What Are Features?

* **Features** are distinctive patterns in an image (corners, blobs, edges).
* Useful for **matching**, **tracking**, and **recognition**.

---

### 🛠️ Common Feature Detectors & Descriptors:

| Method            | Type                                    | Description                                               |
| ----------------- | --------------------------------------- | --------------------------------------------------------- |
| **Harris Corner** | Detector                                | Detects corners using intensity change                    |
| **SIFT**          | Detector + Descriptor                   | Scale-Invariant Feature Transform (accurate but patented) |
| **SURF**          | Same as SIFT but faster (also patented) |                                                           |
| **ORB**           | Fast + Free                             | Good alternative to SIFT/SURF (OpenCV-friendly)           |

---

### 🧪 Example: ORB with OpenCV

```python
import cv2

img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create()
keypoints, descriptors = orb.detectAndCompute(gray, None)

output = cv2.drawKeypoints(img, keypoints, None, color=(0,255,0))
cv2.imshow("ORB Features", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

### 🔁 Feature Matching

Once features are detected, you can match them between two images.

**Steps:**

1. Detect keypoints and descriptors in both images.
2. Match descriptors using `cv2.BFMatcher` or `FLANN`.
3. Filter good matches (e.g., Lowe's ratio test).
4. Use matches for homography, stitching, etc.

```python
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(desc1, desc2)
matches = sorted(matches, key=lambda x: x.distance)

matched_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None)
```

---

### 📦 Feature Descriptors vs Detectors:

| Detector   | Finds key points (locations)                           |
| ---------- | ------------------------------------------------------ |
| Descriptor | Converts keypoints to numerical vectors for comparison |

---

### 🔧 Use Cases:

* **Panorama stitching**: Match features across overlapping photos.
* **Object recognition**: Identify an object in different scenes.
* **Augmented Reality**: Anchor virtual objects to real-world markers.

---

### 📚 Summary:

* Learn how to detect and describe **keypoints** in images.
* Start with ORB for practical use.
* Matching features is the basis of **image alignment**, **tracking**, and **recognition**.
