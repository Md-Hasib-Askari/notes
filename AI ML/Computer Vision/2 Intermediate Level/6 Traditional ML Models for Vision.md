## 📊 Topic 6: **Traditional Machine Learning Models for Vision**

### 🎯 Goal:

Use classic ML models like KNN, SVM, Decision Trees for simple image classification tasks using **feature vectors**, not deep learning.

---

### 🧠 Concept: How ML Sees Images

Machine Learning models **don't handle images directly**. You need to convert an image to a **feature vector**:

> Image → Features (flattened pixel values or descriptors) → Classifier → Label

---

### 📦 Approaches:

#### 🔹 1. **Flatten Pixels (Naive)**

Convert image to a long 1D vector:

```python
flattened = image.reshape(-1)
```

But this ignores spatial structure.

---

#### 🔹 2. **Feature Extraction + ML**

Use methods like:

* **HOG (Histogram of Oriented Gradients)**
* **Color histograms**
* **PCA** for dimensionality reduction

Then feed features into classifiers like:

* K-Nearest Neighbors (KNN)
* Support Vector Machines (SVM)
* Decision Trees / Random Forests

---

### 🧪 Example Pipeline: HOG + SVM (Sklearn + OpenCV)

```python
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import cv2
import os
import numpy as np

# Load images and labels
X, y = [], []
for label in ['cat', 'dog']:
    for img_file in os.listdir(f'dataset/{label}'):
        img = cv2.imread(f'dataset/{label}/{img_file}', cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64, 64))
        features = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        X.append(features)
        y.append(label)

# Train SVM
X_train, X_test, y_train, y_test = train_test_split(X, y)
model = LinearSVC()
model.fit(X_train, y_train)
print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))
```

---

### 🤔 Why use Traditional ML?

| Pros                     | Cons                          |
| ------------------------ | ----------------------------- |
| Fast, works on CPU       | Struggles with complex images |
| Easy to train            | Needs handcrafted features    |
| Great for small datasets | Limited scalability           |

---

### 🧪 Mini Project Idea:

* Create a binary classifier (e.g., cats vs dogs) using HOG + SVM.
* Add visualization: show HOG image and prediction.

---

### 📚 Summary:

* Traditional ML uses **flattened images** or **engineered features**.
* Combine OpenCV (for image handling) with Scikit-learn (for models).
* This is a **stepping stone** to deep learning.
