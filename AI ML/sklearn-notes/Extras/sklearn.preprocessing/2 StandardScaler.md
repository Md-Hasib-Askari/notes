## 🔹 `StandardScaler` – Standardization

### ✅ Purpose:

* It standardizes features by removing the **mean** and scaling to **unit variance**.
* After transformation, each feature will have:

  * **mean = 0**
  * **standard deviation = 1**

This is **crucial** for ML models that are sensitive to the scale of data (e.g., SVM, Logistic Regression, KNN, PCA).

---

### 📌 Formula:

For each feature $x$:

<div align="center">
  <img src="https://github.com/user-attachments/assets/456b8b63-d6eb-43e1-81bf-a7378f7262f1" alt="" />
</div>

Where:

* *μ* = mean of feature
* *σ* = standard deviation of feature

---

### 🔧 How to Use:

```python
from sklearn.preprocessing import StandardScaler
import numpy as np

# Example dataset
X = np.array([[1, 2], [3, 6], [5, 10]], dtype=float)

# Create the scaler
scaler = StandardScaler()

# Fit and transform the data
X_scaled = scaler.fit_transform(X)

print(X_scaled)
```

---

### 🔍 Output:

```plaintext
[[-1.22474487 -1.22474487]
 [ 0.          0.        ]
 [ 1.22474487  1.22474487]]
```

Now both features have:

* mean ≈ 0
* std ≈ 1

---

### 📌 Common Use Case in Pipelines:

```python
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

pipeline = make_pipeline(
    StandardScaler(),
    LogisticRegression()
)

pipeline.fit(X_train, y_train)
```

This ensures the data is scaled **before** being fed into the model.

---

### ⚠️ Notes:

* Don’t fit the scaler on the test set!
  Always `fit` on the training set and `transform` both training and test sets:

```python
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```
