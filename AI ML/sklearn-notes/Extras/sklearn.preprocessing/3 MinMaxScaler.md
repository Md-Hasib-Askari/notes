## 🔹 `MinMaxScaler` – Min-Max Normalization

### ✅ Purpose:

* Scales features to a **fixed range**, usually **\[0, 1]**.
* Maintains the **shape** of the original distribution.
* Useful for algorithms like **KNN**, **Neural Networks**, or when **bounded input** is needed.

---

### 📌 Formula:

<div align="center">
  <img src="https://github.com/user-attachments/assets/98b23f6f-b829-41e0-8280-63853b78cd2f" alt="" />
</div>

Where:

* $x$ = original value
* $x_{\text{min}}$, $x_{\text{max}}$ = min & max values of that feature

You can also customize the target range: `[min, max]`.

---

### 🔧 How to Use:

```python
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Sample data
X = np.array([[1, 2], [3, 6], [5, 10]], dtype=float)

# Create scaler
scaler = MinMaxScaler()

# Fit and transform
X_scaled = scaler.fit_transform(X)

print(X_scaled)
```

---

### 🔍 Output:

```plaintext
[[0.  0. ]
 [0.5 0.5]
 [1.  1. ]]
```

Each feature is scaled **independently** to \[0, 1].

---

### 🔄 Custom Feature Range

```python
scaler = MinMaxScaler(feature_range=(-1, 1))
X_scaled = scaler.fit_transform(X)
```

Now values will range between -1 and 1.

---

### 📌 When to Use:

* When **preserving sparsity** and **distribution shape** is important.
* When input features have **known bounds** (e.g., pixel values 0–255).

---

### ⚠️ Notes:

* Like `StandardScaler`, always `fit` on training data and then `transform` test data.
* **Sensitive to outliers**: it stretches the range if outliers are present.

