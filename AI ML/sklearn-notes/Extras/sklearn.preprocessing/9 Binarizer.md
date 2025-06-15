## 🔹 `Binarizer` – Threshold-Based Binarization

### ✅ Purpose:

* Converts numerical values into **binary values (0 or 1)** based on a **threshold**.
* Often used for **feature extraction**, **rule-based models**, or **binary bag-of-words models**.

---

### 📌 Formula:

For a given threshold $t$:

$$
x' =
\begin{cases}
0 & \text{if } x \leq t \\
1 & \text{if } x > t
\end{cases}
$$

---

### 🔧 Basic Example:

```python
from sklearn.preprocessing import Binarizer
import numpy as np

# Sample data
X = np.array([[1], [3], [5], [7]])

# Apply Binarizer with threshold=4
binarizer = Binarizer(threshold=4)
X_bin = binarizer.fit_transform(X)

print(X_bin)
```

---

### 🔍 Output:

```plaintext
[[0]
 [0]
 [1]
 [1]]
```

* Values ≤ 4 → 0
* Values > 4 → 1

---

### 🔄 Multi-Feature Example:

```python
X = np.array([[1, 4, 6],
              [7, 2, 0]])

binarizer = Binarizer(threshold=3)
X_bin = binarizer.fit_transform(X)

print(X_bin)
```

```plaintext
[[0 1 1]
 [1 0 0]]
```

---

### 📌 Use Cases:

* Binarizing **continuous features** for **probabilistic models**.
* Simplifying **numeric features** to check if they exceed certain thresholds.
* Text data after TF or TF-IDF: convert term weights to binary presence/absence.

---

### ⚠️ Notes:

* `Binarizer` is not for **categorical values** — use encoders for that.
* Doesn’t scale values — just threshold-based **binary transformation**.
