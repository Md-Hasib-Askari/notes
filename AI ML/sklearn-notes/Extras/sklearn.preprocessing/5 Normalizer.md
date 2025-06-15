## 🔹 `Normalizer` – Sample-Wise Normalization

### ✅ Purpose:

* Scales **each row** (sample) to have **unit norm** (length = 1).
* Often used in **text classification**, **clustering**, or **distance-based algorithms**.

---

### 📌 How It Works:

Unlike `StandardScaler` or `MinMaxScaler`, which scale **features (columns)**, `Normalizer` scales each **sample (row)** individually.

### 📌 Formula:

For a sample vector $\mathbf{x} = [x_1, x_2, ..., x_n]$:

$$
\mathbf{x}' = \frac{\mathbf{x}}{||\mathbf{x}||}
$$

Where $||\mathbf{x}||$ can be:

* **L2 norm** (default): $\sqrt{x_1^2 + x_2^2 + \dots + x_n^2}$
* **L1 norm**: $|x_1| + |x_2| + \dots + |x_n|$

---

### 🔧 How to Use:

```python
from sklearn.preprocessing import Normalizer
import numpy as np

# Example data
X = np.array([[4, 3], [1, 2], [10, 0]])

# Create and apply the normalizer (L2 by default)
normalizer = Normalizer()
X_normalized = normalizer.fit_transform(X)

print(X_normalized)
```

---

### 🔍 Output (approx):

```plaintext
[[0.8        0.6       ]
 [0.4472136  0.89442719]
 [1.         0.        ]]
```

Each row now has unit length:

$$
\sqrt{x_1^2 + x_2^2} = 1
$$

---

### 🔄 L1 Normalization:

```python
normalizer = Normalizer(norm='l1')
X_normalized = normalizer.fit_transform(X)
```

Each row’s absolute values sum to 1.

---

### 📌 When to Use:

* **Cosine similarity**-based models (e.g., text mining, clustering).
* When **direction** of the vector matters more than its **magnitude**.
* Common in NLP pipelines (with TF-IDF, bag-of-words).

---

### ⚠️ Notes:

* Operates **row-wise**, not column-wise.
* Doesn’t affect feature distributions — just rescales vector **length**.
* Not interchangeable with `StandardScaler` or `MinMaxScaler`.
