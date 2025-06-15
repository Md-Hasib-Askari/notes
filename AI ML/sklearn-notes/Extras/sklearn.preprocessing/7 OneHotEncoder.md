## 🔹 `OneHotEncoder` – One-Hot Encoding of Features

### ✅ Purpose:

* Converts **categorical feature values** into a **sparse binary matrix**.
* Each category is represented as a separate **binary (0/1)** column.
* Prevents ML models from **assuming ordinal relationships**.

---

### 📌 Example:

```python
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Sample feature (2D array)
X = np.array([['red'], ['green'], ['blue'], ['green']])

encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X)

print(X_encoded.toarray())
```

---

### 🔍 Output:

```plaintext
[[0. 0. 1.]  # red
 [0. 1. 0.]  # green
 [1. 0. 0.]  # blue
 [0. 1. 0.]] # green
```

The columns (by default, sorted alphabetically):

* `'blue'`, `'green'`, `'red'`

---

### 🔄 Decoding:

```python
encoder.inverse_transform([[0, 1, 0]])  # Output: [['green']]
```

---

### 🔧 Multiple Features:

```python
X = np.array([
    ['Male', 'US'],
    ['Female', 'Canada'],
    ['Female', 'US']
])

encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X).toarray()
print(X_encoded)
```

Output will have one-hot encoded columns for **both** features.

---

### 📌 Key Parameters:

* `sparse=False`: returns dense NumPy array instead of sparse matrix
* `drop='first'`: drops the first category to avoid multicollinearity (useful in regression)
* `handle_unknown='ignore'`: avoids error if unseen category appears at transform time

```python
encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse=False)
```

---

### ⚠️ Caution:

* Always apply **`fit` on training data**, then **`transform` test data**.
* One-hot encoding increases **dimensionality** — watch out with high-cardinality features.

---

### ✅ Typical Use in Pipeline:

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

categorical_cols = ['color', 'gender']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_cols)
    ]
)

pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', LogisticRegression())
])
```
