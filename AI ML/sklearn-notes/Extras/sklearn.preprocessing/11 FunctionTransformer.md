## 🔹 `FunctionTransformer` – Custom Preprocessing with Any Function

### ✅ Purpose:

* Allows you to wrap **any custom function** into a transformer compatible with Scikit-learn pipelines.
* Ideal for **applying NumPy functions**, **custom scaling**, **log transforms**, etc.

---

### 📌 Syntax:

```python
from sklearn.preprocessing import FunctionTransformer

transformer = FunctionTransformer(func=None, inverse_func=None, validate=True)
```

---

### 🔧 Example 1: Log Transform

```python
import numpy as np
from sklearn.preprocessing import FunctionTransformer

X = np.array([[1], [10], [100]])

# Apply natural log transform
log_transformer = FunctionTransformer(np.log1p, validate=True)
X_log = log_transformer.fit_transform(X)

print(X_log)
```

> `np.log1p(x)` = log(1 + x) avoids issues with log(0)

---

### 🔧 Example 2: Scaling by 100

```python
scale_transformer = FunctionTransformer(lambda x: x * 100)
X_scaled = scale_transformer.fit_transform(np.array([[0.5], [0.8]]))
print(X_scaled)  # [[50.], [80.]]
```

---

### 🔄 Reversible Transform:

```python
transformer = FunctionTransformer(
    func=np.log1p,
    inverse_func=np.expm1  # inverse of log1p
)

X_trans = transformer.transform(X)
X_inv = transformer.inverse_transform(X_trans)
```

---

### 🔧 Use in Pipelines:

```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('log', FunctionTransformer(np.log1p)),
    ('model', LinearRegression())
])
```

---

### 📌 Parameters:

| Parameter           | Purpose                                         |
| ------------------- | ----------------------------------------------- |
| `func`              | Function to apply (e.g., `np.log1p`)            |
| `inverse_func`      | Function to invert transform (e.g., `np.expm1`) |
| `validate`          | Ensures input is 2D                             |
| `feature_names_out` | Custom output names (optional)                  |

---

### 🧠 Common Uses:

* Log or square root transforms for skewed features
* Feature engineering (e.g., `x²`, `sin(x)`, `normalize`)
* Replacing custom transformers in pipelines

---

### ⚠️ Notes:

* **Must work on NumPy arrays**.
* Not used for categorical encoding — only numerical transforms.
* Combine with `ColumnTransformer` to apply it to specific columns.
