## 🔹 `OrdinalEncoder` – Encoding Ordered Categorical Features

### ✅ Purpose:

* Transforms **categorical input features** into **integer values**.
* Unlike `LabelEncoder`, it works on **features (X)** not labels (y).
* Assumes an **ordinal relationship** (e.g., Low < Medium < High).

---

### 📌 Example:

```python
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

# Ordered feature
X = np.array([['low'], ['medium'], ['high'], ['medium']])

encoder = OrdinalEncoder(categories=[['low', 'medium', 'high']])
X_encoded = encoder.fit_transform(X)

print(X_encoded)
```

---

### 🔍 Output:

```plaintext
[[0.]
 [1.]
 [2.]
 [1.]]
```

Each category is mapped to an integer **based on the order you define**.

---

### 🧠 Default Behavior:

If you **don’t specify `categories`**, it assumes **lexicographical order**:

```python
# 'high' < 'low' < 'medium'
```

So it's **better to explicitly define the order** when dealing with meaningful categories.

---

### 🔄 Decoding:

```python
encoder.inverse_transform([[2]])  # Output: [['high']]
```

---

### 🔧 Multiple Columns:

```python
X = np.array([
    ['low', 'small'],
    ['medium', 'medium'],
    ['high', 'large']
])

encoder = OrdinalEncoder(categories=[
    ['low', 'medium', 'high'],
    ['small', 'medium', 'large']
])

X_encoded = encoder.fit_transform(X)
```

---

### ⚠️ Use Case:

* Only use when the **order of categories** is **meaningful**.

  * Good: `'low'`, `'medium'`, `'high'`
  * Bad: `'red'`, `'blue'`, `'green'` ❌ → Use `OneHotEncoder` instead

---

### ✅ In Pipelines:

```python
from sklearn.compose import ColumnTransformer

ordinal_features = ['education']
ordinal_order = [['highschool', 'bachelor', 'master', 'phd']]

preprocessor = ColumnTransformer(transformers=[
    ('ord', OrdinalEncoder(categories=ordinal_order), ordinal_features)
])
```
