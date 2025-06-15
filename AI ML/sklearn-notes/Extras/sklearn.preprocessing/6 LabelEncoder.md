## 🔹 `LabelEncoder` – Encode Target Labels as Integers

### ✅ Purpose:

* Converts **categorical labels** (usually in the target `y`) into numeric values.
* Each unique label is mapped to an integer.
* Useful for algorithms that require numerical labels (e.g., classification tasks).

---

### 📌 Example:

```python
from sklearn.preprocessing import LabelEncoder

# Sample labels
y = ['spam', 'ham', 'spam', 'eggs']

# Create encoder
le = LabelEncoder()

# Fit and transform
y_encoded = le.fit_transform(y)

print(y_encoded)
```

---

### 🔍 Output:

```plaintext
[2 1 2 0]
```

The mapping (alphabetical by default):

* `'eggs'` → 0
* `'ham'` → 1
* `'spam'` → 2

---

### 🔄 Decoding:

```python
le.inverse_transform([2, 1, 0])
# Output: ['spam' 'ham' 'eggs']
```

---

### 📌 Use Case:

* Best used for encoding the **target variable** `y`.
* Should **not** be used for input features with **unordered categorical values** (use `OneHotEncoder` or `OrdinalEncoder` instead).

---

### ⚠️ Warning:

Using `LabelEncoder` on input features can cause models to assume **ordinal relationships** where there aren’t any:

```plaintext
Color: Red = 0, Green = 1, Blue = 2  ❌
```

Instead, use `OneHotEncoder` for such cases.

---

### ✅ Summary

| Use Case                | Encoder to Use     |
| ----------------------- | ------------------ |
| Target variable         | ✅ `LabelEncoder`   |
| Input feature (ordinal) | ✅ `OrdinalEncoder` |
| Input feature (nominal) | ✅ `OneHotEncoder`  |
