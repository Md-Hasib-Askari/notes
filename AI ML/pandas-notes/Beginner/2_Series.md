## 🟢 **2. Series**

### ✅ What is a Series?

* A **Pandas Series** is a **one-dimensional** labeled array capable of holding any data type: integers, strings, floats, objects, etc.
* Think of it as a column in a table (but by itself).

---

### ✅ Creating a Series

#### From a list:

```python
import pandas as pd

s = pd.Series([10, 20, 30])
print(s)
```

**Output:**

```
0    10
1    20
2    30
dtype: int64
```

#### From a list with custom index:

```python
s = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
print(s)
```

#### From a dictionary:

```python
s = pd.Series({'a': 100, 'b': 200})
print(s)
```

---

### ✅ Accessing Elements

#### By Index:

```python
print(s['a'])    # 100
print(s[0])      # 100
```

#### Slicing:

```python
print(s[:2])     # First two elements
```

---

### ✅ Basic Operations

```python
s = pd.Series([1, 2, 3])
print(s + 2)           # Add scalar
print(s * 3)           # Multiply
print(s.sum())         # Sum of elements
print(s.mean())        # Mean
print(s.max())         # Max value
```

---

### ✅ Attributes and Methods

* `s.index` – returns the index.
* `s.values` – returns the underlying data as NumPy array.
* `s.dtype` – shows data type.
* `s.head(n)` – returns first n elements.
* `s.tail(n)` – returns last n elements.

