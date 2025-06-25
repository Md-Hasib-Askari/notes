## 🟢 **3. Basic Array Operations in NumPy**

NumPy arrays support fast **element-wise** operations using basic arithmetic and broadcasting.

---

### 🔹 1. **Element-wise Arithmetic**

```python
import numpy as np

a = np.array([10, 20, 30])
b = np.array([1, 2, 3])

# Arithmetic Operations
print(a + b)   # [11 22 33]
print(a - b)   # [9 18 27]
print(a * b)   # [10 40 90]
print(a / b)   # [10. 10. 10.]
print(a ** 2)  # [100 400 900]
```

Note: Operations are element-by-element, not matrix multiplication.

---

### 🔹 2. **Broadcasting**

Broadcasting allows NumPy to work with arrays of **different shapes** during arithmetic operations.

```python
a = np.array([1, 2, 3])
b = 2

print(a + b)   # [3 4 5]
```

```python
# 2D with 1D example
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([10, 20, 30])

print(a + b)
# [[11 22 33]
#  [14 25 36]]
```

**Broadcasting Rule**: NumPy compares shapes from the end. Dimensions must be equal or one of them must be 1.

---

### 🔹 3. **Comparison & Boolean Operations**

```python
a = np.array([1, 2, 3, 4])
b = np.array([2, 2, 2, 2])

print(a > b)      # [False False  True  True]
print(a == b)     # [False  True False False]
```

**Useful functions**:

```python
np.any(a > b)     # True
np.all(a > b)     # False
```

---

### 🔹 4. **Logical Operations**

```python
a = np.array([1, 2, 3])
b = np.array([3, 2, 1])

print(np.logical_and(a < 3, b > 1))  # [ True False False]
print(np.logical_or(a > 2, b < 2))   # [False False  True]
print(np.logical_not(a > 2))         # [ True  True False]
```

---

### 🔹 5. **Aggregate/Reduction Operations**

For an array `arr = np.array([[1, 2, 3], [4, 5, 6]])`

| Operation          | Code                  | Output    |
| ------------------ | --------------------- | --------- |
| Sum all            | `np.sum(arr)`         | `21`      |
| Column-wise sum    | `np.sum(arr, axis=0)` | `[5 7 9]` |
| Row-wise sum       | `np.sum(arr, axis=1)` | `[6 15]`  |
| Mean               | `np.mean(arr)`        | `3.5`     |
| Min/Max            | `np.min(arr)`         | `1`       |
| Standard deviation | `np.std(arr)`         | `1.7078`  |

---

### 🔹 6. **Other Handy Functions**

```python
np.sqrt([1, 4, 9])     # [1. 2. 3.]
np.exp([1, 2, 3])      # [2.718, 7.389, 20.085]
np.round([1.21, 2.56]) # [1. 3.]
```

---

### 📝 Summary

| Task             | Example                       |
| ---------------- | ----------------------------- |
| Element-wise Ops | `a + b`, `a * b`, `a ** 2`    |
| Broadcasting     | `a + scalar`, `a + [x, y, z]` |
| Comparison Ops   | `a > b`, `a == b`             |
| Logical Ops      | `np.logical_and()`, etc.      |
| Aggregates       | `sum()`, `mean()`, `std()`    |
| Math Functions   | `np.sqrt()`, `np.exp()`, etc. |

---
