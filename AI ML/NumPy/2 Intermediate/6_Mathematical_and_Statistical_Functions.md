## 🟡 **6. Mathematical & Statistical Functions in NumPy**

NumPy provides a wide range of mathematical and statistical operations that can be performed on entire arrays efficiently.

---

### 🔹 1. **Mathematical Operations**

#### 🔸 Element-wise Arithmetic

```python
a = np.array([1, 2, 3])

np.add(a, 1)      # [2 3 4]
np.subtract(a, 1) # [0 1 2]
np.multiply(a, 2) # [2 4 6]
np.divide(a, 2)   # [0.5 1.  1.5]
np.power(a, 2)    # [1 4 9]
np.mod(a, 2)      # [1 0 1]
```

> These functions are preferred over Python operators for clarity and broadcasting.

---

### 🔹 2. **Exponential and Logarithmic Functions**

```python
a = np.array([1, np.e, np.e**2])

np.exp([1, 2])          # [2.718, 7.389]
np.log(a)               # Natural log
np.log10(a)             # Base 10 log
np.log2(a)              # Base 2 log
```

---

### 🔹 3. **Trigonometric Functions**

```python
angles = np.array([0, np.pi/2, np.pi])

np.sin(angles)          # [0, 1, 0]
np.cos(angles)          # [1, 0, -1]
np.tan(angles)          # [0, inf, 0]

# Inverse
np.arcsin([0, 1])       # [0, π/2]
np.arccos([1, 0])       # [0, π/2]
np.arctan([0, 1])       # [0, π/4]
```

---

### 🔹 4. **Rounding Functions**

```python
a = np.array([1.234, 2.678])

np.round(a, 1)      # [1.2 2.7]
np.floor(a)         # [1. 2.]
np.ceil(a)          # [2. 3.]
np.trunc(a)         # [1. 2.]
```

---

### 🔹 5. **Statistical Functions**

#### On a 2D array:

```python
a = np.array([[1, 2, 3], [4, 5, 6]])
```

| Operation          | Code           | Output             |
| ------------------ | -------------- | ------------------ |
| Minimum            | `np.min(a)`    | `1`                |
| Maximum            | `np.max(a)`    | `6`                |
| Mean (Average)     | `np.mean(a)`   | `3.5`              |
| Median             | `np.median(a)` | `3.5`              |
| Standard Deviation | `np.std(a)`    | `1.7078`           |
| Variance           | `np.var(a)`    | `2.9166`           |
| Sum                | `np.sum(a)`    | `21`               |
| Cumulative Sum     | `np.cumsum(a)` | `[1 3 6 10 15 21]` |
| Product            | `np.prod(a)`   | `720`              |

---

### 🔹 6. **Aggregation by Axis**

Use `axis=0` for column-wise, `axis=1` for row-wise.

```python
np.sum(a, axis=0)       # [5 7 9]
np.mean(a, axis=1)      # [2. 5.]
```

---

### 🔹 7. **Percentile and Quantile**

```python
b = np.array([1, 3, 5, 7, 9])

np.percentile(b, 50)    # Median = 5
np.quantile(b, 0.25)    # First quartile = 3
```

---

### 📝 Summary

| Function Type       | Key Functions                           |
| ------------------- | --------------------------------------- |
| Arithmetic          | `add`, `subtract`, `multiply`, `divide` |
| Trigonometric       | `sin`, `cos`, `tan`, `arcsin`, etc.     |
| Exponential & Log   | `exp`, `log`, `log10`, `log2`           |
| Rounding            | `round`, `floor`, `ceil`, `trunc`       |
| Statistical (total) | `sum`, `mean`, `std`, `var`, `median`   |
| Statistical (axis)  | `np.sum(a, axis=0)`, `axis=1`           |
| Percentile/Quantile | `percentile`, `quantile`                |

---