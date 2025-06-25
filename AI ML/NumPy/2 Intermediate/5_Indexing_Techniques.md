## 🟡 **5. Indexing Techniques in NumPy**

Indexing is essential for selecting and modifying array elements. NumPy offers several powerful indexing methods:

---

### 🔹 1. **Basic Indexing (Similar to Python Lists)**

#### 1D Array:

```python
a = np.array([10, 20, 30, 40])
print(a[0])     # 10
print(a[-1])    # 40
```

#### 2D Array:

```python
a = np.array([[1, 2, 3], [4, 5, 6]])

print(a[0, 1])      # 2
print(a[1][2])      # 6
print(a[:, 1])      # column: [2 5]
print(a[1, :])      # row: [4 5 6]
```

---

### 🔹 2. **Slicing**

Syntax: `arr[start:stop:step]`

```python
a = np.array([0, 10, 20, 30, 40, 50])
print(a[1:5:2])     # [10 30]
```

2D Slicing:

```python
a = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
a[0:2, 1:3]         # [[1 2]
                    #  [4 5]]
```

---

### 🔹 3. **Boolean Indexing**

Select values based on condition:

```python
a = np.array([10, 15, 20, 25])
mask = a > 15
print(a[mask])          # [20 25]

# Direct usage:
print(a[a % 2 == 0])    # [10 20]
```

---

### 🔹 4. **Fancy Indexing (Index Arrays)**

Use arrays of indices to access multiple elements:

```python
a = np.array([10, 20, 30, 40])
indices = [0, 2, 3]
print(a[indices])       # [10 30 40]
```

2D Example:

```python
a = np.array([[1, 2], [3, 4], [5, 6]])
print(a[[0, 2], [1, 0]])   # [2, 5]
```

---

### 🔹 5. **`np.where()` for Conditional Indexing**

```python
a = np.array([10, 20, 30, 40])
idx = np.where(a > 20)
print(idx)          # (array([2, 3]),)
print(a[idx])       # [30 40]

# If-else like replacement:
np.where(a > 25, 1, 0)   # [0 0 1 1]
```

---

### 🔹 6. **`np.nonzero()` and `np.argwhere()`**

```python
a = np.array([0, 1, 0, 3])
np.nonzero(a)           # (array([1, 3]),)

b = np.array([[0, 1], [2, 0]])
np.argwhere(b)          # [[0 1]
                        #  [1 0]]
```

---

### 📝 Summary

| Technique        | Description                               |
| ---------------- | ----------------------------------------- |
| Basic Indexing   | `a[0]`, `a[1, 2]`                         |
| Slicing          | `a[start:stop:step]`                      |
| Boolean Indexing | `a[a > 10]`                               |
| Fancy Indexing   | `a[[0, 2]]`, `a[[i], [j]]`                |
| `np.where()`     | Conditional index or replacement          |
| `np.nonzero()`   | Indices of non-zero elements              |
| `np.argwhere()`  | Row/column positions of matching elements |

---