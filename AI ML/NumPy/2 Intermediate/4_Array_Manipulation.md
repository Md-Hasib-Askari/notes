## 🟡 **4. Array Manipulation in NumPy**

Array manipulation in NumPy includes reshaping, joining, splitting, and changing the orientation of arrays.

---

### 🔹 1. **Reshaping Arrays**

Changes the shape without changing the data.

```python
a = np.arange(6)           # [0 1 2 3 4 5]
a.reshape((2, 3))          # [[0 1 2]
                           #  [3 4 5]]
```

> ❗ Total number of elements must remain the same.

---

### 🔹 2. **Flattening Arrays**

Converts multi-dimensional arrays to 1D.

```python
a = np.array([[1, 2], [3, 4]])

a.flatten()  # Returns a copy: [1 2 3 4]
a.ravel()    # Returns a view (faster): [1 2 3 4]
```

---

### 🔹 3. **Transpose and Axis Operations**

```python
a = np.array([[1, 2], [3, 4]])

a.T  # Transpose: [[1 3]
     #             [2 4]]

np.transpose(a)  # Same as a.T
```

You can also change specific axes:

```python
a = np.arange(8).reshape(2, 2, 2)
np.swapaxes(a, 0, 2)
np.moveaxis(a, 0, -1)
```

---

### 🔹 4. **Joining Arrays (Concatenation)**

#### 1D Arrays

```python
a = np.array([1, 2])
b = np.array([3, 4])
np.concatenate((a, b))   # [1 2 3 4]
```

#### 2D Arrays

```python
a = np.array([[1, 2]])
b = np.array([[3, 4]])

np.vstack((a, b))        # Vertical: [[1 2]
                         #            [3 4]]

np.hstack((a, b))        # Horizontal: [[1 2 3 4]]

np.stack((a, b))         # Stack along new dimension (3D)
```

---

### 🔹 5. **Splitting Arrays**

```python
a = np.array([1, 2, 3, 4, 5, 6])
np.split(a, 3)           # Split into 3 arrays
```

For 2D:

```python
a = np.array([[1, 2], [3, 4], [5, 6]])
np.vsplit(a, 3)          # Split into vertical slices
np.hsplit(a.T, 2)        # Horizontal split after transpose
```

---

### 🔹 6. **Insert, Delete, Append**

```python
a = np.array([1, 2, 3])

np.append(a, [4, 5])     # [1 2 3 4 5]
np.insert(a, 1, 10)      # [1 10 2 3]
np.delete(a, 0)          # [2 3]
```

---

### 📝 Summary

| Operation     | Function(s)                        |
| ------------- | ---------------------------------- |
| Reshape       | `reshape()`                        |
| Flatten       | `ravel()`, `flatten()`             |
| Transpose     | `.T`, `transpose()`                |
| Stack/Join    | `vstack()`, `hstack()`, `stack()`  |
| Split         | `split()`, `vsplit()`, `hsplit()`  |
| Insert/Delete | `insert()`, `delete()`, `append()` |

---
