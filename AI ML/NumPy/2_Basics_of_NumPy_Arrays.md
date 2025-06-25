## 🟢 **2. Basics of NumPy Arrays**

### 🔹 What is a NumPy Array?

* A NumPy array is a grid of values, **all of the same type**, and is indexed by a tuple of nonnegative integers.
* The number of dimensions is called the array’s **rank**, and each dimension has a size known as its **shape**.

---

### 🔹 Creating Arrays

```python
import numpy as np
```

#### 1. From Python List or Tuple

```python
arr1 = np.array([1, 2, 3])
arr2 = np.array([(1, 2, 3), (4, 5, 6)])
```

#### 2. Predefined Arrays

```python
np.zeros((2, 3))      # 2x3 array of zeros
np.ones((3, 4))       # 3x4 array of ones
np.full((2, 2), 7)    # 2x2 array filled with 7
np.eye(3)             # 3x3 identity matrix
np.empty((2, 2))      # Uninitialized 2x2 array
```

---

### 🔹 Array Properties

```python
a = np.array([[1, 2, 3], [4, 5, 6]])
```

| Property     | Description                        | Example          |
| ------------ | ---------------------------------- | ---------------- |
| `a.shape`    | Returns dimensions (rows, columns) | `(2, 3)`         |
| `a.ndim`     | Number of dimensions               | `2`              |
| `a.size`     | Total number of elements           | `6`              |
| `a.dtype`    | Data type of array elements        | `dtype('int64')` |
| `a.itemsize` | Size in bytes of one array element | `8` (for int64)  |
| `a.nbytes`   | Total bytes consumed               | `48`             |

---

### 🔹 Array Indexing and Slicing

#### 1D Array:

```python
a = np.array([10, 20, 30, 40])
a[0]        # 10
a[1:3]      # [20 30]
a[-1]       # 40
```

#### 2D Array:

```python
a = np.array([[1, 2, 3], [4, 5, 6]])
a[0, 1]     # 2
a[:, 1]     # column 1: [2 5]
a[1, :]     # row 1: [4 5 6]
```

---

### 🔹 Reshaping Arrays

```python
a = np.array([1, 2, 3, 4, 5, 6])
a.reshape((2, 3))    # Reshape to 2 rows, 3 columns
```

### 🔹 Flattening Arrays

```python
a.ravel()    # Returns flattened array (1D)
a.flatten()  # Same as ravel, but returns a copy
```

---

### 📝 Summary

| Task                 | Code Example                       |
| -------------------- | ---------------------------------- |
| Create array         | `np.array([1, 2, 3])`              |
| Zeros/Ones/Full      | `np.zeros((2,2))`, `np.full(... )` |
| Identity matrix      | `np.eye(3)`                        |
| Shape and Dimensions | `arr.shape`, `arr.ndim`            |
| Indexing/Slicing     | `arr[1:3]`, `arr[:, 0]`            |
| Reshape and Flatten  | `reshape()`, `flatten()`           |

---

