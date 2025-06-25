## 🔴 **12. Broadcasting Tricks & Advanced Functions**

---

### 🔹 1. **Meshgrids with `np.meshgrid()` and `np.mgrid[]`**

#### 🔸 `np.meshgrid()` – Generate coordinate matrices from coordinate vectors:

```python
x = np.array([1, 2, 3])
y = np.array([4, 5])
X, Y = np.meshgrid(x, y)
```

* `X` shape: (2, 3)
* `Y` shape: (2, 3)

Useful in plotting and 2D computations.

#### 🔸 `np.mgrid[]` – Dense coordinate grid (more concise):

```python
x, y = np.mgrid[0:5, 0:5]   # Generates a full grid of shape (2, 5, 5)
```

---

### 🔹 2. **`np.einsum()` – Einstein Summation**

Einstein summation notation allows for high-performance and readable tensor operations.

#### Examples:

```python
a = np.array([1, 2, 3])
b = np.array([0, 1, 0])

np.dot(a, b) == np.einsum('i,i->', a, b)
```

Matrix multiplication:

```python
A = np.random.rand(2, 3)
B = np.random.rand(3, 4)
C = np.einsum('ij,jk->ik', A, B)
```

> ⚡ `einsum` is often faster than `dot()` for large tensor operations.

---

### 🔹 3. **Memory Tricks with `stride_tricks`**

Efficiently manipulate memory without copying data.

#### Example: Sliding window

```python
from numpy.lib.stride_tricks import sliding_window_view

a = np.array([1, 2, 3, 4, 5])
sliding_window_view(a, window_shape=3)
# [[1 2 3]
#  [2 3 4]
#  [3 4 5]]
```

> Use carefully. Misuse of `stride_tricks` can lead to memory corruption.

---

### 🔹 4. **Custom ufuncs with `np.frompyfunc()`**

You can vectorize your Python function over NumPy arrays.

```python
def add_strings(x, y):
    return str(x) + str(y)

vadd = np.frompyfunc(add_strings, 2, 1)
vadd(['a', 'b'], ['x', 'y'])  # ['ax', 'by']
```

* Input args: 2
* Output args: 1

---

### 📝 Summary

| Feature           | Tool                    | Use Case                                  |
| ----------------- | ----------------------- | ----------------------------------------- |
| Coordinate grids  | `meshgrid()`, `mgrid[]` | Vector fields, surface plots, simulations |
| Tensor operations | `einsum()`              | Efficient multi-index summation           |
| Memory views      | `stride_tricks`         | Zero-copy data reshaping                  |
| Custom vector ops | `frompyfunc()`          | Extend NumPy with custom Python logic     |
