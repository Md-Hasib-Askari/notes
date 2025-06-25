## 🔵 **10. Memory & Performance Optimization in NumPy**

Optimizing NumPy code helps you write faster, more efficient programs—especially useful for large datasets and scientific computing.

---

### 🔹 1. **In-Place Operations**

Avoid unnecessary memory usage by modifying arrays **in-place**.

#### ❌ Not in-place (creates a new array):

```python
a = np.array([1, 2, 3])
a = a + 1
```

#### ✅ In-place:

```python
a = np.array([1, 2, 3])
a += 1    # [2 3 4]
```

> ✅ Reduces memory overhead
> ❗ Always be careful when sharing references between arrays

---

### 🔹 2. **Use `dtype` Wisely**

Smaller data types use less memory:

```python
a = np.array([1, 2, 3], dtype=np.int8)
print(a.nbytes)    # 3 bytes (instead of 24 for int64)
```

> Use the smallest data type that fits your range of values.

---

### 🔹 3. **Avoid Python Loops (Use Vectorization)**

#### ❌ Slow Python loop:

```python
a = np.arange(100000)
b = []

for x in a:
    b.append(x * 2)
```

#### ✅ Fast NumPy vectorization:

```python
a = np.arange(100000)
b = a * 2
```

> NumPy’s C backend is much faster than Python loops.

---

### 🔹 4. **Use `np.vectorize()` for Element-wise Functions**

If you must use a custom function:

```python
def square(x):
    return x * x

vec_square = np.vectorize(square)
vec_square(np.array([1, 2, 3]))   # [1 4 9]
```

> Note: This doesn’t improve speed over loops but makes code cleaner.

---

### 🔹 5. **Pre-allocate Arrays Instead of Appending**

#### ❌ Bad (slow):

```python
arr = []
for i in range(10000):
    arr.append(i)
```

#### ✅ Good (fast):

```python
arr = np.empty(10000)
for i in range(10000):
    arr[i] = i
```

---

### 🔹 6. **Use `numexpr` or `Numba` for Heavy Computation**

```python
import numexpr as ne

a = np.random.rand(1000000)
b = np.random.rand(1000000)

# Much faster than plain NumPy
c = ne.evaluate("a + b * 2")
```

Or with **Numba**:

```python
from numba import njit

@njit
def fast_sum(arr):
    return np.sum(arr)
```

---

### 🔹 7. **Use `np.all()` and `np.any()` Efficiently**

These are faster than `sum()` or full comparisons:

```python
np.any(arr > 100)    # Stops at first True
np.all(arr < 1000)   # Stops at first False
```

---

### 🔹 8. **Use `memoryviews` and `stride tricks` (Advanced)**

For advanced memory layout optimization:

```python
from numpy.lib.stride_tricks import as_strided
```

This lets you reuse memory efficiently without copying—but must be used with caution.

---

### 📝 Summary

| Optimization Tip            | Example / Benefit                          |
| --------------------------- | ------------------------------------------ |
| In-place operations         | `a += 1`                                   |
| Efficient dtype             | Use `np.int8`, `np.float32`                |
| Avoid Python loops          | Use vectorized NumPy ops                   |
| Use `np.vectorize()`        | For applying custom functions element-wise |
| Pre-allocate arrays         | `np.empty()` or `np.zeros()`               |
| Use `numexpr` or `Numba`    | Faster computation for large data          |
| Use `np.all()` / `np.any()` | Fast boolean operations                    |
| Avoid copying large arrays  | Use `as_strided` or memoryviews (advanced) |
