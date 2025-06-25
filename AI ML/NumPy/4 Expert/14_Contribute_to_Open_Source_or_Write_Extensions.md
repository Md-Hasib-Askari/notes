## 🔴 **14. Contribute to Open Source or Write Extensions**

To truly **master NumPy**, you can explore its internals, write high-performance extensions, or even contribute to its development.

---

### 🔹 1. **Learn How NumPy Is Implemented**

NumPy is mostly written in **C** with Python bindings.

* Core library: [`numpy/core/src`](https://github.com/numpy/numpy/tree/main/numpy/core/src)
* Uses CPython C-API for performance
* UFuncs (universal functions) are written in C and exposed to Python

📘 Explore [NumPy internals documentation](https://numpy.org/doc/stable/dev/)

---

### 🔹 2. **Write High-Performance Code with Cython**

**Cython** allows writing C extensions using Python-like syntax.

#### Example: Speeding up a function

```cython
# file: square.pyx
def square(double x):
    return x * x
```

Then compile with a `setup.py`, and import from Python.

> ✅ Cython is excellent for optimizing slow loops or math-heavy code.

---

### 🔹 3. **Use Numba for JIT Compilation**

Numba lets you compile Python + NumPy code to fast machine code using LLVM.

#### Example:

```python
from numba import njit
import numpy as np

@njit
def fast_dot(a, b):
    return np.dot(a, b)

a = np.random.rand(1000, 1000)
b = np.random.rand(1000, 1000)
result = fast_dot(a, b)
```

> ⚡ No need for a C compiler. Just install Numba and use decorators.

---

### 🔹 4. **Memory Layout: C vs Fortran Order**

* **C-contiguous**: Row-major (default in NumPy)
* **F-contiguous**: Column-major (used in Fortran, MATLAB)

```python
a = np.array([[1, 2], [3, 4]], order='F')
a.flags
```

Use `a.T.copy(order='C')` to ensure correct layout when interoperating with C/Fortran libraries.

---

### 🔹 5. **Write and Register Custom UFuncs in C**

For maximum control:

* Use the NumPy C API to register your own universal functions
* Compile as a shared library and import in Python

🧠 Recommended if you’re building domain-specific math libraries or need absolute performance.

---

### 🔹 6. **Contribute to NumPy**

Ways to get started:

* [GitHub: numpy/numpy](https://github.com/numpy/numpy)
* Look for “good first issues”
* Read the [contributor guide](https://numpy.org/devdocs/dev/index.html)
* Use `pytest` for testing and write docstrings in NumPy doc style

> 🌟 Even doc improvements and example code are valued contributions!

---

### 📝 Summary

| Goal                    | Tool / Action                       |
| ----------------------- | ----------------------------------- |
| Explore NumPy internals | Read `numpy/core/src/` codebase     |
| Speed up loops          | Use `Cython` or `Numba`             |
| JIT compilation         | `@njit` decorator (Numba)           |
| Deep memory control     | Use `order='C'` or `'F'`, `strides` |
| Contribute to NumPy     | GitHub issues, tests, documentation |
| Write C extensions      | Via NumPy C-API + shared libraries  |

