## 🟢 **1. Setup and Introduction to NumPy**

### 🔹 What is NumPy?

* **NumPy (Numerical Python)** is a powerful library in Python used for:

  * Numerical computations
  * Multi-dimensional array objects
  * Tools for integrating C/C++ and Fortran code
  * Linear algebra, Fourier transforms, and random number capabilities

### 🔹 Why use NumPy?

* Faster than Python lists (thanks to vectorization and low-level optimizations)
* More memory efficient
* Useful for Data Science, Machine Learning, and Scientific Computing

---

### 🔹 Installing NumPy

Install NumPy using pip:

```bash
pip install numpy
```

If you’re using Jupyter Notebook:

```python
!pip install numpy
```

If using Anaconda:

```bash
conda install numpy
```

---

### 🔹 Importing NumPy

```python
import numpy as np
```

> Alias `np` is a convention used by the entire Python community.

---

### 🔹 Checking Version

```python
np.__version__
```

---

### 🔹 First Array Example

```python
import numpy as np

arr = np.array([1, 2, 3, 4])
print(arr)
print(type(arr))  # <class 'numpy.ndarray'>
```

---

### 🔹 Summary

| Task          | Command/Explanation            |
| ------------- | ------------------------------ |
| Install       | `pip install numpy`            |
| Import        | `import numpy as np`           |
| Create array  | `np.array([1, 2, 3])`          |
| Check version | `np.__version__`               |
| Why NumPy?    | Fast, efficient, and versatile |

---

