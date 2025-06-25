## 🧱 **Internal Structure of NumPy**

NumPy is structured as a **multi-layered library**, combining:

* **Python** for user-level interfaces,
* **C** for performance-critical computation,
* and **tools** like Cython and Numba for extending functionality.

---

### 🔹 1. **High-Level Python API**

These are the functions **you use directly**, like:

```python
np.array(), np.mean(), np.dot(), np.linspace(), np.random.rand()
```

Located in:

* `numpy/__init__.py`
* `numpy/core/`
* `numpy/lib/`, `numpy/linalg/`, `numpy/fft/`, etc.

---

### 🔹 2. **Core (ndarray Object)**

At the heart of NumPy is the `ndarray` object:

```python
a = np.array([1, 2, 3])
```

This object is:

* **Typed**: every element has the same dtype (e.g., float32, int64)
* **Multidimensional**
* **Memory-efficient**: stores data in a contiguous memory block

Its structure includes:

* `data`: a pointer to the memory
* `shape`: tuple of dimensions
* `strides`: how many bytes to step in each dimension
* `dtype`: the data type
* `flags`: memory layout info (C/F-contiguous)

---

### 🔹 3. **C Backend (Performance Core)**

The core computation engine of NumPy is written in **C** for speed:

* Located in: `numpy/core/src/`
* Implements:

  * Mathematical operations (add, subtract, multiply…)
  * ufuncs (universal functions)
  * Broadcasting
  * Memory management

> These C functions are exposed to Python via bindings and wrappers.

---

### 🔹 4. **Universal Functions (UFuncs)**

UFuncs are **vectorized functions** (implemented in C), such as:

* `np.add()`, `np.exp()`, `np.sin()`, etc.

They:

* Automatically apply to entire arrays
* Support broadcasting
* Can be overridden by custom implementations

Internals in: `numpy/core/src/umath/`

---

### 🔹 5. **Submodules**

| Submodule          | Purpose                           |
| ------------------ | --------------------------------- |
| `numpy.linalg`     | Linear algebra (uses BLAS/LAPACK) |
| `numpy.fft`        | Fast Fourier Transform            |
| `numpy.random`     | Random number generation          |
| `numpy.polynomial` | Polynomial manipulation           |
| `numpy.ma`         | Masked arrays                     |
| `numpy.typing`     | Type hints and stubs for IDEs     |

---

### 🔹 6. **Extension Interfaces**

* **C API**: `numpy/core/include/numpy/` — allows C extensions
* **Cython support**: C-level access in Python syntax
* **Numba**: JIT compiler that understands NumPy semantics
* **Third-party libraries** (e.g., Pandas, TensorFlow) build on NumPy arrays

---

### 🔹 7. **Testing & Documentation**

* Unit tests: in `numpy/tests/`
* Docs: written in reStructuredText using Sphinx (`numpy/doc/`)

---

### 📝 Summary Table

| Layer               | Description                                   |
| ------------------- | --------------------------------------------- |
| Python API          | User-facing functions in `numpy`              |
| Core Object         | `ndarray`: multidimensional array object      |
| UFuncs              | Fast, element-wise operations in C            |
| C Backend           | All math logic, loops, and dispatching        |
| Submodules          | Specialized functionality (FFT, linalg, etc.) |
| Extension Interface | C API, Cython, Numba, PyTorch, etc.           |
| Tests & Docs        | Full test coverage and Sphinx-based docs      |
