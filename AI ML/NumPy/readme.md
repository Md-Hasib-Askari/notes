## 🟢 Beginner Level

### 1. **Setup**

* Install NumPy:

  ```bash
  pip install numpy
  ```
* Import:

  ```python
  import numpy as np
  ```

### 2. **Basics of NumPy Arrays**

* Creating arrays: `np.array()`, `np.zeros()`, `np.ones()`, `np.full()`, `np.eye()`
* Array properties: `shape`, `dtype`, `ndim`, `size`, `itemsize`
* Array indexing and slicing
* Reshaping arrays: `reshape()`, `ravel()`, `flatten()`

### 3. **Basic Array Operations**

* Element-wise arithmetic: `+`, `-`, `*`, `/`, `**`
* Broadcasting rules
* Comparison and logical operations: `>`, `<`, `==`, `np.logical_and()`, etc.
* Useful functions: `np.sum()`, `np.min()`, `np.max()`, `np.mean()`, `np.std()`, `np.sort()`

---

## 🟡 Intermediate Level

### 4. **Array Manipulation**

* Joining arrays: `np.concatenate()`, `np.vstack()`, `np.hstack()`, `np.stack()`
* Splitting arrays: `np.split()`, `np.hsplit()`, `np.vsplit()`
* Transpose and axis operations: `.T`, `np.transpose()`, `np.swapaxes()`, `np.moveaxis()`

### 5. **Indexing Techniques**

* Boolean indexing
* Fancy indexing (index arrays)
* `np.where()`, `np.nonzero()`, `np.argwhere()`
* Conditional replacement: `np.where(condition, x, y)`

### 6. **Mathematical & Statistical Functions**

* Aggregation functions: `sum()`, `mean()`, `median()`, `std()`, `var()`, `cumsum()`
* Trigonometric functions: `sin()`, `cos()`, `tan()`, `arcsin()`, etc.
* Exponential and log: `np.exp()`, `np.log()`, `np.log10()`

### 7. **Random Number Generation**

* `np.random.rand()`, `randn()`, `randint()`
* `np.random.choice()`, `np.random.shuffle()`
* Setting seed: `np.random.seed()`

---

## 🔵 Advanced Level

### 8. **Advanced Indexing and Broadcasting**

* Use `np.newaxis` or `reshape()` to control dimensions
* Master broadcasting mechanics for performance
* Apply functions across axes using `np.apply_along_axis()`

### 9. **Linear Algebra**

* `np.dot()`, `np.matmul()`, `@` operator
* `np.linalg.inv()`, `np.linalg.det()`, `np.linalg.eig()`, `np.linalg.svd()`
* Solving systems: `np.linalg.solve()`

### 10. **Memory & Performance Optimization**

* In-place operations to reduce memory: `arr += 1` instead of `arr = arr + 1`
* Use `np.vectorize()` to optimize Python loops
* Profile with `%timeit`, `%memit`, and use `numexpr` for faster computation

### 11. **Masked Arrays and NaN Handling**

* `np.isnan()`, `np.nan_to_num()`, `np.nanmean()`, etc.
* Use `np.ma` module for masked arrays

---

## 🔴 Expert Level

### 12. **Broadcasting Tricks & Advanced Functions**

* Meshgrids: `np.meshgrid()`, `np.mgrid[]`
* `np.einsum()` for optimized tensor operations
* Use `stride_tricks` for advanced memory handling
* Create custom `ufuncs` using `np.frompyfunc()`

### 13. **Integration with Other Libraries**

* Use NumPy with:

  * **Pandas** for dataframes
  * **Matplotlib/Seaborn** for visualization
  * **SciPy** for scientific computing
  * **TensorFlow/PyTorch** for deep learning
* Convert between: `numpy <-> pandas <-> torch/tensorflow`

### 14. **Contribute to Open Source or Write Extensions**

* Learn how NumPy is implemented (in C)
* Write Cython or Numba-accelerated code
* Explore memory layout, C-contiguous vs F-contiguous

---

## 📚 Recommended Resources

* **Official NumPy Docs**: [https://numpy.org/doc/](https://numpy.org/doc/)
* **Book**: *Python for Data Analysis* by Wes McKinney
* **Free Course**: NumPy tutorials on [W3Schools](https://www.w3schools.com/python/numpy/) or [Kaggle](https://www.kaggle.com/learn/numpy)
* **YouTube Channels**: Corey Schafer, Tech With Tim

---
