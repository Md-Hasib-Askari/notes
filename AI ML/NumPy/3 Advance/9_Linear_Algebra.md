## 🔵 **9. Linear Algebra in NumPy**

NumPy provides powerful linear algebra capabilities through the `numpy.linalg` module. This includes operations like dot products, matrix inversion, solving linear systems, and more.

---

### 🔹 1. **Dot Product and Matrix Multiplication**

#### 🔸 Dot Product (1D or 2D):

```python
a = np.array([1, 2])
b = np.array([3, 4])

np.dot(a, b)       # 1×3 + 2×4 = 11
```

#### 🔸 Matrix Multiplication:

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

np.dot(A, B)
# or use the shorthand operator:
A @ B
```

> `@` is recommended for cleaner syntax in matrix multiplication.

---

### 🔹 2. **Transpose of a Matrix**

```python
A = np.array([[1, 2], [3, 4]])
A.T
# [[1 3]
#  [2 4]]
```

---

### 🔹 3. **Identity Matrix**

```python
np.eye(3)
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]
```

---

### 🔹 4. **Inverse of a Matrix**

```python
A = np.array([[1, 2], [3, 4]])
inv_A = np.linalg.inv(A)
```

> Only works if matrix is **square and non-singular** (i.e., `det(A) ≠ 0`).

---

### 🔹 5. **Determinant of a Matrix**

```python
A = np.array([[1, 2], [3, 4]])
np.linalg.det(A)    # Output: -2.000...
```

---

### 🔹 6. **Solving Systems of Linear Equations**

Given:

```
2x + 3y = 8  
3x + 4y = 11
```

Write as matrix form: `Ax = b`

```python
A = np.array([[2, 3], [3, 4]])
b = np.array([8, 11])

x = np.linalg.solve(A, b)   # Solution: [1. 2.]
```

---

### 🔹 7. **Eigenvalues and Eigenvectors**

```python
A = np.array([[1, 2], [2, 4]])
eigenvalues, eigenvectors = np.linalg.eig(A)
```

---

### 🔹 8. **Matrix Rank**

```python
np.linalg.matrix_rank(A)
```

---

### 🔹 9. **Singular Value Decomposition (SVD)**

```python
U, S, Vt = np.linalg.svd(A)
```

SVD is used for:

* Dimensionality reduction (e.g., PCA)
* Noise reduction
* Image compression

---

### 🔹 10. **Norms**

```python
a = np.array([3, 4])
np.linalg.norm(a)     # Euclidean norm (L2) = 5.0

# L1 norm
np.linalg.norm(a, ord=1)   # = 7
```

---

### 📝 Summary

| Operation                    | Function                   |
| ---------------------------- | -------------------------- |
| Dot product                  | `np.dot(a, b)` or `a @ b`  |
| Transpose                    | `A.T`                      |
| Inverse                      | `np.linalg.inv(A)`         |
| Determinant                  | `np.linalg.det(A)`         |
| Solve Ax = b                 | `np.linalg.solve(A, b)`    |
| Eigenvalues/Vectors          | `np.linalg.eig(A)`         |
| Matrix rank                  | `np.linalg.matrix_rank(A)` |
| Singular Value Decomposition | `np.linalg.svd(A)`         |
| Norm                         | `np.linalg.norm(a)`        |
