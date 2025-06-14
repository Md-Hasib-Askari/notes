

## 🟢 **1. Math & Programming Basics – Part 1: Linear Algebra**

### 🔍 **Key Concepts**

* **Scalars, Vectors, Matrices, Tensors**
* **Matrix Operations**: addition, subtraction, multiplication
* **Dot Product**: essential for understanding neural layers
* **Matrix Transpose and Inverse**
* **Identity and Diagonal Matrices**

### ✍️ **Quick Notes**

* A **vector** is a 1D array, a **matrix** is 2D, and **tensors** generalize to n-dimensions.
* **Dot product** measures similarity:

  $$
  \vec{a} \cdot \vec{b} = \sum_i a_i b_i
  $$
* **Matrix multiplication** is not element-wise:

  $$
  (AB)_{ij} = \sum_k A_{ik} B_{kj}
  $$
* **Transpose**: flips rows and columns:

  $$
  (A^T)_{ij} = A_{ji}
  $$

### 🧠 **Why This Matters in Generative AI**

Neural networks use matrix operations to propagate signals forward and backward. Understanding these is non-negotiable for building any deep model.

---

### 🧪 **Practice Exercises**

1. Create two vectors in Python and compute:

   * their dot product
   * cosine similarity
2. Multiply a 2x3 matrix with a 3x2 matrix using `NumPy`.
3. Implement a matrix transpose function from scratch (no NumPy).

```python
import numpy as np

# Example vectors
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Dot product
print("Dot Product:", np.dot(a, b))

# Matrix multiplication
A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[7, 8], [9, 10], [11, 12]])
print("Matrix Product:\n", np.dot(A, B))
```

