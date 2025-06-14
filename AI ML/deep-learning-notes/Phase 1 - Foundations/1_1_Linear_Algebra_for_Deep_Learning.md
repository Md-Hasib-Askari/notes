## 📘 Topic 1.1: **Linear Algebra for Deep Learning**

### 🔑 Key Concepts

1. **Scalars, Vectors, Matrices, Tensors**

   * Scalar: Single value (e.g., 3)
   * Vector: 1D array (e.g., \[2, 5, 7])
   * Matrix: 2D array (e.g., \[\[1, 2], \[3, 4]])
   * Tensor: N-dimensional array (used in deep learning frameworks)

2. **Matrix Operations**

   * **Addition/Subtraction**: Same shape matrices
   * **Multiplication**:

     * Element-wise: A \* B
     * Dot Product: A · B (inner product)
     * Matrix product: A @ B

3. **Identity Matrix (I)**

   * Diagonal ones: acts as a multiplicative identity.

4. **Transpose (Aᵀ)**

   * Flip rows ↔ columns

5. **Inverse (A⁻¹)**

   * Only square matrices
   * A · A⁻¹ = I

6. **Determinant (|A|)**

   * Used to check invertibility
   * If det(A) = 0 → Not invertible

7. **Eigenvalues & Eigenvectors**

   * Describe how a transformation changes a vector's direction and magnitude.
   * Used in PCA and optimization.

---

### 🧠 Intuition for Deep Learning

* **Data** = Tensors
* **Weights** = Matrices
* **Operations** = Dot products & transformations
* Backpropagation = Heavy use of matrix calculus

---

### 🧪 Exercises

#### ✅ Conceptual

1. Explain the difference between a matrix and a tensor.
2. Why do we use dot products in neural networks?

#### ✅ Coding (Python + NumPy)

```python
import numpy as np

# Define matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Element-wise addition and multiplication
print("A + B =\n", A + B)
print("A * B (element-wise) =\n", A * B)

# Matrix multiplication
print("A @ B =\n", A @ B)

# Transpose
print("Transpose of A:\n", A.T)

# Inverse
A_inv = np.linalg.inv(A)
print("Inverse of A:\n", A_inv)

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigenvalues:\n", eigenvalues)
print("Eigenvectors:\n", eigenvectors)
```
