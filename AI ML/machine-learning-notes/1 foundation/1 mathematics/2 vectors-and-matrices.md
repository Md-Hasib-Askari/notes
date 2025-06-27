# Vectors and Matrices

## Vector Operations

### Vector Basics
- Definition and notation
- Vector addition and subtraction
- Scalar multiplication
- Dot product (inner product)
- Cross product
- Vector norms (L1, L2, infinity)

### Geometric Interpretation
- Vectors as points in space
- Vector projections
- Orthogonality and orthonormality
- Basis vectors

## Matrix Operations

### Matrix Fundamentals
- Matrix notation and indexing
- Matrix addition and subtraction
- Matrix multiplication rules
- Identity matrix
- Matrix transpose

### Special Matrices
- Symmetric matrices
- Diagonal matrices
- Orthogonal matrices
- Positive definite matrices

### Matrix Properties
- Rank of a matrix
- Determinant
- Trace
- Inverse matrices

## Applications
- Data representation in ML
- Linear transformations
- System of linear equations
- Coordinate transformations

## Practice Exercises
1. Vector dot product calculations
2. Matrix multiplication examples
3. Finding matrix inverses
4. Solving linear systems

## Code Examples
```python
import numpy as np

# Vector operations
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
dot_product = np.dot(v1, v2)

# Matrix operations
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = np.dot(A, B)
```
