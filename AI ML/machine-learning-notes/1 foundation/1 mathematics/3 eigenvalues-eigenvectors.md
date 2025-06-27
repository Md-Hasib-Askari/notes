# Eigenvalues and Eigenvectors

## Fundamental Concepts

### Definitions
- **Eigenvector**: A non-zero vector that changes only by a scalar factor when a linear transformation is applied
- **Eigenvalue**: The scalar factor by which the eigenvector is scaled

### Mathematical Representation
For a square matrix A and vector v:
```
Av = λv
```
Where:
- A is the matrix
- v is the eigenvector
- λ (lambda) is the eigenvalue

## Properties

### Key Properties
- Eigenvalues can be real or complex
- Eigenvectors are not unique (can be scaled)
- Sum of eigenvalues = trace of matrix
- Product of eigenvalues = determinant of matrix

### Geometric Interpretation
- Eigenvectors represent directions of stretching
- Eigenvalues represent the amount of stretching
- Principal axes of data ellipsoids

## Applications in Machine Learning

### Principal Component Analysis (PCA)
- Finding principal components
- Dimensionality reduction
- Data compression
- Feature extraction

### Other Applications
- Spectral clustering
- Graph analysis
- Markov chains
- Stability analysis

## Calculation Methods

### Characteristic Polynomial
1. Form the matrix (A - λI)
2. Calculate determinant
3. Solve polynomial equation
4. Find corresponding eigenvectors

### Numerical Methods
- Power iteration
- QR algorithm
- Jacobi method

## Practical Examples

### Example 1: 2x2 Matrix
```python
import numpy as np

A = np.array([[4, 2], [1, 3]])
eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:", eigenvectors)
```

### Example 2: PCA Implementation
```python
# Covariance matrix eigendecomposition
cov_matrix = np.cov(data.T)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
```

## Learning Checklist
- [ ] Understand eigen-decomposition
- [ ] Calculate eigenvalues manually
- [ ] Implement PCA using eigendecomposition
- [ ] Interpret geometric meaning
- [ ] Apply to real datasets

## Resources
- Linear Algebra textbooks
- NumPy documentation
- Scikit-learn PCA implementation
