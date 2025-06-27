# NumPy Basics

## Introduction to NumPy

### What is NumPy?
- Fundamental package for scientific computing in Python
- Provides N-dimensional array objects (ndarray)
- Mathematical functions for arrays
- Broadcasting capabilities
- Linear algebra, random number generation

### Why NumPy?
- **Performance**: C/Fortran implementations
- **Vectorization**: Operate on entire arrays
- **Memory efficiency**: Homogeneous data types
- **Foundation**: Base for other scientific libraries

## Array Creation

### Basic Array Creation
```python
import numpy as np

# From lists
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([[1, 2, 3], [4, 5, 6]])

# Array properties
print(f"Shape: {arr2.shape}")        # (2, 3)
print(f"Dimension: {arr2.ndim}")     # 2
print(f"Size: {arr2.size}")          # 6
print(f"Data type: {arr2.dtype}")    # int64
```

### Array Generation Functions
```python
# Zeros and ones
zeros = np.zeros((3, 4))
ones = np.ones((2, 3))
full = np.full((2, 2), 7)

# Range arrays
range_arr = np.arange(0, 10, 2)      # [0, 2, 4, 6, 8]
linspace_arr = np.linspace(0, 1, 5)  # [0, 0.25, 0.5, 0.75, 1]

# Random arrays
random_arr = np.random.random((3, 3))
normal_arr = np.random.normal(0, 1, (100,))
randint_arr = np.random.randint(0, 10, (2, 3))

# Identity matrix
identity = np.eye(3)
```

## Array Operations

### Arithmetic Operations
```python
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

# Element-wise operations
addition = a + b         # [6, 8, 10, 12]
subtraction = a - b      # [-4, -4, -4, -4]
multiplication = a * b   # [5, 12, 21, 32]
division = a / b         # [0.2, 0.33, 0.43, 0.5]
power = a ** 2           # [1, 4, 9, 16]

# Scalar operations
scaled = a * 2           # [2, 4, 6, 8]
shifted = a + 10         # [11, 12, 13, 14]
```

### Mathematical Functions
```python
# Trigonometric functions
angles = np.array([0, np.pi/2, np.pi])
sin_vals = np.sin(angles)
cos_vals = np.cos(angles)

# Exponential and logarithmic
exp_vals = np.exp([1, 2, 3])
log_vals = np.log([1, np.e, np.e**2])

# Statistical functions
data = np.random.normal(0, 1, 1000)
mean = np.mean(data)
std = np.std(data)
median = np.median(data)
```

## Array Indexing and Slicing

### Basic Indexing
```python
arr = np.array([0, 1, 2, 3, 4, 5])

# Single element
first = arr[0]           # 0
last = arr[-1]           # 5

# Slicing
subset = arr[1:4]        # [1, 2, 3]
every_other = arr[::2]   # [0, 2, 4]
reversed_arr = arr[::-1] # [5, 4, 3, 2, 1, 0]
```

### Multi-dimensional Indexing
```python
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Element access
element = matrix[1, 2]   # 6
row = matrix[1, :]       # [4, 5, 6]
column = matrix[:, 1]    # [2, 5, 8]

# Submatrix
submatrix = matrix[0:2, 1:3]  # [[2, 3], [5, 6]]
```

### Boolean Indexing
```python
data = np.array([1, 5, 3, 8, 2, 9])

# Boolean conditions
mask = data > 5          # [False, False, False, True, False, True]
filtered = data[mask]    # [8, 9]

# Complex conditions
complex_mask = (data > 2) & (data < 8)
result = data[complex_mask]  # [5, 3]
```

## Array Manipulation

### Reshaping
```python
arr = np.arange(12)

# Reshape
reshaped = arr.reshape(3, 4)
reshaped_auto = arr.reshape(3, -1)  # Auto-calculate dimension

# Flatten
flattened = reshaped.flatten()
raveled = reshaped.ravel()
```

### Joining and Splitting
```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

# Concatenation
horizontal = np.hstack([a, b])    # [[1, 2, 5, 6], [3, 4, 7, 8]]
vertical = np.vstack([a, b])      # [[1, 2], [3, 4], [5, 6], [7, 8]]
concatenated = np.concatenate([a, b], axis=1)

# Splitting
arr = np.arange(16).reshape(4, 4)
split_h = np.hsplit(arr, 2)
split_v = np.vsplit(arr, 2)
```

## Broadcasting

### Broadcasting Rules
```python
# Scalar with array
arr = np.array([1, 2, 3, 4])
result = arr + 10        # [11, 12, 13, 14]

# Arrays with different shapes
a = np.array([[1], [2], [3]])    # (3, 1)
b = np.array([10, 20, 30])       # (3,)
result = a + b                   # (3, 3) matrix

# Broadcasting example
matrix = np.random.random((5, 3))
row_means = np.mean(matrix, axis=1, keepdims=True)
centered = matrix - row_means    # Center each row
```

## Linear Algebra

### Matrix Operations
```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix multiplication
matmul = np.matmul(A, B)  # or A @ B
dot_product = np.dot(A, B)

# Matrix properties
transpose = A.T
determinant = np.linalg.det(A)
inverse = np.linalg.inv(A)
eigenvals, eigenvecs = np.linalg.eig(A)
```

### Solving Linear Systems
```python
# Solve Ax = b
A = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])
x = np.linalg.solve(A, b)

# Least squares solution
A_overdetermined = np.random.random((10, 5))
b_overdetermined = np.random.random(10)
x_lstsq = np.linalg.lstsq(A_overdetermined, b_overdetermined, rcond=None)[0]
```

## Practical Examples for ML

### Data Normalization
```python
def normalize_features(X):
    """Normalize features to have zero mean and unit variance."""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std, mean, std

# Example usage
data = np.random.normal(10, 5, (100, 3))
normalized_data, data_mean, data_std = normalize_features(data)
```

### Distance Calculations
```python
def euclidean_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt(np.sum((point1 - point2) ** 2))

def manhattan_distance(point1, point2):
    """Calculate Manhattan distance between two points."""
    return np.sum(np.abs(point1 - point2))

# Vectorized distance calculation
def pairwise_distances(X, Y):
    """Calculate pairwise distances between points in X and Y."""
    X_expanded = X[:, np.newaxis, :]  # (n_samples_X, 1, n_features)
    Y_expanded = Y[np.newaxis, :, :]  # (1, n_samples_Y, n_features)
    distances = np.sqrt(np.sum((X_expanded - Y_expanded) ** 2, axis=2))
    return distances
```

### Statistical Operations
```python
def compute_statistics(data):
    """Compute comprehensive statistics for a dataset."""
    stats = {
        'mean': np.mean(data, axis=0),
        'median': np.median(data, axis=0),
        'std': np.std(data, axis=0),
        'var': np.var(data, axis=0),
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0),
        'q25': np.percentile(data, 25, axis=0),
        'q75': np.percentile(data, 75, axis=0)
    }
    return stats

# Correlation matrix
def correlation_matrix(X):
    """Compute correlation matrix."""
    return np.corrcoef(X.T)
```

### Data Generation for ML
```python
def generate_classification_data(n_samples=100, n_features=2, n_classes=2):
    """Generate synthetic classification data."""
    np.random.seed(42)
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate labels with some pattern
    weights = np.random.randn(n_features)
    scores = X @ weights
    y = (scores > np.median(scores)).astype(int)
    
    return X, y

def generate_regression_data(n_samples=100, n_features=1, noise=0.1):
    """Generate synthetic regression data."""
    X = np.random.randn(n_samples, n_features)
    true_weights = np.random.randn(n_features)
    y = X @ true_weights + noise * np.random.randn(n_samples)
    
    return X, y, true_weights
```

## Performance Tips

### Vectorization
```python
# Avoid loops when possible
# Slow:
result = []
for i in range(len(arr)):
    result.append(arr[i] ** 2)

# Fast:
result = arr ** 2

# Use built-in functions
# Slow:
total = 0
for value in arr:
    total += value

# Fast:
total = np.sum(arr)
```

### Memory Considerations
```python
# Avoid unnecessary copies
arr = np.random.random((1000, 1000))

# Creates copy (slower)
modified = arr + 1

# In-place operation (faster)
arr += 1

# Use views when possible
subset = arr[100:200, 100:200]  # View, not copy
copy_subset = arr[100:200, 100:200].copy()  # Explicit copy
```

## Common Pitfalls

### Array vs Matrix Multiplication
```python
# Element-wise multiplication
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
elementwise = a * b      # [[5, 12], [21, 32]]

# Matrix multiplication
matrix_mult = a @ b      # [[19, 22], [43, 50]]
```

### Data Type Issues
```python
# Integer division truncation
int_arr = np.array([1, 2, 3])
result = int_arr / 2     # [0.5, 1.0, 1.5] (float result)

# Specify dtype to avoid issues
float_arr = np.array([1, 2, 3], dtype=float)
```

## Learning Objectives
- [ ] Create and manipulate NumPy arrays
- [ ] Perform mathematical operations efficiently
- [ ] Use broadcasting for vectorized computations
- [ ] Apply linear algebra operations
- [ ] Implement ML algorithms using NumPy

## Practice Exercises
1. Implement k-means clustering from scratch
2. Create a linear regression solver
3. Build a neural network forward pass
4. Implement PCA using eigendecomposition

## Resources
- NumPy documentation
- "Python for Data Analysis" by Wes McKinney
- NumPy tutorials and exercises online
