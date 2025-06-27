# Principal Component Analysis (PCA)

## Overview
Principal Component Analysis (PCA) is a dimensionality reduction technique that transforms high-dimensional data into a lower-dimensional space while preserving as much variance as possible. It's widely used for data visualization, noise reduction, and feature extraction.

## Dimensionality Reduction
- **Variance maximization**: Finds directions with maximum variance in data
- **Eigenvalue decomposition**: Uses linear algebra to find principal components
- **Principal components**: Orthogonal directions that capture most data variance
- **Unsupervised technique**: Doesn't require labeled data

## Mathematical Foundation

### Core Concepts
```
Goal: Find k orthogonal directions that maximize variance
Where: k < original number of features

Steps:
1. Center the data: X_centered = X - mean(X)
2. Compute covariance matrix: C = (X_centered^T * X_centered) / (n-1)
3. Find eigenvectors and eigenvalues: C * v = λ * v
4. Sort by eigenvalues (descending)
5. Select top k eigenvectors as principal components
```

### Mathematical Formulation
- **Covariance matrix**: Measures how features vary together
- **Eigenvectors**: Directions of maximum variance (principal components)
- **Eigenvalues**: Amount of variance explained by each component
- **Explained variance ratio**: λᵢ / Σλⱼ for component i

## Python Implementation

### PCA from Scratch
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, make_classification

class PCA:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance_ratio = None
    
    def fit(self, X):
        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Compute covariance matrix
        cov_matrix = np.cov(X_centered.T)
        
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort eigenvectors by eigenvalues (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Store principal components
        self.components = eigenvectors[:, :self.n_components]
        
        # Calculate explained variance ratio
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio = eigenvalues[:self.n_components] / total_variance
        
        return self
    
    def transform(self, X):
        # Center and project data
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_pca):
        # Reconstruct original data
        return np.dot(X_pca, self.components.T) + self.mean

# Example with Iris dataset
iris = load_iris()
X = iris.data

# Apply custom PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print(f"Original shape: {X.shape}")
print(f"Reduced shape: {X_pca.shape}")
print(f"Explained variance ratio: {pca.explained_variance_ratio}")
print(f"Total variance explained: {np.sum(pca.explained_variance_ratio):.3f}")
```

### Using Scikit-learn
```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# Load digits dataset
digits = load_digits()
X, y = digits.data, digits.target

print(f"Original shape: {X.shape}")

# Standardize the data (important for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Plot explained variance
plt.figure(figsize=(12, 5))

# Cumulative explained variance
plt.subplot(1, 2, 1)
cumsum = np.cumsum(pca.explained_variance_ratio_)
plt.plot(range(1, len(cumsum) + 1), cumsum, 'bo-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance')
plt.grid(True)

# Individual component variance
plt.subplot(1, 2, 2)
plt.plot(range(1, 21), pca.explained_variance_ratio_[:20], 'ro-')
plt.xlabel('Component Number')
plt.ylabel('Explained Variance Ratio')
plt.title('Individual Component Variance')
plt.grid(True)

plt.tight_layout()
plt.show()

# Find number of components for 95% variance
n_components_95 = np.argmax(cumsum >= 0.95) + 1
print(f"Components needed for 95% variance: {n_components_95}")
```

### Choosing Number of Components
```python
def find_optimal_components(X, variance_threshold=0.95):
    """Find optimal number of components for given variance threshold"""
    
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit PCA with all components
    pca = PCA()
    pca.fit(X_scaled)
    
    # Calculate cumulative variance
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    
    # Find optimal number of components
    n_components = np.argmax(cumsum >= variance_threshold) + 1
    
    return n_components, cumsum, pca.explained_variance_ratio_

# Example usage
iris = load_iris()
n_comp, cumsum, individual_var = find_optimal_components(iris.data, 0.95)

print(f"Optimal components for 95% variance: {n_comp}")
print(f"Variance explained by first 2 components: {cumsum[1]:.3f}")

# Visualize
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumsum) + 1), cumsum, 'bo-', label='Cumulative')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
plt.axvline(x=n_comp, color='g', linestyle='--', label=f'Optimal: {n_comp}')
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Component Selection')
plt.legend()
plt.grid(True)
plt.show()
```

## Visualization and Interpretation

### 2D Visualization
```python
from sklearn.datasets import load_wine
import seaborn as sns

# Load wine dataset
wine = load_wine()
X, y = wine.data, wine.target

# Standardize and apply PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create visualization
plt.figure(figsize=(12, 5))

# Original features (first 2)
plt.subplot(1, 2, 1)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='viridis')
plt.xlabel(f'Feature 1: {wine.feature_names[0]}')
plt.ylabel(f'Feature 2: {wine.feature_names[1]}')
plt.title('Original Features')
plt.colorbar()

# PCA components
plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.title('Principal Components')
plt.colorbar()

plt.tight_layout()
plt.show()

print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.2%}")
```

### Component Interpretation
```python
def interpret_components(pca, feature_names, n_components=2):
    """Interpret what each principal component represents"""
    
    components_df = pd.DataFrame(
        pca.components_[:n_components].T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=feature_names
    )
    
    # Visualize component loadings
    plt.figure(figsize=(12, 8))
    sns.heatmap(components_df.T, annot=True, cmap='RdBu_r', center=0,
                fmt='.3f', cbar_kws={'label': 'Component Loading'})
    plt.title('Principal Component Loadings')
    plt.xlabel('Features')
    plt.ylabel('Principal Components')
    plt.tight_layout()
    plt.show()
    
    # Print top contributors for each component
    for i in range(n_components):
        print(f"\nPC{i+1} (Explains {pca.explained_variance_ratio_[i]:.2%} variance):")
        loadings = components_df[f'PC{i+1}'].abs().sort_values(ascending=False)
        print("Top 5 contributing features:")
        for feature, loading in loadings.head().items():
            direction = "+" if components_df.loc[feature, f'PC{i+1}'] > 0 else "-"
            print(f"  {direction} {feature}: {loading:.3f}")
    
    return components_df

# Example with wine dataset
import pandas as pd
components_df = interpret_components(pca, wine.feature_names)
```

## Real-world Applications

### Data Compression
```python
def image_compression_pca(image_array, n_components=50):
    """Compress image using PCA"""
    
    # Reshape image to 2D
    original_shape = image_array.shape
    if len(original_shape) == 3:  # Color image
        image_2d = image_array.reshape(-1, original_shape[-1])
    else:  # Grayscale
        image_2d = image_array.reshape(-1, 1)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    image_compressed = pca.fit_transform(image_2d)
    
    # Reconstruct
    image_reconstructed = pca.inverse_transform(image_compressed)
    
    if len(original_shape) == 3:
        image_reconstructed = image_reconstructed.reshape(original_shape)
    else:
        image_reconstructed = image_reconstructed.reshape(original_shape[0], original_shape[1])
    
    compression_ratio = (n_components * (image_2d.shape[1] + image_2d.shape[0])) / image_2d.size
    
    return image_reconstructed, compression_ratio, pca.explained_variance_ratio_.sum()

# Example with sample image (would need actual image data)
# compressed_img, ratio, variance = image_compression_pca(image_array, 50)
# print(f"Compression ratio: {ratio:.3f}")
# print(f"Variance preserved: {variance:.3f}")
```

### Feature Extraction for ML
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def compare_with_without_pca(X, y, n_components=None):
    """Compare ML performance with and without PCA"""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Without PCA
    clf_original = RandomForestClassifier(random_state=42)
    clf_original.fit(X_train_scaled, y_train)
    score_original = clf_original.score(X_test_scaled, y_test)
    
    # With PCA
    if n_components is None:
        n_components = min(X_train.shape[1] // 2, 10)
    
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    clf_pca = RandomForestClassifier(random_state=42)
    clf_pca.fit(X_train_pca, y_train)
    score_pca = clf_pca.score(X_test_pca, y_test)
    
    print(f"Original features: {X_train.shape[1]}")
    print(f"PCA components: {n_components}")
    print(f"Variance explained: {pca.explained_variance_ratio_.sum():.3f}")
    print(f"Accuracy without PCA: {score_original:.3f}")
    print(f"Accuracy with PCA: {score_pca:.3f}")
    print(f"Feature reduction: {n_components/X_train.shape[1]:.2%}")
    
    return score_original, score_pca, pca

# Example usage
digits = load_digits()
score_orig, score_pca, pca_model = compare_with_without_pca(digits.data, digits.target, 30)
```

### Anomaly Detection
```python
def pca_anomaly_detection(X, n_components=2, threshold_percentile=95):
    """Use PCA reconstruction error for anomaly detection"""
    
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Reconstruct data
    X_reconstructed = pca.inverse_transform(X_pca)
    
    # Calculate reconstruction error
    reconstruction_error = np.sum((X_scaled - X_reconstructed) ** 2, axis=1)
    
    # Set threshold
    threshold = np.percentile(reconstruction_error, threshold_percentile)
    
    # Identify anomalies
    anomalies = reconstruction_error > threshold
    
    return anomalies, reconstruction_error, threshold

# Example usage
from sklearn.datasets import make_blobs

# Generate normal data with some outliers
X_normal, _ = make_blobs(n_samples=200, centers=1, cluster_std=1, random_state=42)
X_outliers = np.random.uniform(-6, 6, (20, 2))
X_combined = np.vstack([X_normal, X_outliers])

# Detect anomalies
anomalies, errors, threshold = pca_anomaly_detection(X_combined, n_components=1)

# Visualize
plt.figure(figsize=(10, 6))
normal_points = ~anomalies
plt.scatter(X_combined[normal_points, 0], X_combined[normal_points, 1], 
           c='blue', alpha=0.7, label='Normal')
plt.scatter(X_combined[anomalies, 0], X_combined[anomalies, 1], 
           c='red', alpha=0.7, label='Anomaly')
plt.title('PCA-based Anomaly Detection')
plt.legend()
plt.show()

print(f"Detected {np.sum(anomalies)} anomalies out of {len(X_combined)} points")
```

## Implementation Best Practices

### Data Preprocessing
```python
def preprocess_for_pca(X, method='standard'):
    """Proper preprocessing for PCA"""
    
    if method == 'standard':
        # Standard scaling (most common)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    elif method == 'minmax':
        # Min-max scaling
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
    elif method == 'robust':
        # Robust scaling (for outliers)
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X
        scaler = None
    
    return X_scaled, scaler

# Example
X_processed, scaler = preprocess_for_pca(iris.data, 'standard')
```

### Handling Missing Values
```python
def pca_with_missing_values(X, n_components=2, strategy='mean'):
    """Handle missing values before PCA"""
    
    from sklearn.impute import SimpleImputer
    
    # Impute missing values
    if strategy == 'mean':
        imputer = SimpleImputer(strategy='mean')
    elif strategy == 'median':
        imputer = SimpleImputer(strategy='median')
    else:
        imputer = SimpleImputer(strategy='most_frequent')
    
    X_imputed = imputer.fit_transform(X)
    
    # Standardize and apply PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    return X_pca, pca, scaler, imputer
```

## Common Issues and Solutions

### Issue 1: Features on Different Scales
```python
# Problem: Features with larger scales dominate PCA
# Solution: Always standardize data

# Wrong approach
pca_wrong = PCA(n_components=2)
X_pca_wrong = pca_wrong.fit_transform(iris.data)

# Correct approach
scaler = StandardScaler()
X_scaled = scaler.fit_transform(iris.data)
pca_correct = PCA(n_components=2)
X_pca_correct = pca_correct.fit_transform(X_scaled)

print("Without scaling - explained variance:", pca_wrong.explained_variance_ratio_)
print("With scaling - explained variance:", pca_correct.explained_variance_ratio_)
```

### Issue 2: Choosing Wrong Number of Components
```python
def validate_component_choice(X, max_components=None):
    """Validate component selection using multiple criteria"""
    
    if max_components is None:
        max_components = min(X.shape[0], X.shape[1])
    
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Test different numbers of components
    results = []
    for n_comp in range(1, max_components + 1):
        pca = PCA(n_components=n_comp)
        X_pca = pca.fit_transform(X_scaled)
        
        # Reconstruction error
        X_reconstructed = pca.inverse_transform(X_pca)
        mse = np.mean((X_scaled - X_reconstructed) ** 2)
        
        results.append({
            'n_components': n_comp,
            'explained_variance': pca.explained_variance_ratio_.sum(),
            'reconstruction_mse': mse
        })
    
    return pd.DataFrame(results)

# Example usage
validation_results = validate_component_choice(iris.data, 4)
print(validation_results)
```

### Issue 3: Interpreting Components
```python
def component_interpretation_guide(pca, feature_names):
    """Guide for interpreting PCA components"""
    
    print("PCA Component Interpretation Guide:")
    print("=" * 50)
    
    for i, component in enumerate(pca.components_):
        print(f"\nPrincipal Component {i+1}:")
        print(f"Explains {pca.explained_variance_ratio_[i]:.2%} of variance")
        
        # Find features with highest absolute loadings
        feature_loadings = list(zip(feature_names, component))
        feature_loadings.sort(key=lambda x: abs(x[1]), reverse=True)
        
        print("Top contributing features:")
        for feature, loading in feature_loadings[:5]:
            direction = "increases" if loading > 0 else "decreases"
            print(f"  - {feature}: {loading:.3f} ({direction} PC{i+1})")
        
        # Interpretation hint
        positive_features = [f for f, l in feature_loadings if l > 0.3]
        negative_features = [f for f, l in feature_loadings if l < -0.3]
        
        if positive_features and negative_features:
            print(f"  Interpretation: Contrasts {positive_features} vs {negative_features}")
        elif positive_features:
            print(f"  Interpretation: Represents {positive_features}")
        elif negative_features:
            print(f"  Interpretation: Inverse of {negative_features}")

# Example usage
pca = PCA(n_components=2)
pca.fit(StandardScaler().fit_transform(iris.data))
component_interpretation_guide(pca, iris.feature_names)
```

## Advantages and Limitations

### Advantages
- **Dimensionality reduction**: Reduces computational complexity
- **Noise reduction**: Filters out noise in lower components
- **Visualization**: Enables plotting of high-dimensional data
- **Feature extraction**: Creates meaningful composite features
- **No supervision needed**: Works without labeled data
- **Mathematical foundation**: Well-understood linear algebra basis

### Limitations
- **Linear technique**: Cannot capture non-linear relationships
- **Interpretability**: Components may be hard to interpret
- **Variance focus**: May not preserve class separability
- **Scaling sensitive**: Requires proper data preprocessing
- **Information loss**: Lower components are discarded
- **Computational cost**: Eigenvalue decomposition for large datasets

## Alternatives to Standard PCA

### Kernel PCA
```python
from sklearn.decomposition import KernelPCA

# For non-linear dimensionality reduction
kernel_pca = KernelPCA(n_components=2, kernel='rbf', gamma=0.1)
X_kernel_pca = kernel_pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y)
plt.title('Original Data')

plt.subplot(1, 3, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
plt.title('Linear PCA')

plt.subplot(1, 3, 3)
plt.scatter(X_kernel_pca[:, 0], X_kernel_pca[:, 1], c=y)
plt.title('Kernel PCA')

plt.tight_layout()
plt.show()
```

### Sparse PCA
```python
from sklearn.decomposition import SparsePCA

# For sparse components (feature selection)
sparse_pca = SparsePCA(n_components=2, alpha=0.1)
X_sparse_pca = sparse_pca.fit_transform(X_scaled)

print("Sparse PCA components:")
print(sparse_pca.components_)
```

### Incremental PCA
```python
from sklearn.decomposition import IncrementalPCA

# For large datasets that don't fit in memory
inc_pca = IncrementalPCA(n_components=2, batch_size=50)
X_inc_pca = inc_pca.fit_transform(X_scaled)
```

## Learning Objectives
- [ ] **Understand PCA theory**: Master variance maximization and eigenvalue decomposition
- [ ] **Implement from scratch**: Code PCA using linear algebra fundamentals  
- [ ] **Reduce dimensionality**: Apply PCA for feature reduction and visualization
- [ ] **Interpret components**: Understand what principal components represent
- [ ] **Choose optimal components**: Use variance explained to select components
- [ ] **Preprocess data properly**: Apply standardization and handle missing values
- [ ] **Apply to real problems**: Use PCA for compression, visualization, and ML preprocessing
- [ ] **Recognize limitations**: Know when PCA is inappropriate (non-linear data)
- [ ] **Use advanced variants**: Apply Kernel PCA, Sparse PCA for specific needs
- [ ] **Evaluate results**: Assess reconstruction error and variance preservation

## Practice Exercises
1. Implement PCA from scratch and verify against scikit-learn
2. Apply PCA to the digits dataset and visualize in 2D
3. Use PCA for image compression and measure quality vs compression ratio
4. Compare classification performance with and without PCA preprocessing
5. Implement PCA-based anomaly detection system
6. Create component interpretation visualization for wine dataset
7. Apply Kernel PCA to non-linear data and compare with linear PCA
8. Build automated component selection based on explained variance thresholds