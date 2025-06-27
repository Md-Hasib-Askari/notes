# K-Nearest Neighbors (KNN)

## Overview
K-Nearest Neighbors is a simple, intuitive machine learning algorithm used for both classification and regression. It makes predictions based on the 'k' closest training examples in the feature space.

## Algorithm Principle

### Key Characteristics
- **Distance-based**: Uses distance metrics to find similar data points
- **Lazy learning**: No training phase - stores all data and computes at prediction time
- **Non-parametric**: Makes no assumptions about data distribution
- **Instance-based**: Predictions based on specific training instances

### How it Works
1. Store all training data
2. For new prediction, calculate distance to all training points
3. Find k nearest neighbors
4. For classification: majority vote; for regression: average

```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=4, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
```

## Implementation Details

### Distance Metrics
Choose appropriate distance metric based on data type:

```python
from sklearn.neighbors import KNeighborsClassifier

# Euclidean distance (default) - good for continuous features
knn_euclidean = KNeighborsClassifier(n_neighbors=5, metric='euclidean')

# Manhattan distance - good when features have different scales
knn_manhattan = KNeighborsClassifier(n_neighbors=5, metric='manhattan')

# Minkowski distance - generalization of Euclidean and Manhattan
knn_minkowski = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=3)

# Hamming distance - good for categorical features
knn_hamming = KNeighborsClassifier(n_neighbors=5, metric='hamming')
```

### K Selection
Finding optimal k value is crucial:

```python
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# Test different k values
k_range = range(1, 31)
k_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5)
    k_scores.append(scores.mean())

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(k_range, k_scores)
plt.xlabel('Value of K')
plt.ylabel('Cross-Validated Accuracy')
plt.title('KNN: Varying Number of Neighbors')
plt.grid(True)
plt.show()

# Find optimal k
optimal_k = k_range[np.argmax(k_scores)]
print(f"Optimal K: {optimal_k}")
```

### Weighted Voting
Give more influence to closer neighbors:

```python
# Distance-weighted KNN
knn_weighted = KNeighborsClassifier(
    n_neighbors=5, 
    weights='distance'  # or 'uniform' for equal weights
)
knn_weighted.fit(X_train, y_train)
```

## Optimization Techniques

### Data Structures for Efficiency

```python
# KD-Tree (default for low dimensions)
knn_kdtree = KNeighborsClassifier(
    n_neighbors=5,
    algorithm='kd_tree'
)

# Ball Tree (better for high dimensions)
knn_balltree = KNeighborsClassifier(
    n_neighbors=5,
    algorithm='ball_tree'
)

# Brute force (exact but slow)
knn_brute = KNeighborsClassifier(
    n_neighbors=5,
    algorithm='brute'
)

# Auto (sklearn chooses best)
knn_auto = KNeighborsClassifier(
    n_neighbors=5,
    algorithm='auto'
)
```

### Feature Scaling
KNN is sensitive to feature scales:

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Or normalization
normalizer = MinMaxScaler()
X_train_norm = normalizer.fit_transform(X_train)
X_test_norm = normalizer.transform(X_test)

# Train on scaled data
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)
```

## Complete Example: Iris Classification

```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning
param_grid = {
    'n_neighbors': range(1, 21),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

knn = KNeighborsClassifier()
grid_search = GridSearchCV(
    knn, param_grid, cv=5, 
    scoring='accuracy', n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)

# Best model
best_knn = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")

# Predictions
y_pred = best_knn.predict(X_test_scaled)

# Evaluation
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names, 
            yticklabels=iris.target_names)
plt.title('KNN Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

## Advantages and Disadvantages

### Advantages
- **Simple to understand and implement**
- **No assumptions about data distribution**
- **Works well with small datasets**
- **Can be used for both classification and regression**
- **Naturally handles multi-class problems**

### Disadvantages
- **Computationally expensive at prediction time**
- **Sensitive to irrelevant features**
- **Sensitive to feature scaling**
- **Struggles with high-dimensional data (curse of dimensionality)**
- **Memory intensive**

## Best Practices

### 1. Feature Preprocessing
```python
# Remove irrelevant features
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif, k=3)
X_selected = selector.fit_transform(X_train_scaled, y_train)
```

### 2. Handle Imbalanced Data
```python
from sklearn.utils.class_weight import compute_sample_weight

# Calculate sample weights
sample_weights = compute_sample_weight('balanced', y_train)

# Use in cross-validation
scores = cross_val_score(knn, X_train_scaled, y_train, 
                        fit_params={'sample_weight': sample_weights})
```

### 3. Dimensionality Reduction
```python
from sklearn.decomposition import PCA

# Reduce dimensions before KNN
pca = PCA(n_components=0.95)  # Keep 95% variance
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
```

## Common Pitfalls

1. **Not scaling features** - Always standardize or normalize
2. **Using odd k for binary classification** - Avoids ties
3. **Ignoring computational complexity** - Consider dataset size
4. **Not handling missing values** - Impute before distance calculation
5. **Using KNN with high-dimensional sparse data** - Consider dimensionality reduction

## When to Use KNN

### Good for:
- Small to medium datasets
- Non-linear decision boundaries
- Multi-class classification
- Recommendation systems
- Anomaly detection

### Avoid when:
- Very large datasets
- High-dimensional data without preprocessing
- Real-time predictions required
- Features are mostly irrelevant

## Learning Objectives
- [x] Understand KNN algorithm principles
- [x] Implement KNN for classification and regression
- [x] Choose optimal K value using cross-validation
- [x] Apply appropriate distance metrics
- [x] Handle feature scaling and preprocessing
- [x] Optimize for performance with data structures
- [x] Handle curse of dimensionality
- [x] Evaluate model performance properly