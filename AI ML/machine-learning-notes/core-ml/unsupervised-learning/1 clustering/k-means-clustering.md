# K-Means Clustering

## Overview
K-means is a centroid-based clustering algorithm that partitions data into k clusters by minimizing within-cluster sum of squares (WCSS). It's one of the most popular unsupervised learning algorithms due to its simplicity and effectiveness.

## Algorithm Overview
- **Centroid-based clustering**: Each cluster represented by its center point
- **Iterative optimization**: Alternates between assignment and update steps
- **Euclidean distance**: Default distance metric for similarity
- **Hard clustering**: Each point belongs to exactly one cluster

## How K-Means Works

### Implementation Steps
1. **Initialize centroids**: Randomly place k centroids in feature space
2. **Assign points to clusters**: Assign each point to nearest centroid
3. **Update centroids**: Move centroids to mean of assigned points
4. **Repeat until convergence**: Continue until centroids stop moving significantly

### Mathematical Foundation
```
Objective: Minimize WCSS = Σ(i=1 to k) Σ(x in Ci) ||x - μi||²
Where:
- k = number of clusters
- Ci = set of points in cluster i
- μi = centroid of cluster i
```

## Python Implementation

### Basic K-Means from Scratch
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class KMeans:
    def __init__(self, k=3, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
    
    def fit(self, X):
        # Initialize centroids randomly
        self.centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]
        
        for _ in range(self.max_iters):
            # Assign points to closest centroid
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            self.labels = np.argmin(distances, axis=0)
            
            # Update centroids
            new_centroids = np.array([X[self.labels == i].mean(axis=0) for i in range(self.k)])
            
            # Check for convergence
            if np.all(np.abs(self.centroids - new_centroids) < self.tol):
                break
                
            self.centroids = new_centroids
        
        return self
    
    def predict(self, X):
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

# Example usage
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)
kmeans = KMeans(k=4)
kmeans.fit(X)

plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels, cmap='viridis')
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], 
           c='red', marker='x', s=200, linewidths=3)
plt.title('K-Means Clustering Results')
plt.show()
```

### Using Scikit-learn
```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate sample data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)

# Apply K-means
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)

# Visualize results
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.7)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
           c='red', marker='x', s=200, linewidths=3, label='Centroids')
plt.title('K-Means Clustering with Scikit-learn')
plt.legend()
plt.show()

print(f"Inertia (WCSS): {kmeans.inertia_:.2f}")
```

## Choosing Optimal K

### Elbow Method
```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def elbow_method(X, max_k=10):
    inertias = []
    k_range = range(1, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertias, 'bo-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia (WCSS)')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    plt.show()
    
    return inertias

# Example usage
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)
inertias = elbow_method(X)
```

### Silhouette Analysis
```python
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt

def silhouette_analysis(X, max_k=10):
    silhouette_scores = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, silhouette_scores, 'bo-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis for Optimal k')
    plt.grid(True)
    plt.show()
    
    return silhouette_scores

# Example usage
scores = silhouette_analysis(X)
optimal_k = range(2, len(scores) + 2)[np.argmax(scores)]
print(f"Optimal k: {optimal_k}")
```

## Variations and Improvements

### K-means++
```python
# Better initialization strategy
kmeans_plus = KMeans(n_clusters=4, init='k-means++', random_state=42)
labels_plus = kmeans_plus.fit_predict(X)

print(f"K-means++ inertia: {kmeans_plus.inertia_:.2f}")
```

### Mini-batch K-means
```python
from sklearn.cluster import MiniBatchKMeans

# For large datasets
mini_kmeans = MiniBatchKMeans(n_clusters=4, batch_size=100, random_state=42)
labels_mini = mini_kmeans.fit_predict(X)

print(f"Mini-batch K-means inertia: {mini_kmeans.inertia_:.2f}")
```

## Real-world Applications

### Customer Segmentation
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Example: Customer segmentation
def customer_segmentation_example():
    # Sample customer data
    data = {
        'annual_spending': [20000, 15000, 50000, 75000, 30000, 60000],
        'frequency_visits': [12, 8, 24, 36, 18, 30],
        'avg_purchase': [150, 120, 300, 400, 200, 350]
    }
    df = pd.DataFrame(data)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    
    # Apply K-means
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    print("Customer Segments:")
    print(df.groupby('cluster').mean())
    
    return df

# Run example
customer_segments = customer_segmentation_example()
```

### Image Color Quantization
```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def color_quantization(image_path, k=8):
    # Load and reshape image
    image = Image.open(image_path)
    data = np.array(image).reshape(-1, 3)
    
    # Apply K-means
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data)
    
    # Replace colors with cluster centers
    new_colors = kmeans.cluster_centers_[kmeans.labels_]
    new_image = new_colors.reshape(image.size[1], image.size[0], 3)
    
    return new_image.astype(np.uint8)

# Example usage would require an actual image file
# quantized = color_quantization('image.jpg', k=16)
```

## Advantages and Limitations

### Advantages
- **Simple and intuitive**: Easy to understand and implement
- **Computationally efficient**: O(n*k*i*d) time complexity
- **Guaranteed convergence**: Always converges to local minimum
- **Works well on spherical clusters**: Good for globular cluster shapes

### Limitations
- **Requires pre-specified k**: Need to know number of clusters
- **Sensitive to initialization**: Different starts can yield different results
- **Assumes spherical clusters**: Struggles with irregular shapes
- **Sensitive to outliers**: Outliers can skew centroid positions
- **Scale sensitive**: Features with larger scales dominate

## Best Practices

### Data Preprocessing
```python
from sklearn.preprocessing import StandardScaler

# Always scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Or use robust scaling for outliers
from sklearn.preprocessing import RobustScaler
robust_scaler = RobustScaler()
X_robust = robust_scaler.fit_transform(X)
```

### Handling Categorical Data
```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd

# For mixed data types
def preprocess_mixed_data(df):
    numeric_features = df.select_dtypes(include=[np.number]).columns
    categorical_features = df.select_dtypes(include=['object']).columns
    
    # Scale numeric features
    scaler = StandardScaler()
    df[numeric_features] = scaler.fit_transform(df[numeric_features])
    
    # Encode categorical features
    df_encoded = pd.get_dummies(df, columns=categorical_features)
    
    return df_encoded
```

### Evaluation Metrics
```python
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def evaluate_clustering(X, labels):
    """Comprehensive clustering evaluation"""
    silhouette = silhouette_score(X, labels)
    calinski = calinski_harabasz_score(X, labels)
    davies_bouldin = davies_bouldin_score(X, labels)
    
    print(f"Silhouette Score: {silhouette:.3f}")
    print(f"Calinski-Harabasz Score: {calinski:.3f}")
    print(f"Davies-Bouldin Score: {davies_bouldin:.3f}")
    
    return silhouette, calinski, davies_bouldin

# Example usage
scores = evaluate_clustering(X, labels)
```

## Common Issues and Solutions

### Issue 1: Poor Initialization
```python
# Solution: Use k-means++ or multiple random starts
kmeans = KMeans(n_clusters=4, init='k-means++', n_init=10)
```

### Issue 2: Choosing Wrong K
```python
# Solution: Use multiple methods to determine k
def find_optimal_k(X, max_k=10):
    methods = {}
    
    # Elbow method
    inertias = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    methods['elbow'] = inertias
    
    # Silhouette method
    silhouettes = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        silhouettes.append(silhouette_score(X, labels))
    methods['silhouette'] = silhouettes
    
    return methods
```

### Issue 3: Non-spherical Clusters
```python
# Solution: Consider alternative algorithms
from sklearn.cluster import DBSCAN, AgglomerativeClustering

# For irregular shapes, use DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X)

# For hierarchical structure, use Agglomerative
agg_clustering = AgglomerativeClustering(n_clusters=4)
agg_labels = agg_clustering.fit_predict(X)
```

## Learning Objectives
- [ ] **Understand K-means algorithm**: Master the iterative centroid-update process
- [ ] **Implement from scratch**: Code basic K-means without libraries
- [ ] **Choose optimal K**: Use elbow method and silhouette analysis
- [ ] **Handle initialization issues**: Apply K-means++ and multiple runs
- [ ] **Preprocess data properly**: Scale features and handle categorical data
- [ ] **Evaluate clustering quality**: Use multiple metrics for assessment
- [ ] **Apply to real problems**: Customer segmentation, image processing
- [ ] **Recognize limitations**: Know when K-means is inappropriate
- [ ] **Compare with alternatives**: Understand when to use other clustering methods

## Practice Exercises
1. Implement K-means from scratch and compare with scikit-learn
2. Apply K-means to the Iris dataset and analyze results
3. Use K-means for customer segmentation with RFM analysis
4. Perform image color quantization using K-means
5. Compare K-means with DBSCAN on non-spherical data
6. Implement and test different initialization strategies
7. Create a comprehensive clustering evaluation framework