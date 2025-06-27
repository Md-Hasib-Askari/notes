# Hierarchical Clustering

## Learning Objectives
- [ ] Understand agglomerative and divisive clustering
- [ ] Build and interpret dendrograms
- [ ] Choose appropriate linkage methods
- [ ] Determine optimal number of clusters
- [ ] Implement hierarchical clustering from scratch
- [ ] Apply to real-world clustering problems

## Introduction

Hierarchical clustering creates a tree of clusters by iteratively merging or splitting clusters. Unlike K-means, it doesn't require pre-specifying the number of clusters and provides a hierarchy of cluster relationships.

## Types of Hierarchical Clustering

### 1. Agglomerative (Bottom-Up)
- Start with each point as its own cluster
- Iteratively merge closest clusters
- Most common approach

### 2. Divisive (Top-Down)
- Start with all points in one cluster
- Iteratively split clusters
- Less common, computationally expensive

## Linkage Criteria

### Single Linkage (Minimum)
```python
# Distance between closest points in different clusters
def single_linkage(cluster1, cluster2):
    return min(distance(p1, p2) for p1 in cluster1 for p2 in cluster2)
```

### Complete Linkage (Maximum)
```python
# Distance between farthest points in different clusters
def complete_linkage(cluster1, cluster2):
    return max(distance(p1, p2) for p1 in cluster1 for p2 in cluster2)
```

### Average Linkage
```python
# Average distance between all pairs
def average_linkage(cluster1, cluster2):
    distances = [distance(p1, p2) for p1 in cluster1 for p2 in cluster2]
    return sum(distances) / len(distances)
```

### Ward Linkage
- Minimizes within-cluster variance
- Best for compact, similarly-sized clusters

## Implementation with Scikit-learn

### Basic Implementation
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

# Generate sample data
X, y_true = make_blobs(n_samples=300, centers=4, n_features=2, 
                       random_state=42, cluster_std=0.60)

# Perform hierarchical clustering
hierarchical = AgglomerativeClustering(n_clusters=4, linkage='ward')
y_pred = hierarchical.fit_predict(X)

# Plot results
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis')
plt.title('True Clusters')

plt.subplot(1, 3, 2)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
plt.title('Hierarchical Clustering')

plt.tight_layout()
plt.show()
```

### Creating Dendrograms
```python
# Create dendrogram
plt.figure(figsize=(10, 6))
linkage_matrix = linkage(X, method='ward')
dendrogram(linkage_matrix, truncate_mode='level', p=3)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()
```

## Advanced Implementation

### Custom Hierarchical Clustering Class
```python
import numpy as np
from scipy.spatial.distance import pdist, squareform

class HierarchicalClustering:
    def __init__(self, linkage='single'):
        self.linkage = linkage
        self.labels_ = None
        self.n_clusters_ = None
        
    def fit(self, X, n_clusters=2):
        self.n_samples = X.shape[0]
        
        # Compute distance matrix
        distances = pdist(X)
        dist_matrix = squareform(distances)
        
        # Initialize clusters (each point is its own cluster)
        clusters = [[i] for i in range(self.n_samples)]
        cluster_distances = []
        
        # Merge clusters until we have n_clusters
        while len(clusters) > n_clusters:
            # Find closest pair of clusters
            min_dist = float('inf')
            merge_i, merge_j = 0, 1
            
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    dist = self._compute_cluster_distance(
                        clusters[i], clusters[j], dist_matrix
                    )
                    if dist < min_dist:
                        min_dist = dist
                        merge_i, merge_j = i, j
            
            # Merge clusters
            new_cluster = clusters[merge_i] + clusters[merge_j]
            clusters = [clusters[k] for k in range(len(clusters)) 
                       if k not in [merge_i, merge_j]]
            clusters.append(new_cluster)
            cluster_distances.append(min_dist)
        
        # Assign labels
        self.labels_ = np.zeros(self.n_samples, dtype=int)
        for cluster_id, cluster in enumerate(clusters):
            for point_id in cluster:
                self.labels_[point_id] = cluster_id
        
        self.n_clusters_ = len(clusters)
        return self
    
    def _compute_cluster_distance(self, cluster1, cluster2, dist_matrix):
        if self.linkage == 'single':
            return min(dist_matrix[i, j] for i in cluster1 for j in cluster2)
        elif self.linkage == 'complete':
            return max(dist_matrix[i, j] for i in cluster1 for j in cluster2)
        elif self.linkage == 'average':
            distances = [dist_matrix[i, j] for i in cluster1 for j in cluster2]
            return np.mean(distances)
        else:
            raise ValueError(f"Unknown linkage: {self.linkage}")

# Example usage
hc = HierarchicalClustering(linkage='ward')
hc.fit(X, n_clusters=4)
print(f"Predicted clusters: {hc.labels_}")
```

## Dendrogram Analysis

### Cutting Trees at Different Heights
```python
from scipy.cluster.hierarchy import fcluster

# Create linkage matrix
linkage_matrix = linkage(X, method='ward')

# Cut tree at different heights
clusters_2 = fcluster(linkage_matrix, 2, criterion='maxclust')
clusters_3 = fcluster(linkage_matrix, 3, criterion='maxclust')
clusters_4 = fcluster(linkage_matrix, 4, criterion='maxclust')

# Visualize different cuts
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Dendrogram
dendrogram(linkage_matrix, ax=axes[0, 0])
axes[0, 0].set_title('Dendrogram')

# Different cluster cuts
for i, (n_clusters, labels) in enumerate([(2, clusters_2), 
                                         (3, clusters_3), 
                                         (4, clusters_4)]):
    row, col = (i + 1) // 2, (i + 1) % 2
    axes[row, col].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    axes[row, col].set_title(f'{n_clusters} Clusters')

plt.tight_layout()
plt.show()
```

### Optimal Number of Clusters
```python
def find_optimal_clusters(X, max_clusters=10):
    """Find optimal number of clusters using silhouette score"""
    from sklearn.metrics import silhouette_score
    
    scores = []
    cluster_range = range(2, max_clusters + 1)
    
    for n_clusters in cluster_range:
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        cluster_labels = clusterer.fit_predict(X)
        score = silhouette_score(X, cluster_labels)
        scores.append(score)
    
    # Plot scores
    plt.figure(figsize=(8, 5))
    plt.plot(cluster_range, scores, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Optimal Number of Clusters')
    plt.grid(True)
    plt.show()
    
    return cluster_range[np.argmax(scores)]

optimal_k = find_optimal_clusters(X)
print(f"Optimal number of clusters: {optimal_k}")
```

## Practical Applications

### Customer Segmentation
```python
# Load customer data
import pandas as pd

# Example customer data
np.random.seed(42)
n_customers = 1000

customer_data = pd.DataFrame({
    'annual_spending': np.random.gamma(2, 2000, n_customers),
    'frequency': np.random.poisson(10, n_customers),
    'recency': np.random.exponential(30, n_customers),
    'age': np.random.normal(45, 15, n_customers)
})

# Normalize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(customer_data)

# Perform hierarchical clustering
clusterer = AgglomerativeClustering(n_clusters=5, linkage='ward')
customer_segments = clusterer.fit_predict(X_scaled)

# Analyze segments
customer_data['segment'] = customer_segments
segment_summary = customer_data.groupby('segment').agg({
    'annual_spending': ['mean', 'std'],
    'frequency': ['mean', 'std'],
    'recency': ['mean', 'std'],
    'age': ['mean', 'std']
}).round(2)

print("Customer Segment Analysis:")
print(segment_summary)
```

### Gene Expression Analysis
```python
# Simulated gene expression data
n_genes = 100
n_samples = 50

# Create expression matrix
expression_data = np.random.lognormal(0, 1, (n_genes, n_samples))

# Add some structure (gene groups)
for i in range(0, 25):
    expression_data[i, :25] *= 3  # Upregulated in first half of samples
for i in range(25, 50):
    expression_data[i, 25:] *= 3  # Upregulated in second half of samples

# Cluster genes
gene_clusters = AgglomerativeClustering(n_clusters=4, linkage='average')
gene_labels = gene_clusters.fit_predict(expression_data)

# Visualize heatmap with clusters
import seaborn as sns

# Sort genes by cluster
sorted_indices = np.argsort(gene_labels)
sorted_expression = expression_data[sorted_indices]

plt.figure(figsize=(12, 8))
sns.heatmap(sorted_expression, cmap='viridis', cbar=True)
plt.title('Gene Expression Heatmap (Clustered)')
plt.xlabel('Samples')
plt.ylabel('Genes (sorted by cluster)')
plt.show()
```

## Comparison of Linkage Methods

```python
def compare_linkage_methods(X):
    """Compare different linkage methods"""
    linkage_methods = ['single', 'complete', 'average', 'ward']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, method in enumerate(linkage_methods):
        if method == 'ward':
            clusterer = AgglomerativeClustering(n_clusters=4, linkage=method)
        else:
            clusterer = AgglomerativeClustering(n_clusters=4, linkage=method)
        
        labels = clusterer.fit_predict(X)
        
        axes[i].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
        axes[i].set_title(f'{method.capitalize()} Linkage')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

compare_linkage_methods(X)
```

## Advantages and Disadvantages

### Advantages
- No need to specify number of clusters beforehand
- Provides hierarchy of clusters
- Deterministic results
- Works well with any distance metric
- Can find arbitrarily shaped clusters (with single linkage)

### Disadvantages
- Computationally expensive: O(n³) time complexity
- Sensitive to outliers
- Difficult to handle large datasets
- Choice of linkage method affects results significantly

## Best Practices

1. **Choose appropriate linkage method:**
   - Ward: For compact, similar-sized clusters
   - Single: For elongated clusters
   - Complete: For compact clusters
   - Average: Balanced approach

2. **Preprocessing:**
   - Scale features to same range
   - Handle outliers carefully
   - Consider dimensionality reduction for high-dimensional data

3. **Determining optimal clusters:**
   - Use dendrogram visual inspection
   - Apply silhouette analysis
   - Consider domain knowledge

4. **Large datasets:**
   - Use sampling for initial exploration
   - Consider mini-batch approaches
   - Use approximate methods for very large datasets

## Real-World Example: Market Segmentation

```python
# Complete market segmentation pipeline
def market_segmentation_pipeline():
    # Generate synthetic market data
    np.random.seed(42)
    n_customers = 2000
    
    # Customer features
    data = {
        'age': np.random.normal(40, 12, n_customers),
        'income': np.random.lognormal(10.5, 0.5, n_customers),
        'spending_score': np.random.beta(2, 5, n_customers) * 100,
        'online_purchases': np.random.poisson(15, n_customers),
        'store_visits': np.random.negative_binomial(10, 0.3, n_customers)
    }
    
    df = pd.DataFrame(data)
    
    # Preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    
    # Find optimal clusters
    silhouette_scores = []
    K_range = range(2, 11)
    
    for k in K_range:
        clusterer = AgglomerativeClustering(n_clusters=k, linkage='ward')
        labels = clusterer.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(score)
    
    optimal_k = K_range[np.argmax(silhouette_scores)]
    
    # Final clustering
    final_clusterer = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
    df['segment'] = final_clusterer.fit_predict(X_scaled)
    
    # Analysis
    segment_analysis = df.groupby('segment').agg({
        'age': 'mean',
        'income': 'mean',
        'spending_score': 'mean',
        'online_purchases': 'mean',
        'store_visits': 'mean'
    }).round(2)
    
    print(f"Optimal number of segments: {optimal_k}")
    print("\nSegment Characteristics:")
    print(segment_analysis)
    
    return df, optimal_k

# Run the pipeline
df_segmented, n_segments = market_segmentation_pipeline()
```

This comprehensive guide covers hierarchical clustering from basic concepts to advanced implementations and real-world applications.