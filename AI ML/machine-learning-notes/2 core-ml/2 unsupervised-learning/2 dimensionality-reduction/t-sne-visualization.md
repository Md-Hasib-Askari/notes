# t-SNE Visualization

## Overview
t-Distributed Stochastic Neighbor Embedding (t-SNE) is a non-linear dimensionality reduction technique primarily used for visualization of high-dimensional data. It excels at preserving local structure and revealing clusters in complex datasets.

## Non-linear Dimensionality Reduction
- **Probability distributions**: Models similarities as probabilities in high and low dimensions
- **Kullback-Leibler divergence**: Minimizes difference between probability distributions
- **Gradient descent optimization**: Iteratively improves low-dimensional embedding
- **Local structure preservation**: Maintains relationships between nearby points
- **Non-linear mapping**: Can capture complex, non-linear structures unlike PCA

## Mathematical Foundation

### Core Algorithm
```
1. Compute pairwise similarities in high-dimensional space using Gaussian distribution
2. Compute pairwise similarities in low-dimensional space using t-distribution
3. Minimize KL divergence between the two probability distributions
4. Use gradient descent to optimize low-dimensional coordinates
```

### Mathematical Details
```
High-dimensional similarities:
p(j|i) = exp(-||xi - xj||²/2σi²) / Σk≠i exp(-||xi - xk||²/2σi²)
p(ij) = (p(j|i) + p(i|j)) / 2n

Low-dimensional similarities:
q(ij) = (1 + ||yi - yj||²)^(-1) / Σk≠l (1 + ||yk - yl||²)^(-1)

Cost function (KL divergence):
C = Σi Σj p(ij) log(p(ij)/q(ij))
```

## Python Implementation

### Basic t-SNE Implementation
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits, load_iris, make_swiss_roll
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Load sample datasets
digits = load_digits()
iris = load_iris()

print(f"Digits dataset shape: {digits.data.shape}")
print(f"Iris dataset shape: {iris.data.shape}")

# Basic t-SNE application
def basic_tsne_example():
    # Use digits dataset
    X, y = digits.data, digits.target
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    
    # Visualize results
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter)
    plt.title('t-SNE Visualization of Digits Dataset')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    
    # Add legend for digit classes
    for i in range(10):
        plt.scatter([], [], c=plt.cm.tab10(i/9), label=f'Digit {i}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
    return X_tsne, tsne

X_tsne, tsne_model = basic_tsne_example()
```

### Comparing with PCA
```python
from sklearn.decomposition import PCA

def compare_pca_tsne(X, y, dataset_name="Dataset"):
    """Compare PCA and t-SNE visualizations"""
    
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # PCA plot
    scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
    ax1.set_title(f'PCA - {dataset_name}')
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    
    # t-SNE plot
    scatter2 = ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.7)
    ax2.set_title(f't-SNE - {dataset_name}')
    ax2.set_xlabel('t-SNE Component 1')
    ax2.set_ylabel('t-SNE Component 2')
    
    plt.tight_layout()
    plt.show()
    
    return X_pca, X_tsne

# Compare on digits dataset
X_pca, X_tsne = compare_pca_tsne(digits.data, digits.target, "Digits")

# Compare on iris dataset
X_pca_iris, X_tsne_iris = compare_pca_tsne(iris.data, iris.target, "Iris")
```

## Parameters and Their Effects

### Perplexity Analysis
```python
def analyze_perplexity_effect(X, y, perplexity_values=[5, 15, 30, 50, 100]):
    """Analyze the effect of different perplexity values"""
    
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    fig, axes = plt.subplots(1, len(perplexity_values), figsize=(20, 4))
    
    for i, perp in enumerate(perplexity_values):
        # Apply t-SNE with different perplexity
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42, 
                   max_iter=1000, verbose=0)
        X_tsne = tsne.fit_transform(X_scaled)
        
        # Plot results
        scatter = axes[i].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, 
                                cmap='tab10', alpha=0.7, s=20)
        axes[i].set_title(f'Perplexity = {perp}')
        axes[i].set_xlabel('t-SNE 1')
        axes[i].set_ylabel('t-SNE 2')
    
    plt.suptitle('Effect of Perplexity on t-SNE Results')
    plt.tight_layout()
    plt.show()

# Analyze perplexity effect on digits dataset (subset for speed)
subset_idx = np.random.choice(len(digits.data), 500, replace=False)
X_subset = digits.data[subset_idx]
y_subset = digits.target[subset_idx]

analyze_perplexity_effect(X_subset, y_subset)
```

### Learning Rate and Iterations
```python
def analyze_learning_params(X, y, learning_rates=[10, 100, 200, 1000], 
                          max_iters=[250, 500, 1000, 2000]):
    """Analyze learning rate and iteration effects"""
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Test learning rates
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Learning rate analysis
    for i, lr in enumerate(learning_rates):
        tsne = TSNE(n_components=2, perplexity=30, learning_rate=lr, 
                   random_state=42, max_iter=1000)
        X_tsne = tsne.fit_transform(X_scaled)
        
        axes[0, i].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.7)
        axes[0, i].set_title(f'Learning Rate = {lr}')
        axes[0, i].set_xlabel('t-SNE 1')
        axes[0, i].set_ylabel('t-SNE 2')
    
    # Iteration analysis
    for i, max_iter in enumerate(max_iters):
        tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, 
                   max_iter=max_iter, random_state=42)
        X_tsne = tsne.fit_transform(X_scaled)
        
        axes[1, i].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.7)
        axes[1, i].set_title(f'Max Iterations = {max_iter}')
        axes[1, i].set_xlabel('t-SNE 1')
        axes[1, i].set_ylabel('t-SNE 2')
    
    plt.tight_layout()
    plt.show()

# Analyze learning parameters (using subset for speed)
analyze_learning_params(X_subset, y_subset)
```

### Parameter Guidelines
```python
def parameter_guidelines():
    """Print guidelines for t-SNE parameter selection"""
    
    guidelines = {
        "Perplexity": {
            "Range": "5-50 (typical: 20-30)",
            "Small values (5-15)": "Focus on very local structure, may create artificial clusters",
            "Medium values (20-40)": "Good balance, most commonly used",
            "Large values (50+)": "Focus on global structure, may merge distinct clusters",
            "Rule of thumb": "Should be smaller than number of data points"
        },
        
        "Learning Rate": {
            "Range": "10-1000 (typical: 200)",
            "Too low (10-50)": "Slow convergence, may get stuck in poor local minimum",
            "Good range (100-500)": "Reasonable convergence speed",
            "Too high (1000+)": "May overshoot optimal solution, unstable results"
        },
        
        "Max Iterations": {
            "Range": "250-5000 (typical: 1000)",
            "Too few (<250)": "Incomplete optimization, poor results",
            "Sufficient (1000)": "Usually enough for convergence",
            "Many (>2000)": "Longer computation, minimal improvement"
        },
        
        "Early Exaggeration": {
            "Default": "12.0",
            "Purpose": "Helps separate clusters in early stages",
            "Higher values": "More separated clusters initially"
        }
    }
    
    for param, info in guidelines.items():
        print(f"\n{param}:")
        print("-" * (len(param) + 1))
        for key, value in info.items():
            print(f"{key}: {value}")

parameter_guidelines()
```

## Advanced Applications

### Multi-class Visualization
```python
def advanced_multiclass_visualization():
    """Create publication-ready t-SNE visualization"""
    
    # Load digits dataset
    X, y = digits.data, digits.target
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, 
               max_iter=1000, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    
    # Create beautiful visualization
    plt.figure(figsize=(12, 10))
    
    # Custom color palette
    colors = plt.cm.Set3(np.linspace(0, 1, 10))
    
    # Plot each class separately for better control
    for i in range(10):
        mask = y == i
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                   c=[colors[i]], label=f'Digit {i}', 
                   alpha=0.7, s=30, edgecolors='black', linewidth=0.5)
    
    plt.title('t-SNE Visualization of Handwritten Digits', fontsize=16, fontweight='bold')
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, 
              fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return X_tsne

X_tsne_advanced = advanced_multiclass_visualization()
```

### Time Series Data Visualization
```python
def tsne_time_series_analysis():
    """Apply t-SNE to time series data for pattern discovery"""
    
    # Generate sample time series data
    from sklearn.datasets import make_classification
    
    # Create time series-like features
    np.random.seed(42)
    n_samples = 300
    n_timesteps = 50
    
    # Generate different types of patterns
    patterns = []
    labels = []
    
    for i in range(3):  # 3 different pattern types
        for j in range(n_samples // 3):
            if i == 0:  # Increasing trend
                pattern = np.cumsum(np.random.normal(0.1, 0.1, n_timesteps))
            elif i == 1:  # Oscillating pattern
                t = np.linspace(0, 4*np.pi, n_timesteps)
                pattern = np.sin(t) + 0.1 * np.random.normal(0, 1, n_timesteps)
            else:  # Random walk
                pattern = np.cumsum(np.random.normal(0, 0.1, n_timesteps))
            
            patterns.append(pattern)
            labels.append(i)
    
    X_ts = np.array(patterns)
    y_ts = np.array(labels)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne_ts = tsne.fit_transform(X_ts)
    
    # Visualize
    plt.figure(figsize=(15, 5))
    
    # Original time series samples
    plt.subplot(1, 3, 1)
    for i in range(3):
        mask = y_ts == i
        for j, pattern in enumerate(X_ts[mask][:5]):  # Show 5 examples per class
            plt.plot(pattern, alpha=0.7, color=f'C{i}')
    plt.title('Sample Time Series by Class')
    plt.xlabel('Time')
    plt.ylabel('Value')
    
    # t-SNE visualization
    plt.subplot(1, 3, 2)
    scatter = plt.scatter(X_tsne_ts[:, 0], X_tsne_ts[:, 1], c=y_ts, 
                         cmap='viridis', alpha=0.7)
    plt.title('t-SNE of Time Series Patterns')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.colorbar(scatter)
    
    # Class distribution
    plt.subplot(1, 3, 3)
    unique, counts = np.unique(y_ts, return_counts=True)
    plt.bar(['Trend', 'Oscillating', 'Random Walk'], counts, 
           color=['C0', 'C1', 'C2'], alpha=0.7)
    plt.title('Class Distribution')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.show()
    
    return X_tsne_ts

X_tsne_ts = tsne_time_series_analysis()
```

### Feature Importance via t-SNE
```python
def feature_importance_tsne():
    """Use t-SNE to understand feature importance"""
    
    # Load iris dataset for interpretability
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    
    # Apply t-SNE to full dataset
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    tsne_full = TSNE(n_components=2, perplexity=15, random_state=42)
    X_tsne_full = tsne_full.fit_transform(X_scaled)
    
    # Test feature importance by dropping features one by one
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    # Full dataset
    scatter = axes[0].scatter(X_tsne_full[:, 0], X_tsne_full[:, 1], 
                            c=y, cmap='viridis', alpha=0.7)
    axes[0].set_title('All Features')
    
    # Drop each feature and see effect
    for i, feature_name in enumerate(feature_names):
        # Create dataset without feature i
        X_dropped = np.delete(X_scaled, i, axis=1)
        
        # Apply t-SNE
        tsne_dropped = TSNE(n_components=2, perplexity=15, random_state=42)
        X_tsne_dropped = tsne_dropped.fit_transform(X_dropped)
        
        # Plot
        axes[i+1].scatter(X_tsne_dropped[:, 0], X_tsne_dropped[:, 1], 
                         c=y, cmap='viridis', alpha=0.7)
        axes[i+1].set_title(f'Without {feature_name}')
    
    # Add final subplot showing feature correlations
    axes[5].remove()
    ax_corr = fig.add_subplot(2, 3, 6)
    
    # Correlation heatmap
    import pandas as pd
    df = pd.DataFrame(X, columns=feature_names)
    correlation_matrix = df.corr()
    im = ax_corr.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
    ax_corr.set_xticks(range(len(feature_names)))
    ax_corr.set_yticks(range(len(feature_names)))
    ax_corr.set_xticklabels(feature_names, rotation=45)
    ax_corr.set_yticklabels(feature_names)
    ax_corr.set_title('Feature Correlations')
    
    # Add colorbar
    plt.colorbar(im, ax=ax_corr)
    
    plt.tight_layout()
    plt.show()

feature_importance_tsne()
```

## Interpreting Results

### Cluster Quality Assessment
```python
def assess_cluster_quality(X_original, X_embedded, y_true):
    """Assess how well t-SNE preserves cluster structure"""
    
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    from sklearn.cluster import KMeans
    from scipy.spatial.distance import pdist, squareform
    
    # Perform clustering on embedded data
    n_clusters = len(np.unique(y_true))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    y_pred_embedded = kmeans.fit_predict(X_embedded)
    
    # Perform clustering on original data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_original)
    kmeans_orig = KMeans(n_clusters=n_clusters, random_state=42)
    y_pred_original = kmeans_orig.fit_predict(X_scaled)
    
    # Calculate metrics
    ari_embedded = adjusted_rand_score(y_true, y_pred_embedded)
    ari_original = adjusted_rand_score(y_true, y_pred_original)
    
    sil_embedded = silhouette_score(X_embedded, y_true)
    sil_original = silhouette_score(X_scaled, y_true)
    
    # Distance preservation
    dist_original = pdist(X_scaled[:100])  # Subset for computational efficiency
    dist_embedded = pdist(X_embedded[:100])
    
    # Correlation between distance matrices
    dist_correlation = np.corrcoef(dist_original, dist_embedded)[0, 1]
    
    print("Cluster Quality Assessment:")
    print("=" * 40)
    print(f"Adjusted Rand Index (embedded): {ari_embedded:.3f}")
    print(f"Adjusted Rand Index (original): {ari_original:.3f}")
    print(f"Silhouette Score (embedded): {sil_embedded:.3f}")
    print(f"Silhouette Score (original): {sil_original:.3f}")
    print(f"Distance correlation: {dist_correlation:.3f}")
    
    return {
        'ari_embedded': ari_embedded,
        'ari_original': ari_original,
        'sil_embedded': sil_embedded,
        'sil_original': sil_original,
        'dist_correlation': dist_correlation
    }

# Assess quality for digits dataset
quality_metrics = assess_cluster_quality(digits.data, X_tsne, digits.target)
```

### Interactive Visualization
```python
def create_interactive_tsne():
    """Create interactive t-SNE visualization with plotly"""
    
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Apply t-SNE
        X, y = digits.data, digits.target
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        X_tsne = tsne.fit_transform(X_scaled)
        
        # Create interactive plot
        fig = px.scatter(
            x=X_tsne[:, 0], y=X_tsne[:, 1], 
            color=y.astype(str),
            title="Interactive t-SNE Visualization of Digits",
            labels={'color': 'Digit Class'},
            hover_data={'x': X_tsne[:, 0], 'y': X_tsne[:, 1]}
        )
        
        fig.update_layout(
            width=800, height=600,
            xaxis_title="t-SNE Component 1",
            yaxis_title="t-SNE Component 2"
        )
        
        # Note: This would show the interactive plot in a Jupyter notebook
        # fig.show()
        
        print("Interactive plot created (would display in Jupyter notebook)")
        
    except ImportError:
        print("Plotly not available. Install with: pip install plotly")
        
        # Alternative: Create enhanced matplotlib plot
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, 
                            cmap='tab10', alpha=0.7, s=50)
        
        # Add annotations for some points
        for i in range(0, len(X_tsne), 100):  # Annotate every 100th point
            plt.annotate(f'{y[i]}', (X_tsne[i, 0], X_tsne[i, 1]), 
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, alpha=0.7)
        
        plt.colorbar(scatter)
        plt.title('Enhanced t-SNE Visualization with Annotations')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.show()

create_interactive_tsne()
```

## Best Practices and Guidelines

### Data Preprocessing
```python
def tsne_preprocessing_guide():
    """Guide for proper data preprocessing before t-SNE"""
    
    print("t-SNE Preprocessing Best Practices:")
    print("=" * 50)
    
    # Example with different preprocessing approaches
    X, y = digits.data, digits.target
    
    preprocessing_methods = {
        'Raw': X,
        'StandardScaler': StandardScaler().fit_transform(X),
        'MinMaxScaler': MinMaxScaler().fit_transform(X),
        'RobustScaler': RobustScaler().fit_transform(X)
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, (method, X_processed) in enumerate(preprocessing_methods.items()):
        # Apply t-SNE
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, verbose=0)
        X_tsne = tsne.fit_transform(X_processed[:500])  # Subset for speed
        
        # Plot
        scatter = axes[i].scatter(X_tsne[:, 0], X_tsne[:, 1], 
                                c=y[:500], cmap='tab10', alpha=0.7)
        axes[i].set_title(f'Preprocessing: {method}')
        axes[i].set_xlabel('t-SNE 1')
        axes[i].set_ylabel('t-SNE 2')
    
    plt.tight_layout()
    plt.show()
    
    print("\nRecommendations:")
    print("1. StandardScaler: Most common choice, centers and scales features")
    print("2. RobustScaler: Better for data with outliers")
    print("3. MinMaxScaler: Use when preserving zero values is important")
    print("4. Raw data: Only if features are already on similar scales")

from sklearn.preprocessing import MinMaxScaler, RobustScaler
tsne_preprocessing_guide()
```

### Performance Optimization
```python
def optimize_tsne_performance():
    """Tips for optimizing t-SNE performance"""
    
    print("t-SNE Performance Optimization:")
    print("=" * 40)
    
    # Demonstrate different optimization strategies
    X, y = digits.data, digits.target
    
    strategies = [
        ("Full Dataset", X),
        ("PCA Preprocessing", PCA(n_components=50).fit_transform(X)),
        ("Random Subset", X[np.random.choice(len(X), 1000, replace=False)]),
    ]
    
    import time
    
    for name, data in strategies:
        start_time = time.time()
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data)
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, verbose=0)
        X_tsne = tsne.fit_transform(X_scaled)
        
        end_time = time.time()
        
        print(f"{name}:")
        print(f"  Data shape: {data.shape}")
        print(f"  Time taken: {end_time - start_time:.2f} seconds")
        print()
    
    print("Optimization Tips:")
    print("1. Use PCA preprocessing to reduce dimensionality first")
    print("2. Sample large datasets (>5000 samples)")
    print("3. Adjust perplexity based on sample size")
    print("4. Use early_exaggeration for better separation")
    print("5. Consider batch processing for very large datasets")

optimize_tsne_performance()
```

## Applications and Use Cases

### Document Clustering
```python
def document_clustering_example():
    """Example of using t-SNE for document visualization"""
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.datasets import fetch_20newsgroups
    
    # Sample documents (in practice, you'd load your own corpus)
    documents = [
        "Machine learning algorithms can learn patterns from data",
        "Deep neural networks are powerful for image recognition",
        "Natural language processing helps computers understand text",
        "Computer vision enables machines to see and interpret images",
        "Reinforcement learning agents learn through trial and error",
        "Supervised learning uses labeled training data",
        "The stock market showed significant volatility today",
        "Interest rates affect mortgage and loan decisions",
        "Economic indicators suggest potential recession ahead",
        "Investment portfolios should be diversified across sectors",
        "Financial markets react to global political events",
        "Banking regulations impact lending practices"
    ]
    
    # Document labels
    doc_labels = ['ML'] * 6 + ['Finance'] * 6
    
    # Convert to TF-IDF vectors
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    X_tfidf = vectorizer.fit_transform(documents).toarray()
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=3, random_state=42)  # Small perplexity for small dataset
    X_tsne = tsne.fit_transform(X_tfidf)
    
    # Visualize
    plt.figure(figsize=(10, 8))
    colors = {'ML': 'blue', 'Finance': 'red'}
    for label in ['ML', 'Finance']:
        mask = np.array(doc_labels) == label
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                   c=colors[label], label=label, alpha=0.7, s=100)
    
    # Add document annotations
    for i, doc in enumerate(documents):
        plt.annotate(f"Doc {i+1}", (X_tsne[i, 0], X_tsne[i, 1]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.title('t-SNE Visualization of Document Clusters')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("Document clustering reveals thematic groups in text collections")

document_clustering_example()
```

### Gene Expression Analysis
```python
def gene_expression_example():
    """Simulate gene expression analysis with t-SNE"""
    
    # Simulate gene expression data
    np.random.seed(42)
    n_samples = 200
    n_genes = 1000
    
    # Create three cell types with different expression patterns
    cell_types = []
    expression_data = []
    
    for cell_type in range(3):
        for sample in range(n_samples // 3):
            # Each cell type has different gene expression signature
            if cell_type == 0:  # Cell type A
                expression = np.random.normal(5, 1, n_genes)
                expression[:100] += 3  # Higher expression for first 100 genes
            elif cell_type == 1:  # Cell type B
                expression = np.random.normal(5, 1, n_genes)
                expression[100:200] += 3  # Higher expression for genes 100-200
            else:  # Cell type C
                expression = np.random.normal(5, 1, n_genes)
                expression[200:300] += 3  # Higher expression for genes 200-300
            
            expression_data.append(expression)
            cell_types.append(cell_type)
    
    X_genes = np.array(expression_data)
    y_cells = np.array(cell_types)
    
    # Apply t-SNE
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_genes)
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    
    # Visualize
    plt.figure(figsize=(12, 8))
    cell_type_names = ['Neuron', 'Immune', 'Epithelial']
    colors = ['red', 'blue', 'green']
    
    for i, cell_type in enumerate(['Neuron', 'Immune', 'Epithelial']):
        mask = y_cells == i
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                   c=colors[i], label=cell_type, alpha=0.7, s=50)
    
    plt.title('t-SNE Visualization of Single-Cell Gene Expression')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("t-SNE effectively separates cell types based on gene expression profiles")

gene_expression_example()
```

## Common Pitfalls and Solutions

### Issue 1: Misinterpreting Distances
```python
def distance_interpretation_warning():
    """Demonstrate why distances in t-SNE can be misleading"""
    
    # Create simple 3D data
    np.random.seed(42)
    theta = np.linspace(0, 4*np.pi, 100)
    x = np.cos(theta)
    y = np.sin(theta)
    z = theta / (4*np.pi)
    
    X_3d = np.column_stack([x, y, z])
    colors = theta
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=15, random_state=42)
    X_tsne = tsne.fit_transform(X_3d)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Original 3D structure (projected to 2D for visualization)
    scatter1 = ax1.scatter(x, y, c=colors, cmap='viridis', alpha=0.7)
    ax1.set_title('Original Data (3D spiral projected to 2D)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    
    # t-SNE result
    scatter2 = ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors, cmap='viridis', alpha=0.7)
    ax2.set_title('t-SNE Result')
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    
    plt.colorbar(scatter1, ax=ax1)
    plt.colorbar(scatter2, ax=ax2)
    plt.tight_layout()
    plt.show()
    
    print("WARNING: In t-SNE results:")
    print("- Distances between clusters are NOT meaningful")
    print("- Cluster sizes are NOT meaningful")
    print("- Only local neighborhood structure is preserved")
    print("- Different runs may produce different global arrangements")

distance_interpretation_warning()
```

### Issue 2: Hyperparameter Sensitivity
```python
def hyperparameter_sensitivity_demo():
    """Show how sensitive t-SNE is to hyperparameters"""
    
    # Use a subset of digits for speed
    X, y = digits.data[:500], digits.target[:500]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Test different random seeds with same parameters
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Different random seeds
    for i, seed in enumerate([42, 123, 456]):
        tsne = TSNE(n_components=2, perplexity=30, random_state=seed)
        X_tsne = tsne.fit_transform(X_scaled)
        
        axes[0, i].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.7)
        axes[0, i].set_title(f'Random Seed = {seed}')
    
    # Different perplexity values
    for i, perp in enumerate([10, 30, 50]):
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
        X_tsne = tsne.fit_transform(X_scaled)
        
        axes[1, i].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.7)
        axes[1, i].set_title(f'Perplexity = {perp}')
    
    plt.suptitle('t-SNE Sensitivity to Hyperparameters', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    print("Key Points:")
    print("- Different random seeds can produce very different layouts")
    print("- Always run multiple times with different seeds")
    print("- Focus on local structure, not global positioning")
    print("- Perplexity significantly affects cluster formation")

hyperparameter_sensitivity_demo()
```

### Issue 3: Computational Complexity
```python
def complexity_analysis():
    """Analyze computational complexity with dataset size"""
    
    print("t-SNE Computational Complexity Analysis:")
    print("=" * 50)
    
    # Test with increasing dataset sizes
    sizes = [100, 500, 1000, 2000]
    times = []
    
    for size in sizes:
        # Create sample data
        X_sample = digits.data[:size]
        
        # Time the t-SNE computation
        start_time = time.time()
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_sample)
        
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, verbose=0)
        tsne.fit_transform(X_scaled)
        
        end_time = time.time()
        computation_time = end_time - start_time
        times.append(computation_time)
        
        print(f"Size: {size:4d} samples, Time: {computation_time:6.2f} seconds")
    
    # Plot complexity
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Dataset Size')
    plt.ylabel('Computation Time (seconds)')
    plt.title('t-SNE Computational Complexity')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("\nComplexity Notes:")
    print("- t-SNE has O(n²) complexity in the number of samples")
    print("- Consider dimensionality reduction (PCA) for high-dimensional data")
    print("- Use sampling for datasets > 5000 samples")
    print("- Approximate methods available for very large datasets")

import time
complexity_analysis()
```

## Alternatives and Comparisons

### UMAP Comparison
```python
def compare_tsne_umap():
    """Compare t-SNE with UMAP (if available)"""
    
    try:
        import umap
        
        # Prepare data
        X, y = digits.data, digits.target
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply both methods
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        X_tsne = tsne.fit_transform(X_scaled)
        
        umap_model = umap.UMAP(n_components=2, random_state=42)
        X_umap = umap_model.fit_transform(X_scaled)
        
        # Compare results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        scatter1 = ax1.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.7)
        ax1.set_title('t-SNE')
        ax1.set_xlabel('Component 1')
        ax1.set_ylabel('Component 2')
        
        scatter2 = ax2.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='tab10', alpha=0.7)
        ax2.set_title('UMAP')
        ax2.set_xlabel('Component 1')
        ax2.set_ylabel('Component 2')
        
        plt.tight_layout()
        plt.show()
        
        print("t-SNE vs UMAP Comparison:")
        print("t-SNE: Better for local structure, slower")
        print("UMAP: Better for global structure, faster, preserves more structure")
        
    except ImportError:
        print("UMAP not available. Install with: pip install umap-learn")
        print("\nComparison would show:")
        print("- t-SNE: Excellent local structure preservation")
        print("- UMAP: Better global structure + faster computation")
        print("- Both: Non-linear dimensionality reduction")

compare_tsne_umap()
```

## Advantages and Limitations

### Comprehensive Comparison
```python
def tsne_advantages_limitations():
    """Detailed analysis of t-SNE strengths and weaknesses"""
    
    comparison_data = {
        "Advantages": [
            "Excellent at revealing clusters in high-dimensional data",
            "Preserves local neighborhood structure very well",
            "Non-linear dimensionality reduction",
            "Intuitive 2D/3D visualizations",
            "Works well with various data types",
            "Robust to noise in many cases",
            "Can reveal hidden patterns invisible in original space"
        ],
        
        "Limitations": [
            "Computationally expensive (O(n²) complexity)",
            "Non-deterministic (different runs give different results)",
            "Hyperparameter sensitive (perplexity, learning rate)",
            "Distances between clusters not meaningful",
            "Cannot embed new points (no out-of-sample extension)",
            "May create artificial clusters",
            "Cluster sizes not meaningful",
            "Slow convergence for large datasets"
        ],
        
        "Best Use Cases": [
            "Exploratory data analysis",
            "Cluster visualization",
            "Feature engineering insights",
            "Pattern discovery in complex data",
            "Biological data analysis (gene expression, etc.)",
            "Image analysis and computer vision",
            "Text mining and document analysis"
        ],
        
        "When NOT to Use": [
            "When you need to embed new data points",
            "When interpretability of dimensions is crucial",
            "For very large datasets (>10,000 samples)",
            "When computational resources are limited",
            "When reproducibility is critical",
            "For real-time applications"
        ]
    }
    
    for category, items in comparison_data.items():
        print(f"\n{category}:")
        print("=" * (len(category) + 1))
        for i, item in enumerate(items, 1):
            print(f"{i}. {item}")

tsne_advantages_limitations()
```

## Learning Objectives
- [ ] **Understand t-SNE theory**: Master probability distributions and KL divergence minimization
- [ ] **Apply t-SNE effectively**: Use appropriate parameters for different data types
- [ ] **Visualize high-dimensional data**: Create meaningful 2D projections of complex datasets
- [ ] **Interpret results correctly**: Understand what t-SNE shows and what it doesn't
- [ ] **Handle parameters properly**: Select perplexity, learning rate, and iterations appropriately
- [ ] **Preprocess data correctly**: Apply proper scaling and dimensionality reduction
- [ ] **Avoid common pitfalls**: Understand limitations and misinterpretation risks
- [ ] **Compare with alternatives**: Know when to use t-SNE vs PCA, UMAP, or other methods
- [ ] **Apply to real problems**: Use t-SNE for cluster analysis, feature exploration, and visualization
- [ ] **Optimize performance**: Handle large datasets and computational constraints effectively

## Practice Exercises
1. Implement basic t-SNE parameter tuning for the digits dataset
2. Compare t-SNE with PCA on high-dimensional data
3. Apply t-SNE to image data and analyze cluster formation
4. Create interactive visualizations with hover information
5. Use t-SNE for anomaly detection in multivariate data
6. Analyze the effect of different preprocessing methods
7. Build a pipeline combining PCA preprocessing with t-SNE
8. Apply t-SNE to text data using TF-IDF features
9. Create publication-ready visualizations with proper annotations
10. Develop guidelines for parameter selection based on dataset characteristics