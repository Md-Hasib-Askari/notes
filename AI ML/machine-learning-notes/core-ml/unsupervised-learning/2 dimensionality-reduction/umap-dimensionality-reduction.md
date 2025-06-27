# UMAP Dimensionality Reduction

## Overview
Uniform Manifold Approximation and Projection (UMAP) is a non-linear dimensionality reduction technique that preserves both local and global structure better than t-SNE while being computationally more efficient. It's based on manifold learning and topological data analysis principles.

## Uniform Manifold Approximation
- **Topological data analysis**: Uses concepts from algebraic topology and differential geometry
- **Manifold learning**: Assumes data lies on a lower-dimensional manifold embedded in high-dimensional space
- **Preservation of structure**: Maintains both local neighborhoods and global topology
- **Mathematical foundation**: Based on Riemannian geometry and fuzzy set theory

## How UMAP Works

### Core Algorithm
```
1. Construct a weighted k-nearest neighbor graph in high-dimensional space
2. Transform this into a fuzzy topological representation
3. Initialize low-dimensional embedding randomly
4. Optimize embedding to match the fuzzy topological structure
5. Use stochastic gradient descent for optimization
```

### Mathematical Foundation
```
High-dimensional fuzzy set membership:
μ(x,y) = exp(-max(0, d(x,y) - ρ) / σ)

Low-dimensional probability:
ψ(a,b) = 1 / (1 + a||yi - yj||²)^b

Objective: Minimize cross-entropy between high and low-dimensional representations
```

## Python Implementation

### Basic UMAP Usage
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits, load_iris, make_swiss_roll
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Install UMAP first: pip install umap-learn
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    print("UMAP not installed. Run: pip install umap-learn")
    UMAP_AVAILABLE = False

if UMAP_AVAILABLE:
    # Load sample datasets
    digits = load_digits()
    iris = load_iris()
    
    def basic_umap_example():
        """Basic UMAP application on digits dataset"""
        
        X, y = digits.data, digits.target
        
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply UMAP
        reducer = umap.UMAP(n_components=2, random_state=42)
        X_umap = reducer.fit_transform(X_scaled)
        
        # Visualize results
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='tab10', alpha=0.7)
        plt.colorbar(scatter)
        plt.title('UMAP Visualization of Digits Dataset')
        plt.xlabel('UMAP Component 1')
        plt.ylabel('UMAP Component 2')
        
        # Add legend for digit classes
        for i in range(10):
            plt.scatter([], [], c=plt.cm.tab10(i/9), label=f'Digit {i}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
        
        return X_umap, reducer
    
    X_umap, umap_model = basic_umap_example()
else:
    print("Simulating UMAP results...")
    # Placeholder for when UMAP is not available
    X_umap = np.random.randn(len(digits.data), 2)
```

### UMAP Parameters and Their Effects
```python
if UMAP_AVAILABLE:
    def analyze_umap_parameters():
        """Analyze the effect of different UMAP parameters"""
        
        X, y = digits.data[:500], digits.target[:500]  # Subset for speed
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Test different parameter combinations
        param_combinations = [
            {'n_neighbors': 5, 'min_dist': 0.1, 'title': 'n_neighbors=5, min_dist=0.1'},
            {'n_neighbors': 15, 'min_dist': 0.1, 'title': 'n_neighbors=15, min_dist=0.1'},
            {'n_neighbors': 50, 'min_dist': 0.1, 'title': 'n_neighbors=50, min_dist=0.1'},
            {'n_neighbors': 15, 'min_dist': 0.01, 'title': 'n_neighbors=15, min_dist=0.01'},
            {'n_neighbors': 15, 'min_dist': 0.5, 'title': 'n_neighbors=15, min_dist=0.5'},
            {'n_neighbors': 15, 'min_dist': 0.9, 'title': 'n_neighbors=15, min_dist=0.9'}
        ]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for i, params in enumerate(param_combinations):
            # Apply UMAP with specific parameters
            reducer = umap.UMAP(
                n_components=2, 
                n_neighbors=params['n_neighbors'],
                min_dist=params['min_dist'],
                random_state=42
            )
            X_umap = reducer.fit_transform(X_scaled)
            
            # Plot results
            scatter = axes[i].scatter(X_umap[:, 0], X_umap[:, 1], c=y, 
                                    cmap='tab10', alpha=0.7, s=20)
            axes[i].set_title(params['title'])
            axes[i].set_xlabel('UMAP 1')
            axes[i].set_ylabel('UMAP 2')
        
        plt.suptitle('Effect of UMAP Parameters', fontsize=16)
        plt.tight_layout()
        plt.show()
        
        print("Parameter Guidelines:")
        print("n_neighbors: Controls local vs global structure")
        print("  - Small (5-15): Focus on local structure")
        print("  - Large (50+): Focus on global structure")
        print("min_dist: Controls how tightly packed points are")
        print("  - Small (0.01): Very tight clusters")
        print("  - Large (0.9): More spread out points")
    
    analyze_umap_parameters()
```

### Hyperparameter Tuning
```python
if UMAP_AVAILABLE:
    def comprehensive_parameter_analysis():
        """Comprehensive analysis of UMAP hyperparameters"""
        
        X, y = iris.data, iris.target
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Parameter ranges to test
        n_neighbors_range = [3, 5, 10, 15, 30, 50]
        min_dist_range = [0.0, 0.1, 0.25, 0.5, 0.8, 0.99]
        
        # Create comprehensive grid
        fig, axes = plt.subplots(len(min_dist_range), len(n_neighbors_range), 
                               figsize=(20, 20))
        
        for i, min_dist in enumerate(min_dist_range):
            for j, n_neighbors in enumerate(n_neighbors_range):
                reducer = umap.UMAP(
                    n_components=2,
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    random_state=42
                )
                X_umap = reducer.fit_transform(X_scaled)
                
                # Plot
                axes[i, j].scatter(X_umap[:, 0], X_umap[:, 1], c=y, 
                                 cmap='viridis', alpha=0.7)
                axes[i, j].set_title(f'n_neighbors={n_neighbors}\nmin_dist={min_dist}')
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
        
        plt.suptitle('UMAP Parameter Grid Search - Iris Dataset', fontsize=16)
        plt.tight_layout()
        plt.show()
        
        # Best practice recommendations
        print("\nBest Practice Parameter Selection:")
        print("=" * 50)
        print("n_neighbors:")
        print("  - Small datasets (n<1000): 10-15")
        print("  - Medium datasets (1000<n<10000): 15-30") 
        print("  - Large datasets (n>10000): 30-100")
        print("\nmin_dist:")
        print("  - Tight clusters: 0.0-0.1")
        print("  - Balanced: 0.1-0.5")
        print("  - Spread out: 0.5-0.99")
        
    comprehensive_parameter_analysis()
```

## Advantages over t-SNE

### Computational Performance Comparison
```python
if UMAP_AVAILABLE:
    import time
    from sklearn.manifold import TSNE
    
    def performance_comparison():
        """Compare UMAP and t-SNE performance"""
        
        # Test on different dataset sizes
        sizes = [500, 1000, 2000]
        umap_times = []
        tsne_times = []
        
        for size in sizes:
            print(f"Testing size: {size}")
            
            # Create sample data
            X_sample = digits.data[:size]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_sample)
            
            # Time UMAP
            start_time = time.time()
            reducer = umap.UMAP(n_components=2, random_state=42, verbose=False)
            X_umap = reducer.fit_transform(X_scaled)
            umap_time = time.time() - start_time
            umap_times.append(umap_time)
            
            # Time t-SNE
            start_time = time.time()
            tsne = TSNE(n_components=2, random_state=42, verbose=0)
            X_tsne = tsne.fit_transform(X_scaled)
            tsne_time = time.time() - start_time
            tsne_times.append(tsne_time)
            
            print(f"  UMAP: {umap_time:.2f}s, t-SNE: {tsne_time:.2f}s")
        
        # Plot comparison
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        x_pos = np.arange(len(sizes))
        width = 0.35
        plt.bar(x_pos - width/2, umap_times, width, label='UMAP', alpha=0.7)
        plt.bar(x_pos + width/2, tsne_times, width, label='t-SNE', alpha=0.7)
        plt.xlabel('Dataset Size')
        plt.ylabel('Time (seconds)')
        plt.title('Performance Comparison')
        plt.xticks(x_pos, sizes)
        plt.legend()
        plt.yscale('log')
        
        plt.subplot(1, 2, 2)
        speedup = np.array(tsne_times) / np.array(umap_times)
        plt.plot(sizes, speedup, 'ro-', linewidth=2, markersize=8)
        plt.xlabel('Dataset Size')
        plt.ylabel('Speedup Factor (t-SNE time / UMAP time)')
        plt.title('UMAP Speedup over t-SNE')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"\nAverage speedup: {np.mean(speedup):.1f}x")
        
    performance_comparison()
```

### Structure Preservation Comparison
```python
if UMAP_AVAILABLE:
    def structure_preservation_comparison():
        """Compare how well UMAP and t-SNE preserve structure"""
        
        # Create data with known structure (Swiss roll)
        X_swiss, color = make_swiss_roll(n_samples=1000, noise=0.1, random_state=42)
        
        # Apply both methods
        umap_reducer = umap.UMAP(n_components=2, random_state=42)
        X_umap = umap_reducer.fit_transform(X_swiss)
        
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        X_tsne = tsne.fit_transform(X_swiss)
        
        # Visualize comparison
        fig = plt.figure(figsize=(18, 6))
        
        # Original 3D data
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(X_swiss[:, 0], X_swiss[:, 1], X_swiss[:, 2], c=color, cmap='viridis')
        ax1.set_title('Original Swiss Roll (3D)')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # UMAP result
        ax2 = fig.add_subplot(132)
        scatter2 = ax2.scatter(X_umap[:, 0], X_umap[:, 1], c=color, cmap='viridis')
        ax2.set_title('UMAP Embedding')
        ax2.set_xlabel('UMAP 1')
        ax2.set_ylabel('UMAP 2')
        
        # t-SNE result
        ax3 = fig.add_subplot(133)
        scatter3 = ax3.scatter(X_tsne[:, 0], X_tsne[:, 1], c=color, cmap='viridis')
        ax3.set_title('t-SNE Embedding')
        ax3.set_xlabel('t-SNE 1')
        ax3.set_ylabel('t-SNE 2')
        
        plt.tight_layout()
        plt.show()
        
        print("Structure Preservation Analysis:")
        print("- UMAP: Better preserves global structure and continuity")
        print("- t-SNE: Better at separating clusters but may break continuity")
        print("- UMAP: More suitable for continuous manifolds")
        print("- t-SNE: More suitable for discrete cluster visualization")
        
    structure_preservation_comparison()
```

## Implementation Details

### Custom UMAP Pipeline
```python
if UMAP_AVAILABLE:
    def create_umap_pipeline():
        """Create a comprehensive UMAP analysis pipeline"""
        
        class UMAPAnalyzer:
            def __init__(self, n_components=2, random_state=42):
                self.n_components = n_components
                self.random_state = random_state
                self.scaler = StandardScaler()
                self.reducer = None
                self.embedding = None
                
            def fit_transform(self, X, n_neighbors=15, min_dist=0.1, metric='euclidean'):
                """Fit UMAP and transform data"""
                
                # Standardize data
                X_scaled = self.scaler.fit_transform(X)
                
                # Initialize UMAP
                self.reducer = umap.UMAP(
                    n_components=self.n_components,
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    metric=metric,
                    random_state=self.random_state
                )
                
                # Fit and transform
                self.embedding = self.reducer.fit_transform(X_scaled)
                return self.embedding
            
            def transform_new(self, X_new):
                """Transform new data points using fitted model"""
                if self.reducer is None:
                    raise ValueError("Model must be fitted first")
                
                X_new_scaled = self.scaler.transform(X_new)
                return self.reducer.transform(X_new_scaled)
            
            def plot_embedding(self, y=None, title="UMAP Embedding", figsize=(10, 8)):
                """Plot the embedding with optional color coding"""
                
                if self.embedding is None:
                    raise ValueError("No embedding available. Run fit_transform first.")
                
                plt.figure(figsize=figsize)
                
                if y is not None:
                    scatter = plt.scatter(self.embedding[:, 0], self.embedding[:, 1], 
                                        c=y, cmap='tab10', alpha=0.7)
                    plt.colorbar(scatter)
                else:
                    plt.scatter(self.embedding[:, 0], self.embedding[:, 1], alpha=0.7)
                
                plt.title(title)
                plt.xlabel('UMAP Component 1')
                plt.ylabel('UMAP Component 2')
                plt.show()
            
            def parameter_search(self, X, y, param_grid):
                """Search for optimal parameters"""
                from sklearn.metrics import silhouette_score
                
                best_score = -1
                best_params = None
                results = []
                
                X_scaled = self.scaler.fit_transform(X)
                
                for n_neighbors in param_grid['n_neighbors']:
                    for min_dist in param_grid['min_dist']:
                        reducer = umap.UMAP(
                            n_components=2,
                            n_neighbors=n_neighbors,
                            min_dist=min_dist,
                            random_state=self.random_state
                        )
                        
                        embedding = reducer.fit_transform(X_scaled)
                        score = silhouette_score(embedding, y)
                        
                        results.append({
                            'n_neighbors': n_neighbors,
                            'min_dist': min_dist,
                            'silhouette_score': score
                        })
                        
                        if score > best_score:
                            best_score = score
                            best_params = {'n_neighbors': n_neighbors, 'min_dist': min_dist}
                
                return best_params, results
        
        # Example usage
        analyzer = UMAPAnalyzer()
        
        # Fit on digits data
        X_embedded = analyzer.fit_transform(digits.data)
        analyzer.plot_embedding(digits.target, "UMAP Analysis of Digits")
        
        # Parameter search
        param_grid = {
            'n_neighbors': [10, 15, 30],
            'min_dist': [0.1, 0.25, 0.5]
        }
        
        best_params, results = analyzer.parameter_search(
            digits.data[:500], digits.target[:500], param_grid
        )
        
        print(f"Best parameters: {best_params}")
        
        return analyzer
    
    analyzer = create_umap_pipeline()
```

### Different Distance Metrics
```python
if UMAP_AVAILABLE:
    def explore_distance_metrics():
        """Explore different distance metrics in UMAP"""
        
        X, y = iris.data, iris.target
        
        # Different metrics to test
        metrics = ['euclidean', 'manhattan', 'cosine', 'correlation']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            try:
                reducer = umap.UMAP(
                    n_components=2,
                    n_neighbors=15,
                    min_dist=0.1,
                    metric=metric,
                    random_state=42
                )
                
                X_umap = reducer.fit_transform(StandardScaler().fit_transform(X))
                
                scatter = axes[i].scatter(X_umap[:, 0], X_umap[:, 1], c=y, 
                                        cmap='viridis', alpha=0.7)
                axes[i].set_title(f'Metric: {metric}')
                axes[i].set_xlabel('UMAP 1')
                axes[i].set_ylabel('UMAP 2')
                
            except Exception as e:
                axes[i].text(0.5, 0.5, f'Error with {metric}:\n{str(e)}', 
                           transform=axes[i].transAxes, ha='center', va='center')
                axes[i].set_title(f'Metric: {metric} (Error)')
        
        plt.tight_layout()
        plt.show()
        
        print("Distance Metric Guidelines:")
        print("- Euclidean: Most common, works well for continuous features")
        print("- Manhattan: Better for high-dimensional sparse data")
        print("- Cosine: Good for text data and when magnitude doesn't matter")
        print("- Correlation: Focuses on pattern similarity rather than magnitude")
        
    explore_distance_metrics()
```

## Real-world Applications

### Single-cell RNA Analysis
```python
if UMAP_AVAILABLE:
    def simulate_scrna_analysis():
        """Simulate single-cell RNA analysis with UMAP"""
        
        # Simulate single-cell gene expression data
        np.random.seed(42)
        n_cells = 1000
        n_genes = 2000
        
        # Create different cell types with distinct expression patterns
        cell_types = []
        expression_data = []
        
        for cell_type in range(4):  # 4 cell types
            n_cells_type = n_cells // 4
            for cell in range(n_cells_type):
                # Each cell type has different expression signature
                base_expression = np.random.negative_binomial(5, 0.3, n_genes)
                
                if cell_type == 0:  # T cells
                    base_expression[0:100] += 20  # High immune markers
                elif cell_type == 1:  # B cells  
                    base_expression[100:200] += 15  # High antibody production
                elif cell_type == 2:  # Neurons
                    base_expression[200:300] += 25  # High neural markers
                else:  # Epithelial cells
                    base_expression[300:400] += 18  # High structural markers
                
                expression_data.append(base_expression)
                cell_types.append(cell_type)
        
        X_genes = np.array(expression_data)
        y_cells = np.array(cell_types)
        
        # Preprocessing typical for scRNA-seq
        # Log transformation and scaling
        X_log = np.log1p(X_genes)  # log(1 + x) transformation
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_log)
        
        # Apply UMAP
        reducer = umap.UMAP(
            n_components=2, 
            n_neighbors=30,
            min_dist=0.0,
            metric='correlation',  # Often better for gene expression
            random_state=42
        )
        X_umap = reducer.fit_transform(X_scaled)
        
        # Visualize
        plt.figure(figsize=(12, 8))
        cell_type_names = ['T cells', 'B cells', 'Neurons', 'Epithelial']
        colors = ['red', 'blue', 'green', 'orange']
        
        for i, cell_type in enumerate(cell_type_names):
            mask = y_cells == i
            plt.scatter(X_umap[mask, 0], X_umap[mask, 1], 
                       c=colors[i], label=cell_type, alpha=0.7, s=20)
        
        plt.title('UMAP Visualization of Single-Cell RNA Expression')
        plt.xlabel('UMAP Component 1')
        plt.ylabel('UMAP Component 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        print("scRNA-seq Analysis Notes:")
        print("- UMAP excels at revealing cell type clusters")
        print("- Preserves developmental trajectories better than t-SNE")
        print("- Correlation metric often works better than Euclidean")
        print("- Can reveal rare cell populations and transitions")
        
    simulate_scrna_analysis()
```

### Image Analysis and Computer Vision
```python
if UMAP_AVAILABLE:
    def image_feature_analysis():
        """Use UMAP for image feature visualization"""
        
        # Use digits dataset as example images
        X, y = digits.data, digits.target
        
        # Apply UMAP with different approaches
        approaches = [
            {'name': 'Raw Pixels', 'data': X},
            {'name': 'PCA Features', 'data': PCA(n_components=50).fit_transform(X)},
            {'name': 'Standardized', 'data': StandardScaler().fit_transform(X)}
        ]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, approach in enumerate(approaches):
            reducer = umap.UMAP(n_components=2, random_state=42)
            X_umap = reducer.fit_transform(approach['data'])
            
            scatter = axes[i].scatter(X_umap[:, 0], X_umap[:, 1], c=y, 
                                    cmap='tab10', alpha=0.7)
            axes[i].set_title(f'UMAP on {approach["name"]}')
            axes[i].set_xlabel('UMAP 1')
            axes[i].set_ylabel('UMAP 2')
        
        plt.tight_layout()
        plt.show()
        
        # Show sample images for each cluster
        plt.figure(figsize=(15, 8))
        for digit in range(10):
            plt.subplot(2, 5, digit + 1)
            digit_indices = np.where(y == digit)[0]
            sample_idx = digit_indices[0]
            plt.imshow(X[sample_idx].reshape(8, 8), cmap='gray')
            plt.title(f'Digit {digit}')
            plt.axis('off')
        
        plt.suptitle('Sample Images for Each Digit Class')
        plt.tight_layout()
        plt.show()
        
        print("Image Analysis with UMAP:")
        print("- Raw pixels: Direct analysis of pixel intensities")
        print("- PCA features: Dimensionality reduction before UMAP")
        print("- Standardized: Normalized pixel values")
        print("- UMAP can reveal visual similarity patterns")
        
    image_feature_analysis()
```

### Text Analysis and NLP
```python
if UMAP_AVAILABLE:
    def text_analysis_example():
        """UMAP for text document analysis"""
        
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Sample documents from different topics
        documents = [
            # Technology documents
            "Machine learning algorithms analyze data patterns",
            "Artificial intelligence transforms modern computing",
            "Deep learning networks process complex information",
            "Neural networks mimic brain functionality",
            "Computer vision enables automated image recognition",
            
            # Medical documents
            "Clinical trials test new pharmaceutical treatments",
            "Medical diagnosis requires careful patient examination",
            "Healthcare professionals provide essential patient care",
            "Surgical procedures require precision and expertise",
            "Medical research advances treatment methodologies",
            
            # Finance documents
            "Stock market volatility affects investment portfolios",
            "Financial planning requires long-term strategy",
            "Economic indicators predict market trends",
            "Banking regulations ensure financial stability",
            "Investment strategies diversify risk exposure",
            
            # Sports documents
            "Professional athletes train for competitive performance",
            "Olympic games showcase international athletic talent",
            "Team sports require coordination and strategy",
            "Athletic performance depends on training and nutrition",
            "Sports medicine helps prevent athlete injuries"
        ]
        
        # Document categories
        categories = (['Technology'] * 5 + ['Medical'] * 5 + 
                     ['Finance'] * 5 + ['Sports'] * 5)
        
        # Convert to TF-IDF features
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        X_tfidf = vectorizer.fit_transform(documents).toarray()
        
        # Apply UMAP
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=5,  # Small for small dataset
            min_dist=0.1,
            metric='cosine',  # Good for text data
            random_state=42
        )
        X_umap = reducer.fit_transform(X_tfidf)
        
        # Visualize
        plt.figure(figsize=(12, 8))
        category_colors = {'Technology': 'blue', 'Medical': 'red', 
                          'Finance': 'green', 'Sports': 'orange'}
        
        for category in set(categories):
            mask = np.array(categories) == category
            plt.scatter(X_umap[mask, 0], X_umap[mask, 1], 
                       c=category_colors[category], label=category, 
                       alpha=0.7, s=100)
        
        # Add document numbers as annotations
        for i, doc in enumerate(documents):
            plt.annotate(f'Doc {i+1}', (X_umap[i, 0], X_umap[i, 1]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.title('UMAP Visualization of Document Topics')
        plt.xlabel('UMAP Component 1')
        plt.ylabel('UMAP Component 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        print("Text Analysis Results:")
        print("- UMAP groups documents by semantic similarity")
        print("- Cosine distance works well for TF-IDF features")
        print("- Can reveal topic clusters in document collections")
        print("- Useful for document organization and search")
        
    text_analysis_example()
```

## Advanced Techniques

### Supervised UMAP
```python
if UMAP_AVAILABLE:
    def supervised_umap_example():
        """Demonstrate supervised UMAP for better class separation"""
        
        X, y = digits.data, digits.target
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Compare unsupervised vs supervised UMAP
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Unsupervised UMAP
        reducer_unsup = umap.UMAP(n_components=2, random_state=42)
        X_umap_unsup = reducer_unsup.fit_transform(X_scaled)
        
        scatter1 = ax1.scatter(X_umap_unsup[:, 0], X_umap_unsup[:, 1], 
                             c=y, cmap='tab10', alpha=0.7)
        ax1.set_title('Unsupervised UMAP')
        ax1.set_xlabel('UMAP 1')
        ax1.set_ylabel('UMAP 2')
        
        # Supervised UMAP
        reducer_sup = umap.UMAP(n_components=2, random_state=42)
        X_umap_sup = reducer_sup.fit(X_scaled, y=y).transform(X_scaled)
        
        scatter2 = ax2.scatter(X_umap_sup[:, 0], X_umap_sup[:, 1], 
                             c=y, cmap='tab10', alpha=0.7)
        ax2.set_title('Supervised UMAP')
        ax2.set_xlabel('UMAP 1')
        ax2.set_ylabel('UMAP 2')
        
        plt.tight_layout()
        plt.show()
        
        # Evaluate class separation
        from sklearn.metrics import silhouette_score
        
        sil_unsup = silhouette_score(X_umap_unsup, y)
        sil_sup = silhouette_score(X_umap_sup, y)
        
        print(f"Silhouette Score - Unsupervised: {sil_unsup:.3f}")
        print(f"Silhouette Score - Supervised: {sil_sup:.3f}")
        print(f"Improvement: {(sil_sup - sil_unsup)/sil_unsup*100:.1f}%")
        
        print("\nSupervised UMAP Benefits:")
        print("- Uses label information to improve separation")
        print("- Better class boundaries in embedding space")
        print("- Useful for classification tasks")
        print("- Can reveal discriminative features")
        
    supervised_umap_example()
```

### Inverse Transform and Reconstruction
```python
if UMAP_AVAILABLE:
    def inverse_transform_example():
        """Demonstrate UMAP's ability to transform new points"""
        
        # Split data into train and test
        from sklearn.model_selection import train_test_split
        
        X, y = digits.data, digits.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Fit UMAP on training data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        reducer = umap.UMAP(n_components=2, random_state=42)
        X_train_umap = reducer.fit_transform(X_train_scaled)
        
        # Transform test data
        X_test_umap = reducer.transform(X_test_scaled)
        
        # Visualize
        plt.figure(figsize=(15, 6))
        
        # Training data
        plt.subplot(1, 2, 1)
        scatter1 = plt.scatter(X_train_umap[:, 0], X_train_umap[:, 1], 
                             c=y_train, cmap='tab10', alpha=0.7, label='Train')
        plt.title('Training Data Embedding')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.legend()
        
        # Test data overlaid on training
        plt.subplot(1, 2, 2)
        plt.scatter(X_train_umap[:, 0], X_train_umap[:, 1], 
                   c=y_train, cmap='tab10', alpha=0.3, s=20, label='Train')
        scatter2 = plt.scatter(X_test_umap[:, 0], X_test_umap[:, 1], 
                             c=y_test, cmap='tab10', alpha=0.8, s=50, 
                             marker='s', label='Test')
        plt.title('Test Data Projected onto Training Embedding')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        print("Inverse Transform Capabilities:")
        print("- UMAP can embed new points into existing embedding")
        print("- Useful for real-time analysis of streaming data")
        print("- Enables consistent visualization across datasets")
        print("- Important for production machine learning systems")
        
    inverse_transform_example()
```

## Best Practices and Guidelines

### Data Preprocessing Guidelines
```python
if UMAP_AVAILABLE:
    def preprocessing_best_practices():
        """Guidelines for preprocessing data before UMAP"""
        
        print("UMAP Preprocessing Best Practices:")
        print("=" * 50)
        
        # Different preprocessing approaches
        X, y = digits.data, digits.target
        
        preprocessing_methods = {
            'Raw Data': X,
            'StandardScaler': StandardScaler().fit_transform(X),
            'MinMaxScaler': MinMaxScaler().fit_transform(X),
            'RobustScaler': RobustScaler().fit_transform(X),
            'PCA + Standard': StandardScaler().fit_transform(
                PCA(n_components=50).fit_transform(X)
            )
        }
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for i, (method, X_processed) in enumerate(preprocessing_methods.items()):
            if i < len(axes):
                reducer = umap.UMAP(n_components=2, random_state=42)
                X_umap = reducer.fit_transform(X_processed)
                
                scatter = axes[i].scatter(X_umap[:, 0], X_umap[:, 1], 
                                        c=y, cmap='tab10', alpha=0.7)
                axes[i].set_title(f'{method}')
                axes[i].set_xlabel('UMAP 1')
                axes[i].set_ylabel('UMAP 2')
        
        # Remove extra subplot
        if len(preprocessing_methods) < len(axes):
            axes[-1].remove()
        
        plt.tight_layout()
        plt.show()
        
        print("\nPreprocessing Recommendations:")
        print("1. StandardScaler: Most common choice")
        print("2. RobustScaler: Better for data with outliers")
        print("3. PCA preprocessing: For very high-dimensional data")
        print("4. MinMaxScaler: When preserving zero values is important")
        print("5. Consider domain-specific preprocessing (log transform for counts)")
        
    from sklearn.preprocessing import MinMaxScaler, RobustScaler
    preprocessing_best_practices()
```

### Parameter Selection Strategy
```python
if UMAP_AVAILABLE:
    def parameter_selection_strategy():
        """Systematic approach to parameter selection"""
        
        def evaluate_embedding_quality(X_embedded, y_true):
            """Evaluate embedding quality using multiple metrics"""
            from sklearn.metrics import silhouette_score, adjusted_rand_score
            from sklearn.cluster import KMeans
            
            # Silhouette score
            sil_score = silhouette_score(X_embedded, y_true)
            
            # Clustering performance
            n_clusters = len(np.unique(y_true))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(X_embedded)
            ari_score = adjusted_rand_score(y_true, cluster_labels)
            
            return {
                'silhouette': sil_score,
                'ari': ari_score,
                'combined': (sil_score + ari_score) / 2
            }
        
        # Test different parameter combinations
        X, y = iris.data, iris.target
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        param_grid = {
            'n_neighbors': [5, 10, 15, 30],
            'min_dist': [0.0, 0.1, 0.3, 0.5]
        }
        
        results = []
        
        for n_neighbors in param_grid['n_neighbors']:
            for min_dist in param_grid['min_dist']:
                reducer = umap.UMAP(
                    n_components=2,
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    random_state=42
                )
                
                X_umap = reducer.fit_transform(X_scaled)
                scores = evaluate_embedding_quality(X_umap, y)
                
                results.append({
                    'n_neighbors': n_neighbors,
                    'min_dist': min_dist,
                    **scores
                })
        
        # Convert to DataFrame for analysis
        import pandas as pd
        results_df = pd.DataFrame(results)
        
        # Find best parameters
        best_params = results_df.loc[results_df['combined'].idxmax()]
        
        print("Parameter Selection Results:")
        print("=" * 40)
        print(f"Best parameters:")
        print(f"  n_neighbors: {best_params['n_neighbors']}")
        print(f"  min_dist: {best_params['min_dist']}")
        print(f"  Combined score: {best_params['combined']:.3f}")
        
        # Visualize parameter space
        pivot_sil = results_df.pivot(index='min_dist', columns='n_neighbors', 
                                    values='silhouette')
        pivot_ari = results_df.pivot(index='min_dist', columns='n_neighbors', 
                                    values='ari')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        sns.heatmap(pivot_sil, annot=True, fmt='.3f', cmap='viridis', ax=ax1)
        ax1.set_title('Silhouette Score')
        ax1.set_xlabel('n_neighbors')
        ax1.set_ylabel('min_dist')
        
        sns.heatmap(pivot_ari, annot=True, fmt='.3f', cmap='viridis', ax=ax2)
        ax2.set_title('Adjusted Rand Index')
        ax2.set_xlabel('n_neighbors')
        ax2.set_ylabel('min_dist')
        
        plt.tight_layout()
        plt.show()
        
    parameter_selection_strategy()
```

## Troubleshooting Common Issues

### Issue 1: Poor Separation
```python
if UMAP_AVAILABLE:
    def troubleshoot_poor_separation():
        """Address poor cluster separation in UMAP"""
        
        print("Troubleshooting Poor Cluster Separation:")
        print("=" * 50)
        
        # Create challenging dataset
        from sklearn.datasets import make_blobs
        X_easy, y_easy = make_blobs(n_samples=300, centers=4, cluster_std=0.5, 
                                   random_state=42)
        X_hard, y_hard = make_blobs(n_samples=300, centers=4, cluster_std=2.0, 
                                   random_state=42)
        
        solutions = [
            {'title': 'Default Parameters', 'params': {}},
            {'title': 'Smaller min_dist', 'params': {'min_dist': 0.0}},
            {'title': 'Fewer neighbors', 'params': {'n_neighbors': 5}},
            {'title': 'Combined approach', 'params': {'min_dist': 0.0, 'n_neighbors': 5}}
        ]
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        for i, solution in enumerate(solutions):
            # Easy dataset
            reducer_easy = umap.UMAP(n_components=2, random_state=42, **solution['params'])
            X_umap_easy = reducer_easy.fit_transform(StandardScaler().fit_transform(X_easy))
            
            axes[0, i].scatter(X_umap_easy[:, 0], X_umap_easy[:, 1], 
                             c=y_easy, cmap='viridis', alpha=0.7)
            axes[0, i].set_title(f'Easy Data: {solution["title"]}')
            
            # Hard dataset
            reducer_hard = umap.UMAP(n_components=2, random_state=42, **solution['params'])
            X_umap_hard = reducer_hard.fit_transform(StandardScaler().fit_transform(X_hard))
            
            axes[1, i].scatter(X_umap_hard[:, 0], X_umap_hard[:, 1], 
                             c=y_hard, cmap='viridis', alpha=0.7)
            axes[1, i].set_title(f'Hard Data: {solution["title"]}')
        
        plt.tight_layout()
        plt.show()
        
        print("Solutions for Poor Separation:")
        print("1. Reduce min_dist (0.0 for tightest clusters)")
        print("2. Reduce n_neighbors (focus on local structure)")
        print("3. Try supervised UMAP if labels available")
        print("4. Consider different distance metrics")
        print("5. Preprocess data (PCA, feature selection)")
        
    troubleshoot_poor_separation()
```

### Issue 2: Computational Performance
```python
if UMAP_AVAILABLE:
    def optimize_performance():
        """Strategies for optimizing UMAP performance"""
        
        print("UMAP Performance Optimization:")
        print("=" * 40)
        
        # Simulate large dataset
        np.random.seed(42)
        X_large = np.random.randn(5000, 100)
        y_large = np.random.randint(0, 10, 5000)
        
        optimization_strategies = [
            {
                'name': 'Default',
                'preprocessor': lambda x: StandardScaler().fit_transform(x),
                'params': {}
            },
            {
                'name': 'PCA Preprocessing',
                'preprocessor': lambda x: StandardScaler().fit_transform(
                    PCA(n_components=50).fit_transform(x)
                ),
                'params': {}
            },
            {
                'name': 'Lower precision',
                'preprocessor': lambda x: StandardScaler().fit_transform(x),
                'params': {'low_memory': True}
            },
            {
                'name': 'Fewer neighbors',
                'preprocessor': lambda x: StandardScaler().fit_transform(x),
                'params': {'n_neighbors': 10}
            }
        ]
        
        times = []
        memory_usage = []
        
        for strategy in optimization_strategies:
            import time
            import psutil
            import os
            
            # Measure memory before
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Time the operation
            start_time = time.time()
            
            X_processed = strategy['preprocessor'](X_large[:1000])  # Subset for demo
            reducer = umap.UMAP(n_components=2, random_state=42, **strategy['params'])
            X_umap = reducer.fit_transform(X_processed)
            
            end_time = time.time()
            
            # Measure memory after
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            
            times.append(end_time - start_time)
            memory_usage.append(memory_after - memory_before)
            
            print(f"{strategy['name']}: {end_time - start_time:.2f}s, "
                  f"Memory: +{memory_after - memory_before:.1f}MB")
        
        # Plot performance comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        strategy_names = [s['name'] for s in optimization_strategies]
        
        ax1.bar(strategy_names, times, alpha=0.7)
        ax1.set_title('Execution Time Comparison')
        ax1.set_ylabel('Time (seconds)')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        ax2.bar(strategy_names, memory_usage, alpha=0.7, color='orange')
        ax2.set_title('Memory Usage Comparison')
        ax2.set_ylabel('Memory Increase (MB)')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        print("\nPerformance Optimization Tips:")
        print("1. Use PCA preprocessing for high-dimensional data")
        print("2. Set low_memory=True for memory-constrained environments")
        print("3. Reduce n_neighbors for faster computation")
        print("4. Consider approximate nearest neighbors for very large datasets")
        print("5. Use GPU acceleration if available")
        
    optimize_performance()
```

## Advantages and Limitations

### Comprehensive Analysis
```python
if UMAP_AVAILABLE:
    def umap_advantages_limitations():
        """Comprehensive analysis of UMAP strengths and weaknesses"""
        
        comparison_data = {
            "Advantages": [
                "Preserves both local and global structure",
                "Faster than t-SNE (especially for large datasets)",
                "Can transform new data points (out-of-sample)",
                "Theoretically well-founded (topology + manifold learning)",
                "Scalable to large datasets",
                "Multiple distance metrics supported",
                "Supervised version available",
                "More deterministic than t-SNE",
                "Better preservation of density information",
                "Can work in higher dimensions (not just 2D/3D)"
            ],
            
            "Limitations": [
                "Still relatively new (less established than t-SNE)",
                "Can be sensitive to hyperparameters",
                "May not separate clusters as distinctly as t-SNE", 
                "Theoretical complexity can be intimidating",
                "Less intuitive parameter meanings than t-SNE",
                "Can create artifacts with very sparse data",
                "Memory usage can be high for very large datasets",
                "May struggle with very high-dimensional data without preprocessing"
            ],
            
            "Best Use Cases": [
                "Large dataset visualization (>5000 samples)",
                "When global structure preservation is important",
                "Real-time or production systems (due to transform ability)",
                "Single-cell RNA sequencing analysis",
                "Image analysis and computer vision",
                "When computational efficiency is crucial",
                "Text analysis and NLP tasks",
                "Continuous trajectory analysis",
                "When reproducibility is important"
            ],
            
            "UMAP vs t-SNE": [
                "UMAP: Better global structure, faster, can transform new points",
                "t-SNE: Better local cluster separation, more established",
                "UMAP: More suitable for exploratory analysis",
                "t-SNE: More suitable for final publication figures",
                "UMAP: Better for continuous manifolds",
                "t-SNE: Better for discrete cluster visualization",
                "UMAP: More parameters to tune",
                "t-SNE: Simpler parameter space"
            ]
        }
        
        for category, items in comparison_data.items():
            print(f"\n{category}:")
            print("=" * (len(category) + 1))
            for i, item in enumerate(items, 1):
                print(f"{i}. {item}")
    
    umap_advantages_limitations()
else:
    print("UMAP not available for comprehensive analysis")
```

## Learning Objectives
- [ ] **Understand UMAP theory**: Master manifold learning and topological data analysis concepts
- [ ] **Apply UMAP effectively**: Use appropriate parameters for different data types and sizes
- [ ] **Compare with t-SNE**: Know when to use UMAP vs t-SNE for visualization
- [ ] **Preserve data structure**: Understand how UMAP maintains local and global relationships
- [ ] **Handle large datasets**: Apply UMAP efficiently to big data scenarios
- [ ] **Use supervised UMAP**: Leverage label information for better embeddings
- [ ] **Transform new data**: Apply fitted UMAP models to new data points
- [ ] **Optimize performance**: Use preprocessing and parameter tuning for efficiency
- [ ] **Apply to real problems**: Use UMAP for biology, NLP, computer vision, and other domains
- [ ] **Interpret results correctly**: Understand what UMAP embeddings show and their limitations

## Practice Exercises
1. Compare UMAP and t-SNE on the digits dataset with different parameters
2. Apply UMAP to high-dimensional data with PCA preprocessing
3. Use supervised UMAP to improve class separation
4. Implement a UMAP pipeline for real-time data embedding
5. Apply UMAP to text data using TF-IDF features
6. Create interactive UMAP visualizations with hover information
7. Optimize UMAP parameters using grid search and evaluation metrics
8. Apply UMAP to image data and compare different distance metrics
9. Build a clustering pipeline combining UMAP with K-means
10. Use UMAP for anomaly detection in multivariate time series data