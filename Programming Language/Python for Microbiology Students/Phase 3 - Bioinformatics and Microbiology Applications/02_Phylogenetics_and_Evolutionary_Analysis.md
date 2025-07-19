# Phylogenetics and Evolutionary Analysis

Phylogenetic analysis helps understand evolutionary relationships between microorganisms, track disease outbreaks, and classify new bacterial isolates.

## Introduction to Phylogenetics

### Basic Concepts

```python
import numpy as np
import pandas as pd
from Bio import Phylo, SeqIO, AlignIO
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
from Bio.Phylo.TreeConstruction import ParsimonyScorer, ParsimonySearcher
import matplotlib.pyplot as plt

# Phylogenetic analysis workflow:
# 1. Sequence collection and alignment
# 2. Distance matrix calculation
# 3. Tree construction
# 4. Tree visualization and interpretation
```

### Sample Sequence Data

```python
# Create sample bacterial 16S rRNA sequences for demonstration
sample_sequences = {
    'E_coli': 'AGAGTTTGATCCTGGCTCAGATTGAACGCTGGCGGCAGGCCTAACACATGCAAGTCGAACG',
    'S_aureus': 'AGAGTTTGATCCTGGCTCAGGATGAACGCTGGCGGCGTGCCTAATACATGCAAGTCGAGCG',
    'B_subtilis': 'AGAGTTTGATCCTGGCTCAGATTGAACGCTGGCGGCAGGCCTAACACATGCAAGTCGAACG',
    'P_aeruginosa': 'AGAGTTTGATCCTGGCTCAGATTGAACGCTGGCGGCAGGCCTAACACATGCAAGTCGAGCG',
    'M_tuberculosis': 'AGAGTTTGATCCTGGCTCAGGATGAACGCTGGCGGCGTGCTTAACACATGCAAGTCGAACG'
}

print("Sample bacterial sequences loaded:")
for organism, seq in sample_sequences.items():
    print(f"{organism}: {seq[:30]}... ({len(seq)} bp)")
```

## Distance Matrices and Clustering

### Calculating Genetic Distances

```python
def calculate_hamming_distance(seq1, seq2):
    """Calculate Hamming distance between two sequences"""
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be of equal length")
    
    differences = sum(c1 != c2 for c1, c2 in zip(seq1, seq2))
    return differences / len(seq1)

def create_distance_matrix(sequences):
    """Create pairwise distance matrix from sequences"""
    organisms = list(sequences.keys())
    n = len(organisms)
    
    # Initialize distance matrix
    distance_matrix = np.zeros((n, n))
    
    for i, org1 in enumerate(organisms):
        for j, org2 in enumerate(organisms):
            if i != j:
                distance = calculate_hamming_distance(sequences[org1], sequences[org2])
                distance_matrix[i][j] = distance
    
    # Create DataFrame for better visualization
    dist_df = pd.DataFrame(distance_matrix, 
                          index=organisms, 
                          columns=organisms)
    
    return dist_df

# Calculate distance matrix
distance_df = create_distance_matrix(sample_sequences)
print("Pairwise distance matrix:")
print(distance_df.round(3))
```

### Hierarchical Clustering

```python
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

def perform_hierarchical_clustering(distance_matrix):
    """Perform hierarchical clustering on distance matrix"""
    
    # Convert to condensed distance matrix for scipy
    condensed_distances = squareform(distance_matrix.values)
    
    # Perform linkage (clustering)
    linkage_matrix = linkage(condensed_distances, method='average')
    
    return linkage_matrix

def plot_dendrogram(linkage_matrix, labels):
    """Plot dendrogram from linkage matrix"""
    plt.figure(figsize=(12, 6))
    
    dendrogram_plot = dendrogram(
        linkage_matrix,
        labels=labels,
        leaf_rotation=45,
        leaf_font_size=10
    )
    
    plt.title('Bacterial Phylogenetic Dendrogram')
    plt.xlabel('Organisms')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.show()
    
    return dendrogram_plot

# Perform clustering and visualization
linkage_result = perform_hierarchical_clustering(distance_df)
organisms = list(sample_sequences.keys())
dendrogram_result = plot_dendrogram(linkage_result, organisms)
```

## Tree Construction Methods

### Neighbor-Joining Algorithm

```python
def neighbor_joining(distance_matrix):
    """Simplified neighbor-joining algorithm implementation"""
    
    # Create a copy of the distance matrix
    distances = distance_matrix.copy()
    taxa = list(distances.index)
    n = len(taxa)
    
    # Tree structure storage
    tree_structure = []
    
    while n > 2:
        # Calculate Q matrix
        Q = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    Q[i][j] = (n - 2) * distances.iloc[i, j] - \
                             distances.iloc[i, :].sum() - distances.iloc[j, :].sum()
        
        # Find minimum Q value
        min_pos = np.unravel_index(np.argmin(Q + np.eye(n) * 1000), Q.shape)
        i, j = min_pos
        
        # Join taxa i and j
        taxon_i = taxa[i]
        taxon_j = taxa[j]
        new_taxon = f"({taxon_i},{taxon_j})"
        
        # Calculate branch lengths
        branch_i = (distances.iloc[i, j] + 
                   (distances.iloc[i, :].sum() - distances.iloc[j, :].sum()) / (n - 2)) / 2
        branch_j = distances.iloc[i, j] - branch_i
        
        tree_structure.append((taxon_i, taxon_j, branch_i, branch_j, distances.iloc[i, j]))
        
        # Create new distance matrix
        new_distances = []
        new_taxa = []
        
        for k in range(n):
            if k != i and k != j:
                new_dist = (distances.iloc[i, k] + distances.iloc[j, k] - distances.iloc[i, j]) / 2
                new_distances.append(new_dist)
                new_taxa.append(taxa[k])
        
        # Add new internal node
        new_taxa.append(new_taxon)
        
        # Rebuild distance matrix
        size = len(new_taxa)
        new_dist_matrix = np.zeros((size, size))
        
        for x in range(size - 1):
            for y in range(x + 1, size - 1):
                orig_x = taxa.index(new_taxa[x]) if new_taxa[x] in taxa else None
                orig_y = taxa.index(new_taxa[y]) if new_taxa[y] in taxa else None
                
                if orig_x is not None and orig_y is not None:
                    new_dist_matrix[x][y] = new_dist_matrix[y][x] = distances.iloc[orig_x, orig_y]
        
        # Add distances to new internal node
        for x in range(size - 1):
            new_dist_matrix[x][size-1] = new_dist_matrix[size-1][x] = new_distances[x]
        
        distances = pd.DataFrame(new_dist_matrix, index=new_taxa, columns=new_taxa)
        taxa = new_taxa
        n = len(taxa)
    
    return tree_structure

# Apply neighbor-joining
nj_tree = neighbor_joining(distance_df)
print("Neighbor-joining tree construction:")
for i, (taxon1, taxon2, branch1, branch2, distance) in enumerate(nj_tree):
    print(f"Step {i+1}: Join {taxon1} and {taxon2}")
    print(f"  Branch lengths: {branch1:.4f}, {branch2:.4f}")
    print(f"  Distance: {distance:.4f}\n")
```

### Maximum Parsimony

```python
def calculate_parsimony_score(sequences, tree_topology):
    """Calculate parsimony score for a given tree topology"""
    
    # Simple parsimony scoring based on character changes
    total_score = 0
    seq_length = len(list(sequences.values())[0])
    
    for position in range(seq_length):
        # Get characters at this position for all sequences
        chars = [sequences[organism][position] for organism in tree_topology]
        
        # Count unique characters (minimum changes = unique_chars - 1)
        unique_chars = len(set(chars))
        position_score = max(0, unique_chars - 1)
        total_score += position_score
    
    return total_score

# Calculate parsimony scores for different topologies
topologies = [
    ['E_coli', 'S_aureus', 'B_subtilis', 'P_aeruginosa', 'M_tuberculosis'],
    ['S_aureus', 'B_subtilis', 'E_coli', 'P_aeruginosa', 'M_tuberculosis'],
    ['P_aeruginosa', 'E_coli', 'S_aureus', 'B_subtilis', 'M_tuberculosis']
]

print("Parsimony analysis:")
for i, topology in enumerate(topologies, 1):
    score = calculate_parsimony_score(sample_sequences, topology)
    print(f"Topology {i}: {' - '.join(topology)}")
    print(f"  Parsimony score: {score}\n")
```

## Tree Visualization and Interpretation

### Advanced Tree Plotting

```python
def create_phylogenetic_tree_plot(distance_matrix, method='neighbor_joining'):
    """Create a professional phylogenetic tree plot"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Distance matrix heatmap
    import seaborn as sns
    sns.heatmap(distance_matrix, annot=True, fmt='.3f', cmap='viridis', ax=ax1)
    ax1.set_title('Pairwise Distance Matrix')
    
    # Plot 2: Dendrogram
    linkage_matrix = perform_hierarchical_clustering(distance_matrix)
    
    from scipy.cluster.hierarchy import dendrogram
    dendrogram(linkage_matrix, 
               labels=list(distance_matrix.index),
               leaf_rotation=45,
               ax=ax2)
    ax2.set_title(f'Phylogenetic Tree ({method})')
    ax2.set_xlabel('Organisms')
    ax2.set_ylabel('Genetic Distance')
    
    plt.tight_layout()
    plt.show()

# Create comprehensive visualization
create_phylogenetic_tree_plot(distance_df)
```

### Bootstrap Analysis

```python
def bootstrap_analysis(sequences, n_bootstrap=100):
    """Perform bootstrap analysis to assess tree reliability"""
    
    seq_length = len(list(sequences.values())[0])
    organisms = list(sequences.keys())
    bootstrap_trees = []
    
    for bootstrap in range(n_bootstrap):
        # Create bootstrap sample (sample with replacement)
        bootstrap_positions = np.random.choice(seq_length, seq_length, replace=True)
        
        # Create bootstrap sequences
        bootstrap_seqs = {}
        for org in organisms:
            bootstrap_seq = ''.join(sequences[org][pos] for pos in bootstrap_positions)
            bootstrap_seqs[org] = bootstrap_seq
        
        # Calculate distance matrix for bootstrap sample
        bootstrap_dist = create_distance_matrix(bootstrap_seqs)
        
        # Perform clustering
        linkage_matrix = perform_hierarchical_clustering(bootstrap_dist)
        bootstrap_trees.append(linkage_matrix)
    
    return bootstrap_trees

def analyze_bootstrap_support(bootstrap_trees, original_tree):
    """Analyze bootstrap support values for tree branches"""
    
    # Simplified bootstrap support calculation
    # In practice, this would involve comparing tree topologies
    
    n_bootstrap = len(bootstrap_trees)
    
    print(f"Bootstrap analysis completed with {n_bootstrap} replicates")
    print("Bootstrap support interpretation:")
    print("  > 95%: Very strong support")
    print("  85-95%: Strong support") 
    print("  70-85%: Moderate support")
    print("  < 70%: Weak support")
    
    # Simulate bootstrap values for demonstration
    support_values = np.random.uniform(60, 98, 3)  # 3 internal nodes
    
    for i, support in enumerate(support_values, 1):
        if support > 95:
            interpretation = "Very strong"
        elif support > 85:
            interpretation = "Strong"
        elif support > 70:
            interpretation = "Moderate"
        else:
            interpretation = "Weak"
        
        print(f"Node {i}: {support:.1f}% ({interpretation})")

# Perform bootstrap analysis
bootstrap_trees = bootstrap_analysis(sample_sequences, n_bootstrap=50)
analyze_bootstrap_support(bootstrap_trees, linkage_result)
```

## Molecular Clock Analysis

```python
def molecular_clock_analysis(distance_matrix, divergence_times=None):
    """Analyze molecular clock and estimate divergence times"""
    
    if divergence_times is None:
        # Default divergence times (millions of years ago)
        divergence_times = {
            ('E_coli', 'S_aureus'): 3000,
            ('B_subtilis', 'E_coli'): 2800,
            ('P_aeruginosa', 'E_coli'): 2500,
            ('M_tuberculosis', 'E_coli'): 3500
        }
    
    print("Molecular Clock Analysis:")
    print("Assuming constant rate of evolution")
    
    # Calculate substitution rates
    for pair, time_mya in divergence_times.items():
        org1, org2 = pair
        if org1 in distance_matrix.index and org2 in distance_matrix.columns:
            genetic_distance = distance_matrix.loc[org1, org2]
            
            # Calculate substitution rate (substitutions per site per million years)
            rate = genetic_distance / (2 * time_mya)  # Divide by 2 for divergence time
            
            print(f"{org1} vs {org2}:")
            print(f"  Genetic distance: {genetic_distance:.4f}")
            print(f"  Divergence time: {time_mya} MYA")
            print(f"  Substitution rate: {rate:.2e} subs/site/MY\n")

# Perform molecular clock analysis
molecular_clock_analysis(distance_df)
```

## Best Practices for Phylogenetic Analysis

1. **Use appropriate molecular markers** (16S rRNA for bacteria)
2. **Include adequate outgroups** for rooting trees
3. **Perform multiple tree construction methods** for comparison
4. **Calculate bootstrap support values** to assess reliability
5. **Consider horizontal gene transfer** in bacterial phylogenies
6. **Use proper sequence alignment** before tree construction
7. **Validate results** with additional molecular markers
8. **Document methodology** and parameters used
