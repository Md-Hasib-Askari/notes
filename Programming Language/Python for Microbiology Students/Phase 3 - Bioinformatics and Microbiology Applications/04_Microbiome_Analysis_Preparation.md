# Microbiome Analysis Preparation

Microbiome analysis involves studying microbial communities through diversity metrics, taxonomic classification, and statistical comparisons between samples.

## Understanding OTU Tables and Taxonomy

### OTU Table Structure

```python
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns

# Create sample OTU table (Operational Taxonomic Units)
def create_sample_otu_table():
    """Generate sample OTU table for microbiome analysis"""
    
    np.random.seed(42)
    
    # Sample names
    samples = [f'Sample_{i:02d}' for i in range(1, 21)]
    
    # OTU names (representative sequences)
    otus = [f'OTU_{i:03d}' for i in range(1, 51)]
    
    # Generate count data (with some samples having higher diversity)
    otu_counts = np.random.negative_binomial(n=5, p=0.3, size=(len(otus), len(samples)))
    
    # Create DataFrame
    otu_table = pd.DataFrame(otu_counts, index=otus, columns=samples)
    
    print("OTU Table Summary:")
    print(f"OTUs: {len(otus)}")
    print(f"Samples: {len(samples)}")
    print(f"Total reads: {otu_table.sum().sum():,}")
    print(f"Average reads per sample: {otu_table.sum().mean():.0f}")
    
    return otu_table

def create_taxonomy_table():
    """Create corresponding taxonomy table"""
    
    # Common bacterial phyla in microbiomes
    phyla = ['Firmicutes', 'Bacteroidetes', 'Proteobacteria', 'Actinobacteria', 'Verrucomicrobia']
    genera = {
        'Firmicutes': ['Lactobacillus', 'Clostridium', 'Enterococcus', 'Streptococcus'],
        'Bacteroidetes': ['Bacteroides', 'Prevotella', 'Parabacteroides'],
        'Proteobacteria': ['Escherichia', 'Klebsiella', 'Pseudomonas'],
        'Actinobacteria': ['Bifidobacterium', 'Propionibacterium'],
        'Verrucomicrobia': ['Akkermansia']
    }
    
    taxonomy_data = []
    
    for i in range(50):  # 50 OTUs
        phylum = np.random.choice(phyla, p=[0.4, 0.3, 0.15, 0.1, 0.05])
        genus = np.random.choice(genera[phylum])
        species = f'{genus.lower()}_{i%5 + 1}'
        
        taxonomy_data.append({
            'OTU_ID': f'OTU_{i+1:03d}',
            'Kingdom': 'Bacteria',
            'Phylum': phylum,
            'Class': f'{phylum}_class',
            'Order': f'{phylum}_order',
            'Family': f'{genus}_family',
            'Genus': genus,
            'Species': species
        })
    
    taxonomy_df = pd.DataFrame(taxonomy_data)
    return taxonomy_df

# Generate sample data
otu_table = create_sample_otu_table()
taxonomy_table = create_taxonomy_table()

print("\nTaxonomy Distribution:")
print(taxonomy_table['Phylum'].value_counts())
```

### Data Preprocessing

```python
def filter_low_abundance_otus(otu_table, min_count=10, min_samples=2):
    """Filter out low-abundance OTUs"""
    
    # Remove OTUs with low total counts
    total_counts = otu_table.sum(axis=1)
    abundant_otus = total_counts >= min_count
    
    # Remove OTUs present in few samples
    presence_counts = (otu_table > 0).sum(axis=1)
    prevalent_otus = presence_counts >= min_samples
    
    # Combine filters
    keep_otus = abundant_otus & prevalent_otus
    
    filtered_table = otu_table[keep_otus]
    
    print("OTU Filtering Results:")
    print(f"Original OTUs: {len(otu_table)}")
    print(f"Filtered OTUs: {len(filtered_table)}")
    print(f"Removed: {len(otu_table) - len(filtered_table)}")
    
    return filtered_table

def remove_low_coverage_samples(otu_table, min_reads=1000):
    """Remove samples with insufficient sequencing depth"""
    
    sample_depths = otu_table.sum(axis=0)
    adequate_samples = sample_depths >= min_reads
    
    filtered_table = otu_table.loc[:, adequate_samples]
    
    print(f"Sample Filtering Results:")
    print(f"Original samples: {len(otu_table.columns)}")
    print(f"Filtered samples: {len(filtered_table.columns)}")
    print(f"Removed: {len(otu_table.columns) - len(filtered_table.columns)}")
    
    return filtered_table

# Apply filters
filtered_otu_table = filter_low_abundance_otus(otu_table)
filtered_otu_table = remove_low_coverage_samples(filtered_otu_table)
```

## Alpha and Beta Diversity Concepts

### Alpha Diversity Calculations

```python
def calculate_alpha_diversity(otu_table):
    """Calculate various alpha diversity metrics"""
    
    diversity_metrics = {}
    
    for sample in otu_table.columns:
        counts = otu_table[sample]
        non_zero_counts = counts[counts > 0]
        
        # Species richness (number of observed OTUs)
        richness = len(non_zero_counts)
        
        # Shannon diversity
        proportions = non_zero_counts / non_zero_counts.sum()
        shannon = -np.sum(proportions * np.log(proportions))
        
        # Simpson diversity
        simpson = 1 - np.sum(proportions ** 2)
        
        # Evenness (Pielou's evenness)
        evenness = shannon / np.log(richness) if richness > 1 else 0
        
        diversity_metrics[sample] = {
            'richness': richness,
            'shannon': shannon,
            'simpson': simpson,
            'evenness': evenness
        }
    
    diversity_df = pd.DataFrame(diversity_metrics).T
    
    print("Alpha Diversity Summary:")
    print(diversity_df.describe().round(3))
    
    return diversity_df

# Calculate alpha diversity
alpha_diversity = calculate_alpha_diversity(filtered_otu_table)

# Visualize alpha diversity
def plot_alpha_diversity(diversity_df):
    """Plot alpha diversity metrics"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    metrics = ['richness', 'shannon', 'simpson', 'evenness']
    
    for i, metric in enumerate(metrics):
        axes[i].hist(diversity_df[metric], bins=8, alpha=0.7, edgecolor='black')
        axes[i].set_xlabel(metric.title())
        axes[i].set_ylabel('Frequency')
        axes[i].set_title(f'{metric.title()} Distribution')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

plot_alpha_diversity(alpha_diversity)
```

### Beta Diversity Analysis

```python
def calculate_beta_diversity(otu_table, metric='bray_curtis'):
    """Calculate beta diversity distance matrix"""
    
    # Transpose for sample-wise comparison
    sample_data = otu_table.T
    
    if metric == 'bray_curtis':
        # Bray-Curtis dissimilarity
        distances = pdist(sample_data, metric='braycurtis')
    elif metric == 'jaccard':
        # Jaccard distance (presence/absence)
        binary_data = (sample_data > 0).astype(int)
        distances = pdist(binary_data, metric='jaccard')
    elif metric == 'euclidean':
        # Euclidean distance
        distances = pdist(sample_data, metric='euclidean')
    else:
        raise ValueError("Metric must be 'bray_curtis', 'jaccard', or 'euclidean'")
    
    # Convert to square matrix
    distance_matrix = squareform(distances)
    distance_df = pd.DataFrame(distance_matrix, 
                              index=sample_data.index, 
                              columns=sample_data.index)
    
    print(f"Beta Diversity ({metric}):")
    print(f"Average distance: {distances.mean():.3f}")
    print(f"Distance range: {distances.min():.3f} - {distances.max():.3f}")
    
    return distance_df

def plot_beta_diversity_heatmap(distance_df):
    """Plot beta diversity as heatmap"""
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(distance_df, annot=False, cmap='viridis', square=True)
    plt.title('Beta Diversity Heatmap (Bray-Curtis)')
    plt.tight_layout()
    plt.show()

# Calculate and visualize beta diversity
beta_diversity = calculate_beta_diversity(filtered_otu_table, 'bray_curtis')
plot_beta_diversity_heatmap(beta_diversity)
```

## Data Normalization Techniques

### Relative Abundance Normalization

```python
def normalize_to_relative_abundance(otu_table):
    """Convert counts to relative abundances"""
    
    # Calculate total reads per sample
    sample_totals = otu_table.sum(axis=0)
    
    # Convert to relative abundance (proportions)
    relative_abundance = otu_table.div(sample_totals, axis=1)
    
    print("Relative Abundance Normalization:")
    print(f"Sum check (should be 1.0): {relative_abundance.sum(axis=0).mean():.6f}")
    
    return relative_abundance

def rarefy_to_min_depth(otu_table, seed=42):
    """Rarefy samples to minimum sequencing depth"""
    
    np.random.seed(seed)
    
    # Find minimum sequencing depth
    min_depth = otu_table.sum(axis=0).min()
    
    print(f"Rarefying to {min_depth} reads per sample")
    
    rarefied_table = pd.DataFrame(index=otu_table.index, columns=otu_table.columns)
    
    for sample in otu_table.columns:
        sample_counts = otu_table[sample]
        
        # Create pool of reads
        read_pool = []
        for otu, count in sample_counts.items():
            read_pool.extend([otu] * int(count))
        
        # Randomly sample without replacement
        if len(read_pool) >= min_depth:
            rarefied_reads = np.random.choice(read_pool, min_depth, replace=False)
            
            # Count rarefied reads
            unique, counts = np.unique(rarefied_reads, return_counts=True)
            rarefied_counts = pd.Series(counts, index=unique)
            
            # Fill in zeros for missing OTUs
            for otu in otu_table.index:
                rarefied_table.loc[otu, sample] = rarefied_counts.get(otu, 0)
        else:
            # Exclude samples with insufficient depth
            rarefied_table.drop(columns=[sample], inplace=True)
    
    return rarefied_table.astype(int)

# Apply normalization methods
relative_abundance_table = normalize_to_relative_abundance(filtered_otu_table)
rarefied_table = rarefy_to_min_depth(filtered_otu_table)
```

## Sample Metadata Management

### Creating and Managing Metadata

```python
def create_sample_metadata():
    """Create sample metadata for microbiome analysis"""
    
    sample_ids = filtered_otu_table.columns
    
    # Generate random metadata
    np.random.seed(42)
    
    metadata = []
    for sample_id in sample_ids:
        treatment = np.random.choice(['Control', 'Treatment_A', 'Treatment_B'])
        timepoint = np.random.choice(['Day_0', 'Day_7', 'Day_14'])
        subject_id = f'Subject_{np.random.randint(1, 11):02d}'
        age = np.random.randint(20, 65)
        gender = np.random.choice(['Male', 'Female'])
        
        metadata.append({
            'Sample_ID': sample_id,
            'Treatment': treatment,
            'Timepoint': timepoint,
            'Subject_ID': subject_id,
            'Age': age,
            'Gender': gender
        })
    
    metadata_df = pd.DataFrame(metadata)
    
    print("Sample Metadata Summary:")
    print(f"Samples: {len(metadata_df)}")
    print(f"Treatment groups: {metadata_df['Treatment'].value_counts().to_dict()}")
    print(f"Timepoints: {metadata_df['Timepoint'].value_counts().to_dict()}")
    
    return metadata_df

def compare_groups_by_metadata(diversity_df, metadata_df, group_column='Treatment'):
    """Compare diversity between metadata groups"""
    
    # Merge diversity with metadata
    combined_df = diversity_df.merge(metadata_df.set_index('Sample_ID'), 
                                    left_index=True, right_index=True)
    
    print(f"Comparing diversity by {group_column}:")
    
    # Statistical comparison
    groups = combined_df[group_column].unique()
    
    for metric in ['richness', 'shannon', 'simpson']:
        print(f"\n{metric.title()} comparison:")
        
        group_data = []
        for group in groups:
            group_values = combined_df[combined_df[group_column] == group][metric]
            group_data.append(group_values)
            print(f"  {group}: {group_values.mean():.3f} ± {group_values.std():.3f}")
        
        # Mann-Whitney U test for two groups
        if len(groups) == 2:
            stat, p_value = mannwhitneyu(group_data[0], group_data[1])
            print(f"  Mann-Whitney U test p-value: {p_value:.4f}")

# Create and analyze metadata
sample_metadata = create_sample_metadata()
compare_groups_by_metadata(alpha_diversity, sample_metadata, 'Treatment')
```

### Taxonomic Composition Analysis

```python
def analyze_taxonomic_composition(otu_table, taxonomy_table, level='Phylum'):
    """Analyze taxonomic composition at specified level"""
    
    # Merge OTU table with taxonomy
    merged_data = otu_table.T.merge(taxonomy_table.set_index('OTU_ID'), 
                                   left_on=otu_table.index, right_index=True)
    
    # Group by taxonomic level and sum counts
    taxonomic_composition = merged_data.groupby(level)[otu_table.columns].sum()
    
    # Convert to relative abundance
    relative_composition = taxonomic_composition.div(taxonomic_composition.sum(), axis=1)
    
    print(f"Taxonomic Composition at {level} level:")
    print(f"Number of {level}s: {len(relative_composition)}")
    
    # Average composition across samples
    mean_composition = relative_composition.mean(axis=1).sort_values(ascending=False)
    print(f"\nMost abundant {level}s:")
    for taxon, abundance in mean_composition.head().items():
        print(f"  {taxon}: {abundance:.3f} ({abundance*100:.1f}%)")
    
    return relative_composition

# Analyze taxonomic composition
phylum_composition = analyze_taxonomic_composition(filtered_otu_table, taxonomy_table, 'Phylum')
```

## Best Practices for Microbiome Analysis

1. **Quality control** raw sequencing data before analysis
2. **Document normalization methods** used for comparisons
3. **Use appropriate statistical tests** for compositional data
4. **Consider batch effects** in experimental design
5. **Validate findings** with independent datasets
6. **Report sequencing depths** and filtering criteria
7. **Include negative controls** in analysis pipeline
