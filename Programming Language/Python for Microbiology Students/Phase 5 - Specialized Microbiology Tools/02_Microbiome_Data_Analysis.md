# Microbiome Data Analysis

## Working with QIIME2 Output
Process and analyze QIIME2 output files for microbiome studies.

```python
import pandas as pd
import numpy as np
from skbio import DistanceMatrix
from skbio.diversity import alpha, beta

def load_qiime2_data(feature_table_path, taxonomy_path, metadata_path):
    """Load QIIME2 output files"""
    
    # Load feature table (OTU/ASV table)
    feature_table = pd.read_csv(feature_table_path, sep='\t', index_col=0, skiprows=1)
    
    # Load taxonomy
    taxonomy = pd.read_csv(taxonomy_path, sep='\t', index_col=0)
    taxonomy['Taxon'] = taxonomy['Taxon'].str.replace('D_[0-9]__', '', regex=True)
    
    # Load metadata
    metadata = pd.read_csv(metadata_path, sep='\t', index_col=0)
    
    return feature_table, taxonomy, metadata

# Example usage
feature_table, taxonomy, metadata = load_qiime2_data(
    'feature-table.tsv', 'taxonomy.tsv', 'metadata.tsv'
)
```

## Diversity Analysis and Rarefaction
Calculate alpha and beta diversity metrics with rarefaction.

```python
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

def calculate_alpha_diversity(feature_table, metrics=['shannon', 'simpson']):
    """Calculate alpha diversity metrics"""
    
    diversity_results = pd.DataFrame(index=feature_table.columns)
    
    for metric in metrics:
        if metric == 'shannon':
            diversity_results[metric] = feature_table.apply(
                lambda x: alpha.shannon(x.values), axis=0
            )
        elif metric == 'simpson':
            diversity_results[metric] = feature_table.apply(
                lambda x: alpha.simpson(x.values), axis=0
            )
        elif metric == 'observed_otus':
            diversity_results[metric] = feature_table.apply(
                lambda x: alpha.observed_otus(x.values), axis=0
            )
    
    return diversity_results

def rarefy_samples(feature_table, depth=1000):
    """Rarefy samples to even depth"""
    
    rarefied_table = feature_table.copy()
    
    for sample in feature_table.columns:
        sample_counts = feature_table[sample].values
        if sample_counts.sum() >= depth:
            # Randomly subsample
            rarefied = np.random.multinomial(depth, sample_counts / sample_counts.sum())
            rarefied_table[sample] = rarefied
        else:
            rarefied_table[sample] = 0  # Remove samples with insufficient depth
    
    return rarefied_table

# Calculate diversity
alpha_div = calculate_alpha_diversity(feature_table)
rarefied_table = rarefy_samples(feature_table, depth=1000)
```

## Taxonomic Classification Processing
Process and visualize taxonomic classifications.

```python
def process_taxonomy(feature_table, taxonomy):
    """Process taxonomy at different levels"""
    
    # Split taxonomy into levels
    tax_levels = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    
    # Parse taxonomy string
    taxonomy_split = taxonomy['Taxon'].str.split(';', expand=True)
    taxonomy_split.columns = tax_levels[:taxonomy_split.shape[1]]
    
    # Collapse feature table at genus level
    genus_table = feature_table.copy()
    genus_table['Genus'] = taxonomy_split['Genus'].fillna('Unassigned')
    
    # Group by genus and sum counts
    genus_collapsed = genus_table.groupby('Genus').sum()
    
    return genus_collapsed, taxonomy_split

def plot_taxonomic_composition(genus_table, top_n=10):
    """Plot taxonomic composition"""
    
    # Calculate relative abundance
    rel_abundance = genus_table.div(genus_table.sum(axis=0), axis=1)
    
    # Get top genera
    mean_abundance = rel_abundance.mean(axis=1).sort_values(ascending=False)
    top_genera = mean_abundance.head(top_n).index
    
    # Plot stacked bar chart
    rel_abundance.loc[top_genera].T.plot(kind='bar', stacked=True, figsize=(12, 6))
    plt.title('Taxonomic Composition (Top 10 Genera)')
    plt.xlabel('Samples')
    plt.ylabel('Relative Abundance')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    return rel_abundance

genus_table, taxonomy_df = process_taxonomy(feature_table, taxonomy)
rel_abundance = plot_taxonomic_composition(genus_table)
```

## Visualization of Microbiome Data
Create comprehensive visualizations for microbiome analysis.

```python
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def plot_beta_diversity_pca(feature_table, metadata, color_by='treatment'):
    """PCA plot for beta diversity"""
    
    # Calculate relative abundance
    rel_abundance = feature_table.div(feature_table.sum(axis=0), axis=1).T
    
    # Perform PCA
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(rel_abundance.fillna(0))
    
    # Create plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(pca_coords[:, 0], pca_coords[:, 1], 
                         c=metadata[color_by].astype('category').cat.codes, 
                         cmap='viridis', s=60)
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title('PCA of Microbiome Composition')
    
    # Add legend
    handles, labels = scatter.legend_elements()
    legend_labels = metadata[color_by].unique()
    plt.legend(handles, legend_labels, title=color_by)
    
    return pca_coords

# Generate PCA plot
pca_coords = plot_beta_diversity_pca(feature_table, metadata, color_by='treatment')
```

These tools enable comprehensive microbiome analysis from QIIME2 outputs to publication-ready visualizations.
