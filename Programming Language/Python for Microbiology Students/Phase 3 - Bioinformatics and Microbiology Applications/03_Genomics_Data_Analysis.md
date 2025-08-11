# Genomics Data Analysis

Genomic data analysis is essential for understanding bacterial genomes, comparative studies, and gene expression patterns in microbiology research.

## Processing Genomic Coordinates

### Understanding Coordinate Systems

```python
import pandas as pd
import numpy as np
from collections import namedtuple

# Define genomic coordinate structure
GenomicRegion = namedtuple('GenomicRegion', ['chromosome', 'start', 'end', 'strand', 'feature_type'])

# Example bacterial genome coordinates
genome_features = [
    GenomicRegion('chromosome', 1000, 2500, '+', 'gene'),
    GenomicRegion('chromosome', 3000, 4200, '-', 'gene'),
    GenomicRegion('chromosome', 5500, 6800, '+', 'gene'),
    GenomicRegion('plasmid_1', 100, 800, '+', 'resistance_gene')
]

def calculate_feature_length(region):
    """Calculate length of genomic feature"""
    return region.end - region.start + 1

def find_overlapping_regions(region1, region2):
    """Check if two genomic regions overlap"""
    if region1.chromosome != region2.chromosome:
        return False
    
    return not (region1.end < region2.start or region2.end < region1.start)

# Example usage
for feature in genome_features:
    length = calculate_feature_length(feature)
    print(f"{feature.feature_type} on {feature.chromosome}: {length} bp")
```

### Coordinate Conversion and Manipulation

```python
def convert_coordinates(position, conversion_type='1_to_0_based'):
    """Convert between 1-based and 0-based coordinate systems"""
    if conversion_type == '1_to_0_based':
        return position - 1
    elif conversion_type == '0_to_1_based':
        return position + 1
    return position

def extract_sequence_region(sequence, start, end, coordinate_system='1_based'):
    """Extract subsequence using genomic coordinates"""
    if coordinate_system == '1_based':
        # Convert to 0-based for Python indexing
        return sequence[start-1:end]
    else:
        return sequence[start:end]

# Example genome sequence
genome_sequence = "ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAGATCGATCGATCG"

# Extract gene region
gene_start, gene_end = 5, 20
gene_sequence = extract_sequence_region(genome_sequence, gene_start, gene_end)
print(f"Gene sequence: {gene_sequence}")
```

## Working with Annotation Files (GFF, GTF)

### Parsing GFF3 Files

```python
def parse_gff3_line(line):
    """Parse a single line from GFF3 file"""
    if line.startswith('#') or not line.strip():
        return None
    
    fields = line.strip().split('\t')
    
    if len(fields) != 9:
        return None
    
    gff_record = {
        'seqid': fields[0],
        'source': fields[1],
        'type': fields[2],
        'start': int(fields[3]),
        'end': int(fields[4]),
        'score': fields[5] if fields[5] != '.' else None,
        'strand': fields[6],
        'phase': fields[7] if fields[7] != '.' else None,
        'attributes': {}
    }
    
    # Parse attributes
    if fields[8]:
        for attr in fields[8].split(';'):
            if '=' in attr:
                key, value = attr.split('=', 1)
                gff_record['attributes'][key] = value
    
    return gff_record

def read_gff3_file(filepath):
    """Read and parse GFF3 file"""
    annotations = []
    
    try:
        with open(filepath, 'r') as file:
            for line in file:
                record = parse_gff3_line(line)
                if record:
                    annotations.append(record)
    except FileNotFoundError:
        print(f"File {filepath} not found")
        return []
    
    return pd.DataFrame(annotations)

# Example usage (uncomment when you have GFF3 files)
# annotations_df = read_gff3_file('bacterial_genome.gff3')
# print(f"Loaded {len(annotations_df)} annotations")
```

### Gene Feature Analysis

```python
def analyze_gene_features(annotations_df):
    """Analyze gene features from annotation data"""
    
    if annotations_df.empty:
        print("No annotation data available")
        return
    
    # Gene statistics
    genes = annotations_df[annotations_df['type'] == 'gene']
    
    gene_lengths = genes['end'] - genes['start'] + 1
    
    stats = {
        'total_genes': len(genes),
        'average_length': gene_lengths.mean(),
        'median_length': gene_lengths.median(),
        'min_length': gene_lengths.min(),
        'max_length': gene_lengths.max()
    }
    
    print("Gene Feature Analysis:")
    for key, value in stats.items():
        if 'length' in key:
            print(f"  {key.replace('_', ' ').title()}: {value:.0f} bp")
        else:
            print(f"  {key.replace('_', ' ').title()}: {value}")
    
    # Feature type distribution
    feature_counts = annotations_df['type'].value_counts()
    print(f"\nFeature type distribution:")
    for feature_type, count in feature_counts.head().items():
        print(f"  {feature_type}: {count}")
    
    return stats

# Create sample annotation data for demonstration
sample_annotations = pd.DataFrame({
    'seqid': ['chromosome'] * 10,
    'type': ['gene', 'CDS', 'gene', 'CDS', 'tRNA', 'gene', 'CDS', 'rRNA', 'gene', 'CDS'],
    'start': [1000, 1050, 2000, 2050, 3000, 4000, 4050, 5000, 6000, 6050],
    'end': [1500, 1450, 2800, 2750, 3080, 4600, 4550, 5150, 6900, 6850],
    'strand': ['+', '+', '-', '-', '+', '+', '+', '+', '-', '-']
})

analyze_gene_features(sample_annotations)
```

## Basic Comparative Genomics

### Genome Comparison Functions

```python
def compare_genome_sizes(genomes_dict):
    """Compare sizes of multiple bacterial genomes"""
    
    comparison_data = []
    
    for genome_name, sequence in genomes_dict.items():
        genome_length = len(sequence)
        gc_content = (sequence.count('G') + sequence.count('C')) / genome_length * 100
        
        comparison_data.append({
            'genome': genome_name,
            'length_bp': genome_length,
            'gc_content': gc_content
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    print("Genome Comparison:")
    print(comparison_df.round(2))
    
    return comparison_df

def find_conserved_regions(seq1, seq2, min_length=20):
    """Find conserved regions between two genomes"""
    
    conserved_regions = []
    
    for i in range(len(seq1) - min_length + 1):
        for j in range(len(seq2) - min_length + 1):
            # Check for exact matches
            if seq1[i:i+min_length] == seq2[j:j+min_length]:
                conserved_regions.append({
                    'seq1_start': i,
                    'seq1_end': i + min_length,
                    'seq2_start': j,
                    'seq2_end': j + min_length,
                    'sequence': seq1[i:i+min_length]
                })
    
    return conserved_regions

# Example comparative analysis
sample_genomes = {
    'E_coli_K12': 'ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG' * 100,
    'E_coli_O157': 'ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG' * 95,
    'S_aureus': 'ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATCG' * 80
}

genome_comparison = compare_genome_sizes(sample_genomes)
```

## Gene Expression Data Handling

### Processing Expression Matrices

```python
def process_expression_data(expression_file='gene_expression.csv'):
    """Process gene expression data matrix"""
    
    # Create sample expression data
    genes = [f'gene_{i:03d}' for i in range(1, 101)]
    conditions = ['control_1', 'control_2', 'treatment_1', 'treatment_2']
    
    # Generate random expression values (log2 scale)
    np.random.seed(42)
    expression_data = np.random.normal(5, 2, (len(genes), len(conditions)))
    
    # Create DataFrame
    expression_df = pd.DataFrame(expression_data, 
                                index=genes, 
                                columns=conditions)
    
    print("Gene Expression Data Summary:")
    print(f"Genes: {len(genes)}")
    print(f"Conditions: {len(conditions)}")
    print(f"Expression range: {expression_df.min().min():.2f} to {expression_df.max().max():.2f}")
    
    return expression_df

def calculate_fold_changes(expression_df, control_cols, treatment_cols):
    """Calculate fold changes between conditions"""
    
    control_mean = expression_df[control_cols].mean(axis=1)
    treatment_mean = expression_df[treatment_cols].mean(axis=1)
    
    fold_change = treatment_mean - control_mean  # Log2 fold change
    
    results_df = pd.DataFrame({
        'control_mean': control_mean,
        'treatment_mean': treatment_mean,
        'log2_fold_change': fold_change,
        'fold_change': 2 ** fold_change
    })
    
    # Identify significantly changed genes (arbitrary threshold)
    significantly_changed = abs(results_df['log2_fold_change']) > 1
    
    print(f"Significantly changed genes: {significantly_changed.sum()}")
    print("Top upregulated genes:")
    print(results_df.nlargest(5, 'log2_fold_change')[['log2_fold_change', 'fold_change']])
    
    return results_df

# Example expression analysis
expression_data = process_expression_data()
fold_changes = calculate_fold_changes(
    expression_data, 
    ['control_1', 'control_2'], 
    ['treatment_1', 'treatment_2']
)
```

## Best Practices for Genomics Analysis

1. **Validate coordinate systems** before analysis
2. **Handle large files efficiently** using chunked reading
3. **Document genome versions** and annotation sources
4. **Use appropriate statistical methods** for expression analysis
5. **Consider multiple testing corrections** for differential expression
6. **Visualize results** with appropriate plots
7. **Backup and version control** genomic datasets
