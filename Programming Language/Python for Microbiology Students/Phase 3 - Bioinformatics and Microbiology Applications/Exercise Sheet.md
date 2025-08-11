# Phase 3 Exercises: Bioinformatics and Microbiology Applications

## Overview
These exercises focus on applying Python to real bioinformatics and microbiology workflows using BioPython, phylogenetic analysis, genomics data processing, and microbiome analysis techniques.

---

## Section 1: BioPython Basics

### Exercise 1.1: Sequence Objects and File Formats
```python
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# Create sample sequences
dna_seq = Seq("ATGGCATCGATCGATCGATCGATAA")
protein_seq = Seq("MKRSGWIVLGLVIAMLAGGFSSQAAA")

# Sample FASTA content
fasta_content = """>gi|123456|ref|NM_000001.1| E.coli 16S rRNA gene
ACGAGTGCGAACGCTGGCGGCAGGCCTAACACATGCAAGTCGAACGGCAGCACGGGGA
AGTAGCTTGCTACTTTGCCGGCGAGCGGCGGACGGGTGAGTAATGTCTGGGAAACTGC
CTGATGGAGGGGGATAACTACTGGAAACGGTAGCTAATACCGCATAACGTCGCAAGAC
>gi|789012|ref|NM_000002.1| B.subtilis 16S rRNA gene  
ACGAGTGCGAACGCTGGCGGCAGGCCTAACACATGCAAGTCGAACGGCAGCACGGGGA
AGTAGCTTGCTACTTTGCCGGCGAGCGGCGGACGGGTGAGTAATGTCTGGGAAACTGC
CTGATGGAGGGGGATAACTACTGGAAACGGTAGCTAATACCGCATAACGTCGCAAGAC
"""

with open('sample_sequences.fasta', 'w') as f:
    f.write(fasta_content)
```

**Tasks**:
1. Read FASTA file and extract sequence IDs and descriptions
2. Calculate GC content for each sequence
3. Find the longest and shortest sequences
4. Convert DNA sequences to reverse complement
5. Translate DNA sequences to proteins in all 6 reading frames

### Exercise 1.2: Sequence Manipulation and Analysis
```python
from Bio.SeqUtils import GC, molecular_weight
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# Sample bacterial gene sequence (16S rRNA partial)
gene_sequence = """
AGAGTTTGATCCTGGCTCAGATTGAACGCTGGCGGCAGGCCTAACACATGCAAGTCGA
ACGGCAGCACGGGGAGTAGCTTGCTCTCGGGTGACGAGCGGCGGACGGGTGAGTAATG
TCTGGGAAACTGCCTGATGGAGGGGGATAACTACTGGAAACGGTAGCTAATACCGCAT
AACGTCGCAAGACCAAAGAGGGGGACCTTCGGGCCTCTTGCCATCGGATGTGCCCAGA
TGGGATTAGCTAGTAGGTGGGGTAACGGCTCACCTAGGCGACGATCCCTAGCTGGTCT
"""

# Remove whitespace and newlines
clean_sequence = gene_sequence.replace('\n', '').replace(' ', '')
```

**Tasks**:
1. Find all ORFs (Open Reading Frames) longer than 150 bp
2. Identify restriction enzyme cut sites (EcoRI, BamHI, HindIII)
3. Calculate melting temperature for PCR primers
4. Design primers for the sequence (20-25 bp, GC content 40-60%)
5. Analyze codon usage bias

### Exercise 1.3: BLAST Searches and Result Parsing
```python
from Bio.Blast import NCBIWWW, NCBIXML
from Bio import Entrez
import time

# Set your email for NCBI Entrez
Entrez.email = "your.email@institution.edu"

# Sample query sequence
query_sequence = """
ACGAGTGCGAACGCTGGCGGCAGGCCTAACACATGCAAGTCGAACGGCAGCACGGGGA
AGTAGCTTGCTACTTTGCCGGCGAGCGGCGGACGGGTGAGTAATGTCTGGGAAACTGC
"""

def perform_blast_search(sequence, database="nt", program="blastn"):
    """Perform BLAST search and return results"""
    try:
        result_handle = NCBIWWW.qblast(program, database, sequence)
        return result_handle
    except Exception as e:
        print(f"BLAST search failed: {e}")
        return None
```

**Tasks**:
1. Implement automated BLAST search with error handling
2. Parse BLAST XML results and extract top 10 hits
3. Filter results by E-value threshold (< 1e-5)
4. Create summary table with accession numbers, descriptions, and scores
5. Download sequences of top hits using Entrez

### Exercise 1.4: Protein and DNA Sequence Analysis
```python
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SeqUtils import CodonUsage

# Sample protein sequence (β-lactamase)
protein_seq = """
MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLLCGAVLSRIDAGQEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGDHVTRLDRWEPELNEAIPNDERDTTMPVAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFIADKSGAGERGSRGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW
"""

# Codon usage analysis
codon_table = CodonUsage.CodonsDict.copy()
```

**Tasks**:
1. Calculate protein properties (MW, pI, hydrophobicity)
2. Predict secondary structure elements
3. Identify potential transmembrane domains
4. Analyze amino acid composition and frequency
5. Compare codon usage between different organisms

---

## Section 2: Phylogenetics and Evolutionary Analysis

### Exercise 2.1: Multiple Sequence Alignment Processing
```python
from Bio import AlignIO, Phylo
from Bio.Align.Applications import ClustalwCommandline
import numpy as np

# Sample multiple sequence alignment (16S rRNA fragments)
alignment_data = """>E.coli
ACGAGTGCGAACGCTGGCGGCAGGCCTAACACATGCAAGTCGA
>B.subtilis  
ACGAGTGCGAACGCTGGCGGCAGGCCTAACACATGCAAGTCGA
>S.aureus
ACGAGTGCGAACGCTGGCGGCAGGCCTAACACATGCAAGTCGA
>P.aeruginosa
ACGAGTGCGAACGCTGGCGGCAGGCCTAACACATGCAAGTCGA
"""

with open('sample_alignment.fasta', 'w') as f:
    f.write(alignment_data)
```

**Tasks**:
1. Read and parse multiple sequence alignments
2. Calculate alignment statistics (length, gaps, conservation)
3. Identify conserved and variable regions
4. Remove poorly aligned sequences or regions
5. Convert between alignment formats (FASTA, PHYLIP, CLUSTAL)

### Exercise 2.2: Distance Matrices and Clustering
```python
from Bio import Phylo
from Bio.Phylo.TreeConstruction import DistanceCalculator, NeighborJoining
import matplotlib.pyplot as plt

def calculate_pairwise_distances(alignment):
    """Calculate pairwise distances from alignment"""
    calculator = DistanceCalculator('identity')
    distance_matrix = calculator.get_distance(alignment)
    return distance_matrix

def build_nj_tree(distance_matrix):
    """Build Neighbor-Joining tree"""
    constructor = NeighborJoining()
    tree = constructor.build_tree(distance_matrix)
    return tree
```

**Tasks**:
1. Calculate different distance metrics (Jukes-Cantor, Kimura 2-parameter)
2. Build distance matrices for multiple sequences
3. Construct phylogenetic trees using different methods
4. Compare tree topologies and bootstrap support
5. Root trees using outgroup sequences

### Exercise 2.3: Tree Construction and Visualization
```python
from Bio.Phylo.TreeConstruction import ParsimonyScorer, NNITreeSearcher
from Bio import Phylo
import matplotlib.pyplot as plt

def visualize_tree(tree, title="Phylogenetic Tree"):
    """Visualize phylogenetic tree"""
    fig, ax = plt.subplots(figsize=(12, 8))
    Phylo.draw(tree, axes=ax)
    ax.set_title(title)
    plt.tight_layout()
    return fig

def calculate_tree_statistics(tree):
    """Calculate basic tree statistics"""
    stats = {
        'total_length': tree.total_branch_length(),
        'terminal_count': len(tree.get_terminals()),
        'internal_count': len(tree.get_nonterminals()),
        'max_depth': tree.depths().values()
    }
    return stats
```

**Tasks**:
1. Implement maximum likelihood tree construction
2. Perform bootstrap analysis for tree support
3. Compare different tree building methods
4. Visualize trees with custom styling and annotations
5. Calculate evolutionary distances and divergence times

### Exercise 2.4: Phylogenetic Tree Interpretation
```python
from Bio.Phylo import BaseTree
import pandas as pd

def extract_tree_data(tree):
    """Extract taxonomic and evolutionary data from tree"""
    data = []
    for clade in tree.find_clades():
        if clade.name:
            data.append({
                'name': clade.name,
                'branch_length': clade.branch_length,
                'depth': tree.distance(tree.root, clade),
                'is_terminal': clade.is_terminal()
            })
    return pd.DataFrame(data)

def identify_clades(tree, threshold=0.1):
    """Identify monophyletic clades based on distance threshold"""
    clades = []
    for clade in tree.find_clades():
        if clade.branch_length and clade.branch_length > threshold:
            clades.append(clade)
    return clades
```

**Tasks**:
1. Extract evolutionary relationships from trees
2. Identify monophyletic groups and sister taxa
3. Calculate patristic distances between species
4. Annotate trees with taxonomic information
5. Perform ancestral state reconstruction

---

## Section 3: Genomics Data Analysis

### Exercise 3.1: Processing Genomic Coordinates
```python
import pandas as pd
from collections import namedtuple

# Define genomic interval structure
GenomicInterval = namedtuple('GenomicInterval', ['chromosome', 'start', 'end', 'strand', 'name'])

# Sample genomic coordinates
genomic_features = [
    GenomicInterval('chr1', 1000, 2000, '+', 'gene1'),
    GenomicInterval('chr1', 1500, 2500, '-', 'gene2'),
    GenomicInterval('chr1', 3000, 4000, '+', 'gene3'),
    GenomicInterval('chr2', 500, 1500, '+', 'gene4'),
]

def interval_overlap(interval1, interval2):
    """Check if two genomic intervals overlap"""
    if interval1.chromosome != interval2.chromosome:
        return False
    return not (interval1.end < interval2.start or interval2.end < interval1.start)
```

**Tasks**:
1. Find overlapping genomic features
2. Calculate distances between genes
3. Identify genes within specific regions
4. Convert between different coordinate systems
5. Merge overlapping intervals

### Exercise 3.2: Working with Annotation Files (GFF/GTF)
```python
import pandas as pd

# Sample GFF3 data
gff_data = """##gff-version 3
chr1	RefSeq	gene	1000	2000	.	+	.	ID=gene1;Name=rpoA;product=RNA polymerase subunit alpha
chr1	RefSeq	CDS	1000	2000	.	+	0	ID=cds1;Parent=gene1
chr1	RefSeq	gene	3000	4000	.	-	.	ID=gene2;Name=gyrA;product=DNA gyrase subunit A
chr1	RefSeq	CDS	3000	4000	.	-	0	ID=cds2;Parent=gene2
chr2	RefSeq	gene	500	1500	.	+	.	ID=gene3;Name=recA;product=recombinase A
"""

def parse_gff_attributes(attr_string):
    """Parse GFF3 attribute column"""
    attributes = {}
    for attr in attr_string.split(';'):
        if '=' in attr:
            key, value = attr.split('=', 1)
            attributes[key] = value
    return attributes

def load_gff_file(filename):
    """Load and parse GFF file"""
    columns = ['seqname', 'source', 'feature', 'start', 'end', 
               'score', 'strand', 'frame', 'attribute']
    df = pd.read_csv(filename, sep='\t', comment='#', names=columns)
    return df
```

**Tasks**:
1. Parse GFF/GTF files and extract gene annotations
2. Filter features by type (gene, CDS, exon, etc.)
3. Calculate gene lengths and intergenic distances
4. Find genes on specific strands or chromosomes
5. Export filtered annotations to new formats

### Exercise 3.3: Basic Comparative Genomics
```python
from Bio import SeqIO
import pandas as pd

def compare_gene_content(genome1_gff, genome2_gff):
    """Compare gene content between two genomes"""
    # Extract gene names from both genomes
    genes1 = set()
    genes2 = set()
    
    # Implementation here
    pass

def find_syntenic_regions(genome1_coords, genome2_coords, window_size=5):
    """Identify syntenic regions between genomes"""
    syntenic_blocks = []
    # Implementation here
    return syntenic_blocks

# Sample ortholog data
ortholog_data = {
    'genome1_gene': ['gene1', 'gene2', 'gene3', 'gene4'],
    'genome2_gene': ['geneA', 'geneB', 'geneC', None],
    'identity': [95.2, 87.4, 92.1, None],
    'coverage': [98.5, 89.2, 94.7, None]
}

orthologs_df = pd.DataFrame(ortholog_data)
```

**Tasks**:
1. Identify orthologous genes between genomes
2. Calculate genome synteny and rearrangements
3. Find genome-specific genes and gene families
4. Analyze horizontal gene transfer events
5. Visualize comparative genomic maps

### Exercise 3.4: Gene Expression Data Handling
```python
import pandas as pd
import numpy as np

# Sample RNA-seq expression data
np.random.seed(42)
genes = [f'gene_{i:04d}' for i in range(1, 1001)]
conditions = ['control', 'treatment_1h', 'treatment_6h', 'treatment_24h']
samples = [f'{cond}_rep{rep}' for cond in conditions for rep in range(1, 4)]

# Generate mock expression data
expression_data = np.random.lognormal(mean=5, sigma=1.5, size=(len(genes), len(samples)))
expression_df = pd.DataFrame(expression_data, index=genes, columns=samples)

def normalize_expression(df, method='tpm'):
    """Normalize expression data"""
    if method == 'tpm':
        # Transcripts per million normalization
        return df.div(df.sum()) * 1e6
    elif method == 'log2':
        # Log2 transformation
        return np.log2(df + 1)
    return df

def identify_deg(control_samples, treatment_samples, fold_change_threshold=2):
    """Identify differentially expressed genes"""
    control_mean = control_samples.mean(axis=1)
    treatment_mean = treatment_samples.mean(axis=1)
    fold_change = treatment_mean / control_mean
    
    deg_genes = fold_change[fold_change > fold_change_threshold].index
    return deg_genes
```

**Tasks**:
1. Normalize expression data using different methods
2. Identify differentially expressed genes
3. Perform clustering analysis of expression patterns
4. Create heatmaps and expression profiles
5. Annotate genes with functional categories

---

## Section 4: Microbiome Analysis Preparation

### Exercise 4.1: OTU Tables and Taxonomy
```python
import pandas as pd
import numpy as np

# Sample OTU table
np.random.seed(42)
n_samples = 20
n_otus = 50

# Generate sample names
sample_names = [f'Sample_{i:03d}' for i in range(1, n_samples + 1)]

# Generate OTU abundances (using negative binomial distribution)
otu_data = np.random.negative_binomial(10, 0.3, size=(n_samples, n_otus))
otu_df = pd.DataFrame(otu_data, index=sample_names, columns=[f'OTU_{i:04d}' for i in range(1, n_otus + 1)])

# Sample taxonomy data
taxonomy_levels = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
bacterial_taxa = [
    'Bacteria;Firmicutes;Clostridia;Clostridiales;Lachnospiraceae;Blautia;unknown',
    'Bacteria;Bacteroidetes;Bacteroidia;Bacteroidales;Bacteroidaceae;Bacteroides;fragilis',
    'Bacteria;Firmicutes;Bacilli;Lactobacillales;Lactobacillaceae;Lactobacillus;acidophilus',
    'Bacteria;Proteobacteria;Gammaproteobacteria;Enterobacteriales;Enterobacteriaceae;Escherichia;coli'
]

def parse_taxonomy(taxonomy_string):
    """Parse taxonomy string into dictionary"""
    levels = taxonomy_string.split(';')
    return dict(zip(taxonomy_levels[:len(levels)], levels))

def collapse_taxonomy(otu_df, taxonomy_df, level='Genus'):
    """Collapse OTU table to specific taxonomic level"""
    # Implementation here
    pass
```

**Tasks**:
1. Load and validate OTU tables and taxonomy files
2. Filter low-abundance OTUs and rare taxa
3. Collapse OTU tables to different taxonomic levels
4. Calculate total reads per sample and OTU
5. Create taxonomy summary tables

### Exercise 4.2: Alpha and Beta Diversity
```python
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
import numpy as np

def shannon_diversity(counts):
    """Calculate Shannon diversity index"""
    proportions = counts / counts.sum()
    proportions = proportions[proportions > 0]  # Remove zeros
    return entropy(proportions, base=2)

def simpson_diversity(counts):
    """Calculate Simpson diversity index"""
    n = counts.sum()
    return 1 - sum((counts * (counts - 1)) / (n * (n - 1)))

def chao1_richness(counts):
    """Calculate Chao1 richness estimator"""
    observed = (counts > 0).sum()
    singletons = (counts == 1).sum()
    doubletons = (counts == 2).sum()
    
    if doubletons > 0:
        chao1 = observed + (singletons ** 2) / (2 * doubletons)
    else:
        chao1 = observed + (singletons * (singletons - 1)) / 2
    return chao1

def bray_curtis_distance(sample1, sample2):
    """Calculate Bray-Curtis dissimilarity"""
    numerator = np.sum(np.abs(sample1 - sample2))
    denominator = np.sum(sample1 + sample2)
    return numerator / denominator if denominator > 0 else 0
```

**Tasks**:
1. Calculate multiple alpha diversity metrics for each sample
2. Compute beta diversity distances between samples
3. Perform rarefaction analysis
4. Compare diversity between sample groups
5. Visualize diversity patterns with boxplots and PCoA

### Exercise 4.3: Data Normalization Techniques
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def relative_abundance_transform(otu_df):
    """Convert counts to relative abundances"""
    return otu_df.div(otu_df.sum(axis=1), axis=0)

def clr_transform(otu_df, pseudocount=1):
    """Centered log-ratio transformation"""
    # Add pseudocount to handle zeros
    otu_plus = otu_df + pseudocount
    
    # Calculate geometric mean
    geom_mean = np.exp(np.log(otu_plus).mean(axis=1))
    
    # CLR transformation
    clr_data = np.log(otu_plus.div(geom_mean, axis=0))
    return pd.DataFrame(clr_data, index=otu_df.index, columns=otu_df.columns)

def rarefy_samples(otu_df, depth=None):
    """Rarefy samples to equal sequencing depth"""
    if depth is None:
        depth = otu_df.sum(axis=1).min()
    
    rarefied_data = []
    for idx, sample in otu_df.iterrows():
        # Randomly subsample reads
        reads = []
        for otu, count in sample.items():
            reads.extend([otu] * int(count))
        
        if len(reads) >= depth:
            subsampled = np.random.choice(reads, depth, replace=False)
            rarefied_counts = pd.Series(subsampled).value_counts()
            rarefied_sample = sample.copy()
            rarefied_sample[:] = 0
            rarefied_sample[rarefied_counts.index] = rarefied_counts.values
            rarefied_data.append(rarefied_sample)
    
    return pd.DataFrame(rarefied_data)
```

**Tasks**:
1. Apply different normalization methods to OTU data
2. Compare the effects of normalization on diversity metrics
3. Implement variance stabilizing transformations
4. Handle compositional data appropriately
5. Evaluate normalization methods for downstream analysis

### Exercise 4.4: Sample Metadata Management
```python
import pandas as pd
from datetime import datetime, timedelta

# Sample metadata
metadata_dict = {
    'sample_id': [f'Sample_{i:03d}' for i in range(1, 21)],
    'patient_id': [f'P{i:03d}' for i in range(1, 11)] * 2,  # Two samples per patient
    'timepoint': ['baseline', 'follow_up'] * 10,
    'age': np.random.randint(18, 80, 20),
    'bmi': np.random.normal(25, 5, 20),
    'antibiotic_use': np.random.choice(['none', 'recent', 'current'], 20),
    'disease_status': np.random.choice(['healthy', 'IBD', 'IBS'], 20),
    'collection_date': [datetime(2023, 1, 1) + timedelta(days=x) for x in range(20)]
}

metadata_df = pd.DataFrame(metadata_dict)

def validate_metadata(metadata_df, otu_df):
    """Validate metadata against OTU table"""
    issues = []
    
    # Check sample ID matching
    metadata_samples = set(metadata_df['sample_id'])
    otu_samples = set(otu_df.index)
    
    if metadata_samples != otu_samples:
        missing_in_metadata = otu_samples - metadata_samples
        missing_in_otu = metadata_samples - otu_samples
        
        if missing_in_metadata:
            issues.append(f"Samples in OTU table but not metadata: {missing_in_metadata}")
        if missing_in_otu:
            issues.append(f"Samples in metadata but not OTU table: {missing_in_otu}")
    
    return issues

def create_design_matrix(metadata_df, variables):
    """Create design matrix for statistical analysis"""
    design_df = metadata_df[['sample_id'] + variables].copy()
    
    # Encode categorical variables
    categorical_vars = design_df.select_dtypes(include=['object']).columns
    for var in categorical_vars:
        if var != 'sample_id':
            dummies = pd.get_dummies(design_df[var], prefix=var)
            design_df = pd.concat([design_df, dummies], axis=1)
            design_df.drop(var, axis=1, inplace=True)
    
    return design_df
```

**Tasks**:
1. Load and validate sample metadata
2. Handle missing values and data inconsistencies
3. Create categorical and continuous variable encodings
4. Link metadata with microbiome data
5. Generate summary statistics for different sample groups

---

## Comprehensive Project: Bacterial Genome Analysis Pipeline

### Project Overview
Create an integrated analysis pipeline combining all Phase 3 concepts for analyzing bacterial genome data.

### Dataset Components
- Bacterial genome sequences (FASTA format)
- Gene annotation files (GFF3 format)
- 16S rRNA sequences for phylogenetic analysis
- Expression data from different growth conditions
- Comparative genomics data from related species

### Pipeline Components

#### 1. Genome Sequence Analysis
```python
# Analyze genome composition and features
# Identify gene content and functional categories
# Calculate genome statistics and metrics
```

#### 2. Phylogenetic Analysis
```python
# Construct phylogenetic trees from 16S rRNA
# Analyze evolutionary relationships
# Identify strain-specific variations
```

#### 3. Comparative Genomics
```python
# Compare genomes for synteny and rearrangements
# Identify unique genes and gene families
# Analyze horizontal gene transfer events
```

#### 4. Expression Analysis
```python
# Process RNA-seq data
# Identify differentially expressed genes
# Correlate expression with genomic features
```

### Deliverables
1. **Genome Analysis Report**: Comprehensive genome characterization
2. **Phylogenetic Trees**: Publication-ready phylogenetic analysis
3. **Comparative Maps**: Visual comparison of related genomes
4. **Expression Profiles**: Gene expression patterns and regulation
5. **Integrated Database**: Searchable genome and annotation database

---

## Challenge Exercises

### Challenge 1: Automated Phylogenetic Pipeline
Create a command-line tool that:
- Downloads sequences from NCBI
- Performs multiple sequence alignment
- Constructs phylogenetic trees
- Generates publication-ready figures
- Outputs evolutionary statistics

### Challenge 2: Microbiome Data Processor
Build a microbiome analysis pipeline that:
- Validates and cleans OTU tables
- Calculates diversity metrics
- Performs statistical comparisons
- Generates interactive visualizations
- Exports results in multiple formats

### Challenge 3: Genome Annotation Toolkit
Develop a genome annotation system that:
- Predicts genes and their functions
- Identifies regulatory elements
- Compares annotations across genomes
- Integrates expression data
- Provides web-based visualization

---

## Evaluation Criteria

### BioPython Proficiency (25%)
- Sequence manipulation and analysis
- File format handling and conversion
- BLAST search implementation
- Protein analysis techniques

### Phylogenetic Analysis (25%)
- Tree construction methods
- Distance calculation and interpretation
- Visualization and annotation
- Statistical analysis of trees

### Genomics Skills (25%)
- Coordinate system handling
- Annotation file processing
- Comparative analysis techniques
- Expression data integration

### Microbiome Analysis (25%)
- OTU table processing and validation
- Diversity metric calculations
- Data normalization techniques
- Metadata integration and management

## Next Steps
After completing Phase 3, you'll be ready for:
- **Phase 4**: Advanced statistical analysis and machine learning
- **Specialized Tools**: Domain-specific bioinformatics packages
- **Research Projects**: Independent genomics and microbiome studies
- **Publication**: Presenting bioinformatics results in scientific literature

Good luck with Phase 3! You're now working with real bioinformatics data and techniques used in research laboratories worldwide. 🧬🔬🐍
