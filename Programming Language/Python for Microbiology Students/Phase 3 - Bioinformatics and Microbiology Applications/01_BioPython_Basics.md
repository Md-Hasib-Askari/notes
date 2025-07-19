# BioPython Basics

BioPython is the essential library for handling biological data in Python, providing tools for sequence analysis, file format handling, and bioinformatics operations.

## Installation and Setup

```bash
# Install BioPython
pip install biopython

# Optional: Install additional dependencies
pip install matplotlib numpy pandas
```

## Sequence Objects and Basic Operations

### Creating Sequence Objects

```python
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import IUPAC

# Create DNA sequence
dna_seq = Seq("ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG")
print(f"DNA sequence: {dna_seq}")
print(f"Length: {len(dna_seq)} bp")

# Create protein sequence
protein_seq = Seq("MVIVMGRIDRSGLKVKF")
print(f"Protein sequence: {protein_seq}")

# Create RNA sequence
rna_seq = Seq("AUGGCCAUUGUAAUGGGCCGCUGAAAGGGUGCCCGAUAG")
print(f"RNA sequence: {rna_seq}")
```

### Basic Sequence Operations

```python
# Transcription (DNA to RNA)
rna_from_dna = dna_seq.transcribe()
print(f"Transcribed RNA: {rna_from_dna}")

# Translation (DNA/RNA to protein)
protein_from_dna = dna_seq.translate()
protein_from_rna = rna_seq.translate()
print(f"Translated protein (DNA): {protein_from_dna}")
print(f"Translated protein (RNA): {protein_from_rna}")

# Reverse complement
reverse_comp = dna_seq.reverse_complement()
print(f"Reverse complement: {reverse_comp}")

# GC content calculation
def gc_content(sequence):
    """Calculate GC content of DNA sequence"""
    gc_count = sequence.count('G') + sequence.count('C')
    return (gc_count / len(sequence)) * 100

gc_percent = gc_content(dna_seq)
print(f"GC content: {gc_percent:.1f}%")
```

### Working with SeqRecord Objects

```python
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import SeqFeature, FeatureLocation

# Create a SeqRecord with metadata
record = SeqRecord(
    dna_seq,
    id="sample_001",
    name="hypothetical_gene",
    description="Sample bacterial gene sequence",
    annotations={
        "organism": "Escherichia coli",
        "strain": "K12",
        "isolation_source": "laboratory culture",
        "country": "USA"
    }
)

print(f"Record ID: {record.id}")
print(f"Description: {record.description}")
print(f"Organism: {record.annotations['organism']}")

# Add features to the sequence
gene_feature = SeqFeature(
    FeatureLocation(0, len(dna_seq)),
    type="gene",
    qualifiers={"gene": ["hypothetical"], "product": ["hypothetical protein"]}
)

record.features.append(gene_feature)
print(f"Number of features: {len(record.features)}")
```

## File Format Handling

### Reading FASTA Files

```python
from Bio import SeqIO
import os

# Read single sequence from FASTA file
def read_single_fasta(filename):
    """Read a single sequence from FASTA file"""
    try:
        record = SeqIO.read(filename, "fasta")
        print(f"ID: {record.id}")
        print(f"Description: {record.description}")
        print(f"Length: {len(record.seq)} bp")
        print(f"Sequence: {record.seq[:50]}...")  # First 50 bases
        return record
    except FileNotFoundError:
        print(f"File {filename} not found")
        return None

# Read multiple sequences from FASTA file
def read_multiple_fasta(filename):
    """Read multiple sequences from FASTA file"""
    sequences = []
    try:
        for record in SeqIO.parse(filename, "fasta"):
            sequences.append(record)
            print(f"Loaded: {record.id} ({len(record.seq)} bp)")
        
        print(f"Total sequences loaded: {len(sequences)}")
        return sequences
    except FileNotFoundError:
        print(f"File {filename} not found")
        return []

# Example usage (uncomment when you have FASTA files)
# single_seq = read_single_fasta("sample_gene.fasta")
# multiple_seqs = read_multiple_fasta("bacterial_genomes.fasta")
```

### Writing FASTA Files

```python
# Create sample sequences for demonstration
sample_sequences = [
    SeqRecord(Seq("ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG"), 
              id="gene_001", description="E. coli gene 1"),
    SeqRecord(Seq("ATGAAACGCATTAGCACCACCATTACCACCACCATCACCATTACCACAGGTAACGGTGCGGGCTGA"), 
              id="gene_002", description="E. coli gene 2"),
    SeqRecord(Seq("ATGGCAATTATTAAAGAACGTCTGATGATTCAGACCAAAGATGAAATGGCTTATACGCTTGG"), 
              id="gene_003", description="E. coli gene 3")
]

# Write sequences to FASTA file
output_file = "sample_genes.fasta"
SeqIO.write(sample_sequences, output_file, "fasta")
print(f"Written {len(sample_sequences)} sequences to {output_file}")

# Read back and verify
verification_seqs = list(SeqIO.parse(output_file, "fasta"))
print(f"Verification: Read {len(verification_seqs)} sequences")
```

### Working with GenBank Files

```python
# Create a more detailed GenBank record
from Bio.SeqFeature import SeqFeature, FeatureLocation

# Create a sample GenBank-style record
genbank_record = SeqRecord(
    Seq("ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG"),
    id="SAMPLE_001",
    name="sample_gene",
    description="Sample bacterial gene with annotations",
    annotations={
        "organism": "Escherichia coli",
        "strain": "K12",
        "mol_type": "genomic DNA",
        "date": "15-OCT-2023"
    }
)

# Add gene feature
gene_feature = SeqFeature(
    FeatureLocation(0, 39),
    type="gene",
    qualifiers={
        "gene": ["sampleGene"],
        "locus_tag": ["ECK_0001"]
    }
)

# Add CDS feature
cds_feature = SeqFeature(
    FeatureLocation(0, 39),
    type="CDS",
    qualifiers={
        "gene": ["sampleGene"],
        "product": ["hypothetical protein"],
        "protein_id": ["ABC12345.1"],
        "translation": [str(genbank_record.seq.translate())]
    }
)

genbank_record.features.extend([gene_feature, cds_feature])

# Write GenBank file
SeqIO.write(genbank_record, "sample_gene.gbk", "genbank")
print("GenBank file created: sample_gene.gbk")
```

## Sequence Manipulation and Analysis

### Finding Open Reading Frames (ORFs)

```python
def find_orfs(sequence, min_length=30):
    """Find all ORFs in a DNA sequence"""
    start_codon = "ATG"
    stop_codons = ["TAA", "TAG", "TGA"]
    orfs = []
    
    # Check all three reading frames
    for frame in range(3):
        frame_seq = sequence[frame:]
        
        i = 0
        while i < len(frame_seq) - 2:
            if frame_seq[i:i+3] == start_codon:
                # Found start codon, look for stop codon
                for j in range(i + 3, len(frame_seq) - 2, 3):
                    codon = frame_seq[j:j+3]
                    if codon in stop_codons:
                        orf_length = j + 3 - i
                        if orf_length >= min_length:
                            orf_seq = frame_seq[i:j+3]
                            protein_seq = Seq(orf_seq).translate()
                            orfs.append({
                                'frame': frame + 1,
                                'start': frame + i,
                                'end': frame + j + 3,
                                'length': orf_length,
                                'dna_seq': orf_seq,
                                'protein_seq': str(protein_seq)
                            })
                        break
                i = j + 1 if 'j' in locals() else i + 1
            else:
                i += 1
    
    return orfs

# Example usage
test_sequence = Seq("ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAGCATGAAACGCATTAGCACCACCATTACCACCACCATCACCATTACCACAGGTAACGGTGCGGGCTGA")
orfs_found = find_orfs(test_sequence)

print(f"Found {len(orfs_found)} ORFs:")
for i, orf in enumerate(orfs_found, 1):
    print(f"ORF {i}: Frame {orf['frame']}, Position {orf['start']}-{orf['end']}, Length {orf['length']} bp")
    print(f"  Protein: {orf['protein_seq']}")
```

### Sequence Composition Analysis

```python
def analyze_sequence_composition(sequence):
    """Comprehensive sequence composition analysis"""
    seq_str = str(sequence).upper()
    length = len(seq_str)
    
    # Nucleotide counts
    composition = {
        'A': seq_str.count('A'),
        'T': seq_str.count('T'),
        'G': seq_str.count('G'),
        'C': seq_str.count('C')
    }
    
    # Percentages
    percentages = {base: (count/length)*100 for base, count in composition.items()}
    
    # GC content
    gc_content = (composition['G'] + composition['C']) / length * 100
    
    # AT/GC ratio
    at_content = (composition['A'] + composition['T']) / length * 100
    
    # Dinucleotide analysis
    dinucleotides = {}
    for i in range(length - 1):
        dinuc = seq_str[i:i+2]
        dinucleotides[dinuc] = dinucleotides.get(dinuc, 0) + 1
    
    return {
        'length': length,
        'composition': composition,
        'percentages': percentages,
        'gc_content': gc_content,
        'at_content': at_content,
        'dinucleotides': dinucleotides
    }

# Analyze sample sequence
analysis = analyze_sequence_composition(dna_seq)
print("Sequence Composition Analysis:")
print(f"Length: {analysis['length']} bp")
print(f"GC content: {analysis['gc_content']:.1f}%")
print(f"AT content: {analysis['at_content']:.1f}%")
print("\nNucleotide composition:")
for base, percentage in analysis['percentages'].items():
    print(f"  {base}: {percentage:.1f}%")
```

### Codon Usage Analysis

```python
def analyze_codon_usage(sequence):
    """Analyze codon usage in a coding sequence"""
    seq_str = str(sequence).upper()
    
    # Ensure sequence length is multiple of 3
    if len(seq_str) % 3 != 0:
        seq_str = seq_str[:-(len(seq_str) % 3)]
    
    codons = {}
    for i in range(0, len(seq_str), 3):
        codon = seq_str[i:i+3]
        codons[codon] = codons.get(codon, 0) + 1
    
    total_codons = sum(codons.values())
    
    # Calculate frequencies
    codon_frequencies = {codon: count/total_codons for codon, count in codons.items()}
    
    return {
        'total_codons': total_codons,
        'codon_counts': codons,
        'codon_frequencies': codon_frequencies
    }

# Analyze codon usage
codon_analysis = analyze_codon_usage(dna_seq)
print("Codon Usage Analysis:")
print(f"Total codons: {codon_analysis['total_codons']}")
print("Most frequent codons:")
sorted_codons = sorted(codon_analysis['codon_frequencies'].items(), 
                      key=lambda x: x[1], reverse=True)
for codon, freq in sorted_codons[:5]:
    print(f"  {codon}: {freq:.3f} ({codon_analysis['codon_counts'][codon]} occurrences)")
```

## Best Practices for Sequence Analysis

1. **Always validate input sequences** before analysis
2. **Handle different sequence formats** appropriately
3. **Use appropriate sequence alphabets** (DNA, RNA, protein)
4. **Consider reverse complement** for DNA analysis
5. **Implement error handling** for file operations
6. **Document sequence sources** and modifications
7. **Use SeqRecord objects** for rich sequence annotation
8. **Preserve sequence metadata** throughout analysis pipeline
