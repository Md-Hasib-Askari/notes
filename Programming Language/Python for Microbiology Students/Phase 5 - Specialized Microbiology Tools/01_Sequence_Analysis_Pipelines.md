# Sequence Analysis Pipelines

## Automating BLAST Searches
Automate BLAST searches for sequence similarity analysis using BioPython.

```python
from Bio.Blast import NCBIWWW, NCBIXML
from Bio import SeqIO

def automated_blast_search(sequence, database="nr", program="blastp"):
    """Perform automated BLAST search"""
    # Submit BLAST job
    result_handle = NCBIWWW.qblast(program, database, sequence)
    
    # Parse results
    blast_record = NCBIXML.read(result_handle)
    
    # Extract top hits
    top_hits = []
    for alignment in blast_record.alignments[:5]:  # Top 5 hits
        for hsp in alignment.hsps:
            hit_info = {
                'title': alignment.title,
                'e_value': hsp.expect,
                'score': hsp.score,
                'identity': hsp.identities / hsp.align_length
            }
            top_hits.append(hit_info)
            break  # Only first HSP per alignment
    
    return top_hits

# Example usage
sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAYVBFYYDDAQYAAYDKDLSRALVS"
hits = automated_blast_search(sequence)
```

## Primer Design and PCR Simulation
Design PCR primers and simulate amplification using Primer3 and BioPython.

```python
from Bio import SeqIO
from Bio.SeqUtils import MeltingTemp as mt
import primer3

def design_primers(target_sequence, product_size_range=(100, 300)):
    """Design PCR primers for target sequence"""
    
    primer_design = {
        'SEQUENCE_ID': 'target',
        'SEQUENCE_TEMPLATE': str(target_sequence),
        'PRIMER_PRODUCT_SIZE_RANGE': [[product_size_range[0], product_size_range[1]]],
        'PRIMER_OPT_SIZE': 20,
        'PRIMER_MIN_SIZE': 18,
        'PRIMER_MAX_SIZE': 25,
        'PRIMER_OPT_TM': 60.0,
        'PRIMER_MIN_TM': 57.0,
        'PRIMER_MAX_TM': 63.0
    }
    
    # Design primers
    primers = primer3.bindings.designPrimers(primer_design)
    
    # Extract primer sequences
    primer_results = []
    for i in range(primers['PRIMER_PAIR_NUM_RETURNED']):
        left_seq = primers[f'PRIMER_LEFT_{i}_SEQUENCE']
        right_seq = primers[f'PRIMER_RIGHT_{i}_SEQUENCE']
        
        primer_results.append({
            'forward': left_seq,
            'reverse': right_seq,
            'tm_forward': mt.Tm_NN(left_seq),
            'tm_reverse': mt.Tm_NN(right_seq),
            'product_size': primers[f'PRIMER_PAIR_{i}_PRODUCT_SIZE']
        })
    
    return primer_results
```

## Restriction Enzyme Analysis
Analyze restriction enzyme cut sites in DNA sequences.

```python
from Bio import Restriction
from Bio.Seq import Seq

def analyze_restriction_sites(dna_sequence, enzymes=None):
    """Analyze restriction enzyme cut sites"""
    
    if enzymes is None:
        # Common enzymes
        enzymes = [Restriction.EcoRI, Restriction.BamHI, Restriction.HindIII]
    
    seq = Seq(dna_sequence)
    
    analysis_results = {}
    for enzyme in enzymes:
        cuts = enzyme.search(seq)
        analysis_results[enzyme.site] = {
            'enzyme_name': str(enzyme),
            'recognition_site': enzyme.site,
            'cut_positions': cuts,
            'fragment_count': len(cuts) + 1 if cuts else 1
        }
    
    return analysis_results

# Example usage
dna = "GAATTCAAGCTTGGATCCAAGCTTGAATTC"
results = analyze_restriction_sites(dna)
```

## ORF Finding and Gene Prediction
Identify open reading frames (ORFs) in DNA sequences.

```python
from Bio.Seq import Seq
from Bio.SeqUtils.ProtParam import ProteinAnalysis

def find_orfs(dna_sequence, min_length=100):
    """Find ORFs in DNA sequence"""
    
    sequence = Seq(dna_sequence)
    orfs = []
    
    # Check all 6 reading frames
    for strand, seq in [(+1, sequence), (-1, sequence.reverse_complement())]:
        for frame in range(3):
            # Translate sequence
            protein = seq[frame:].translate()
            
            # Find ORFs between start and stop codons
            start = 0
            while start < len(protein):
                # Find start codon (M)
                start_pos = protein.find('M', start)
                if start_pos == -1:
                    break
                
                # Find stop codon
                stop_pos = start_pos
                while stop_pos < len(protein) and protein[stop_pos] != '*':
                    stop_pos += 1
                
                # Check ORF length
                orf_length = (stop_pos - start_pos) * 3
                if orf_length >= min_length:
                    orf_sequence = protein[start_pos:stop_pos]
                    
                    orfs.append({
                        'strand': strand,
                        'frame': frame + 1,
                        'start': start_pos * 3 + frame,
                        'end': stop_pos * 3 + frame,
                        'length': orf_length,
                        'protein': str(orf_sequence)
                    })
                
                start = start_pos + 1
    
    return sorted(orfs, key=lambda x: x['length'], reverse=True)
```

These pipeline tools automate common molecular biology tasks, enabling efficient sequence analysis workflows.
