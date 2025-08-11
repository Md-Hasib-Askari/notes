# Phase 1 Exercises: Python Fundamentals for Microbiology

## Overview
These exercises are designed to reinforce Python fundamentals with microbiology-focused examples. Complete each section progressively, as later exercises build upon earlier concepts.

---

## Section 1: Basic Syntax and Data Types

### Exercise 1.1: Variables and Basic Operations
```python
# Create variables for a bacterial culture experiment
culture_name = "E. coli K-12"
initial_od600 = 0.05
growth_rate = 0.693  # per hour
temperature = 37.0   # Celsius
ph_level = 7.2

# Calculate doubling time in minutes
doubling_time_hours = 0.693 / growth_rate
doubling_time_minutes = doubling_time_hours * 60

print(f"Culture: {culture_name}")
print(f"Doubling time: {doubling_time_minutes:.1f} minutes")
```

**Task**: Modify the code to calculate the OD600 after 4 hours of growth using the formula:
`final_od = initial_od * (2 ** (time / doubling_time))`

### Exercise 1.2: Working with Lists and Tuples
```python
# List of bacterial species in a microbiome sample
species_list = ["Lactobacillus", "Bifidobacterium", "Escherichia", "Bacteroides"]
abundance_counts = [1250, 890, 340, 2100]

# Tuple for sample metadata (immutable)
sample_info = ("Sample_001", "2023-07-15", "fecal", "healthy_adult")
```

**Tasks**:
1. Add two new species to the species list
2. Calculate the total abundance count
3. Find the most abundant species
4. Create a new tuple with updated sample information

### Exercise 1.3: Dictionaries and Sets
```python
# Dictionary mapping antibiotic names to their MIC values (µg/mL)
antibiotic_mic = {
    "ampicillin": 8.0,
    "tetracycline": 4.0,
    "chloramphenicol": 16.0,
    "streptomycin": 32.0
}

# Set of resistance genes found
resistance_genes = {"bla", "tet", "cat", "str"}
```

**Tasks**:
1. Add three new antibiotics with their MIC values
2. Check if "vancomycin" is in the resistance genes set
3. Create a function to categorize antibiotics as "sensitive" (<16) or "resistant" (≥16)
4. Count how many antibiotics fall into each category

---

## Section 2: Control Structures

### Exercise 2.1: Conditional Statements
```python
def interpret_gram_stain(cell_wall_thickness, color_retention):
    """
    Interpret Gram stain results based on cell wall thickness and color retention
    """
    # Your code here
    pass

# Test cases
test_cases = [
    (30, True),   # Thick wall, retains crystal violet
    (8, False),   # Thin wall, loses crystal violet
    (25, True),   # Thick wall, retains crystal violet
]
```

**Task**: Complete the function to return "Gram-positive" or "Gram-negative" based on the parameters.

### Exercise 2.2: Loops and Iterations
```python
# Growth curve data (time in hours, OD600 values)
growth_data = [
    (0, 0.05), (2, 0.08), (4, 0.15), (6, 0.31), 
    (8, 0.62), (10, 1.24), (12, 2.15), (14, 3.45), 
    (16, 4.12), (18, 4.28), (20, 4.31)
]
```

**Tasks**:
1. Use a for loop to find the time point where OD600 first exceeds 1.0
2. Calculate the average growth rate between consecutive time points
3. Identify the exponential growth phase (where growth rate is highest)

### Exercise 2.3: List Comprehensions
```python
# DNA sequence
dna_sequence = "ATGCGATCGATCGTAGCTAGCTA"

# Codon table (simplified)
codon_table = {
    "ATG": "Met", "TGA": "Stop", "TAA": "Stop", "TAG": "Stop",
    "GCT": "Ala", "GAT": "Asp", "CGT": "Arg", "CGA": "Arg"
}
```

**Tasks**:
1. Create a list of all codons (3-nucleotide groups) from the DNA sequence
2. Use list comprehension to translate codons to amino acids
3. Filter out stop codons using list comprehension
4. Count GC content using list comprehension

---

## Section 3: Functions and Modules

### Exercise 3.1: Basic Functions
```python
def calculate_molarity(mass_grams, molecular_weight, volume_liters):
    """
    Calculate molarity of a solution
    Args:
        mass_grams: mass of solute in grams
        molecular_weight: molecular weight in g/mol
        volume_liters: volume of solution in liters
    Returns:
        molarity in mol/L
    """
    # Your code here
    pass

def dilution_calculator(initial_concentration, initial_volume, final_volume):
    """
    Calculate final concentration after dilution using C1V1 = C2V2
    """
    # Your code here
    pass
```

**Tasks**:
1. Complete both functions
2. Test with realistic laboratory values
3. Add error handling for invalid inputs (negative values, zero volume)

### Exercise 3.2: Advanced Functions
```python
def growth_phase_analyzer(time_points, od_values, lag_threshold=0.1, 
                         stationary_threshold=0.05):
    """
    Analyze bacterial growth phases from OD600 data
    
    Args:
        time_points: list of time values
        od_values: list of corresponding OD600 measurements
        lag_threshold: minimum growth rate for exponential phase
        stationary_threshold: maximum growth rate for stationary phase
    
    Returns:
        dict with phase boundaries and characteristics
    """
    # Your implementation here
    pass
```

**Task**: Implement the function to identify lag, exponential, and stationary growth phases.

### Exercise 3.3: Creating Modules
Create a file called `microbio_utils.py` with the following functions:

```python
# microbio_utils.py
import math

def complement_dna(sequence):
    """Return complement of DNA sequence"""
    complement_map = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    return ''.join(complement_map.get(base, base) for base in sequence.upper())

def reverse_complement(sequence):
    """Return reverse complement of DNA sequence"""
    return complement_dna(sequence)[::-1]

def gc_content(sequence):
    """Calculate GC content percentage"""
    gc_count = sequence.upper().count('G') + sequence.upper().count('C')
    return (gc_count / len(sequence)) * 100

def translate_dna(sequence, codon_table):
    """Translate DNA sequence to amino acids"""
    # Implementation here
    pass
```

**Tasks**:
1. Complete the `translate_dna` function
2. Create a main script that imports and uses these functions
3. Add docstring tests for each function

---

## Section 4: File Handling

### Exercise 4.1: Reading and Writing Text Files
```python
# Sample data to work with
sample_data = """Sample_ID,Species,Colony_Count,Plate_Type
S001,E.coli,45,LB_agar
S002,B.subtilis,23,nutrient_agar
S003,S.aureus,67,blood_agar
S004,P.aeruginosa,34,MacConkey_agar
"""

# Save to file first
with open('bacterial_counts.txt', 'w') as f:
    f.write(sample_data)
```

**Tasks**:
1. Read the file and parse the data into a list of dictionaries
2. Calculate total colony counts for each plate type
3. Find the species with the highest average colony count
4. Write a summary report to a new file

### Exercise 4.2: CSV File Operations
```python
import csv

# Create sample CSV data
csv_data = [
    ['Time_hours', 'OD600', 'pH', 'Temperature'],
    [0, 0.05, 7.2, 37.0],
    [2, 0.08, 7.1, 37.2],
    [4, 0.15, 7.0, 36.8],
    [6, 0.31, 6.9, 37.1],
    [8, 0.62, 6.8, 37.0]
]

# Write CSV file
with open('growth_curve.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(csv_data)
```

**Tasks**:
1. Read the CSV file using the csv module
2. Calculate growth rates between time points
3. Create a new CSV with calculated growth rates
4. Handle missing or invalid data points

### Exercise 4.3: Error Handling and File Paths
```python
import os
from pathlib import Path

def safe_file_reader(filepath, file_type='text'):
    """
    Safely read different types of files with proper error handling
    
    Args:
        filepath: path to the file
        file_type: 'text', 'csv', or 'fasta'
    
    Returns:
        file contents or None if error occurs
    """
    try:
        # Your implementation with proper error handling
        pass
    except FileNotFoundError:
        print(f"File {filepath} not found")
        return None
    except PermissionError:
        print(f"Permission denied for file {filepath}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
```

**Tasks**:
1. Complete the function with appropriate file reading logic
2. Test with existing and non-existing files
3. Create a batch file processor that handles multiple files
4. Implement logging for file operations

---

## Challenge Exercises

### Challenge 1: Laboratory Inventory System
Create a simple inventory management system that can:
- Add new items (reagents, equipment)
- Track quantities and expiration dates
- Generate low-stock alerts
- Save/load inventory data from files

### Challenge 2: Growth Curve Analyzer
Build a comprehensive growth curve analysis tool that:
- Reads growth data from CSV files
- Calculates doubling times
- Identifies growth phases
- Generates summary statistics
- Exports results in multiple formats

### Challenge 3: DNA Sequence Toolkit
Develop a command-line tool that can:
- Read FASTA files
- Perform basic sequence analysis (GC content, length, etc.)
- Find open reading frames (ORFs)
- Translate sequences
- Generate reverse complements

---

## Evaluation Criteria

For each exercise, consider:
- **Correctness**: Does the code produce expected results?
- **Style**: Is the code readable and well-commented?
- **Error Handling**: Does it handle edge cases gracefully?
- **Efficiency**: Is the solution reasonably optimized?
- **Documentation**: Are functions properly documented?

## Tips for Success

1. **Start Simple**: Begin with basic implementations, then add features
2. **Test Frequently**: Use small datasets to verify your logic
3. **Comment Your Code**: Explain complex logic and assumptions
4. **Use Meaningful Names**: Variable and function names should be descriptive
5. **Handle Errors**: Always consider what could go wrong
6. **Practice Regularly**: Consistent practice builds proficiency

## Next Steps

After completing Phase 1 exercises, you'll be ready for:
- Phase 2: Scientific Python Ecosystem (NumPy, Pandas, Matplotlib)
- Working with real biological datasets
- More complex data analysis and visualization tasks

Good luck with your Python journey in microbiology! 🧬🐍
