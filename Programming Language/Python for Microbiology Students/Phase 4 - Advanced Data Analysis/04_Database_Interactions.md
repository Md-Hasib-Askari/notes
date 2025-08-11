# Database Interactions

## Connecting to Biological Databases
Access NCBI and other biological databases using Python libraries.

```python
from Bio import Entrez

# Set email (required for NCBI)
Entrez.email = "your.email@university.edu"

# Search for E. coli genomes
search_handle = Entrez.esearch(db="nucleotide", 
                              term="Escherichia coli[Organism]", retmax=3)
search_results = Entrez.read(search_handle)
search_handle.close()

# Fetch sequence data
fetch_handle = Entrez.efetch(db="nucleotide", 
                           id=search_results["IdList"][0], 
                           rettype="fasta", retmode="text")
sequence = fetch_handle.read()
fetch_handle.close()
```

## SQL Basics for Data Queries
Query structured biological data using SQL with SQLite.

```python
import sqlite3
import pandas as pd

# Connect to database
conn = sqlite3.connect('microbiology_lab.db')

# Create table and insert data
conn.execute("""CREATE TABLE experiments 
                (strain TEXT, temperature REAL, growth_rate REAL)""")

data = [('E.coli', 37.0, 0.45), ('B.subtilis', 30.0, 0.32)]
conn.executemany("INSERT INTO experiments VALUES (?, ?, ?)", data)

# Query with conditions
results = pd.read_sql_query(
    "SELECT strain, growth_rate FROM experiments WHERE temperature > 35", conn)

conn.close()
```

## Working with APIs for Data Retrieval
Fetch biological data from web APIs with error handling.

```python
import requests
import time

def fetch_uniprot_protein(protein_id):
    url = f"https://rest.uniprot.org/uniprotkb/{protein_id}.json"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        return {
            'name': data.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', 'Unknown'),
            'organism': data.get('organism', {}).get('scientificName', 'Unknown')
        }
    except requests.RequestException as e:
        print(f"Error: {e}")
        return None

# Fetch with rate limiting
for protein_id in ['P0A7G6', 'P0A6F5']:
    info = fetch_uniprot_protein(protein_id)
    if info:
        print(f"Protein: {info['name']}")
    time.sleep(1)  # Wait between requests
```

## Data Pipeline Creation
Automate data processing workflows for multiple files.

```python
import pandas as pd
from pathlib import Path
import numpy as np

def process_experiment_data(data_dir, output_file):
    all_data = []
    
    # Read all CSV files in directory
    for csv_file in Path(data_dir).glob("*.csv"):
        df = pd.read_csv(csv_file)
        df['source_file'] = csv_file.name
        all_data.append(df)
    
    # Combine datasets
    combined = pd.concat(all_data, ignore_index=True)
    
    # Clean and process
    combined['od600'] = pd.to_numeric(combined['od600'], errors='coerce')
    combined = combined.dropna(subset=['od600'])
    combined['log_od'] = np.log(combined['od600'])
    
    # Save processed data
    combined.to_csv(output_file, index=False)
    return combined

# Execute pipeline
processed = process_experiment_data("./experiments", "processed_data.csv")
```

These database interaction techniques enable automated data retrieval and processing for microbiology research workflows.
