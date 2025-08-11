# File Handling

File handling is crucial in microbiology for managing experimental data, sample records, and analysis results.

## Reading and Writing Text Files

### Writing Data to Files

```python
# Recording experimental observations
experiment_log = """
Date: 2023-10-15
Organism: E. coli strain K12
Medium: LB broth
Temperature: 37°C
Initial OD600: 0.05
Final OD600: 1.2
Growth time: 6 hours
Notes: Normal growth pattern observed
"""

# Write to file
with open("experiment_log.txt", "w") as file:
    file.write(experiment_log)

# Append additional data
with open("experiment_log.txt", "a") as file:
    file.write("\nContamination check: Negative\n")
```

### Reading Data from Files

```python
# Read entire file
with open("experiment_log.txt", "r") as file:
    content = file.read()
    print(content)

# Read line by line
with open("experiment_log.txt", "r") as file:
    for line_number, line in enumerate(file, 1):
        print(f"Line {line_number}: {line.strip()}")

# Read specific lines
with open("experiment_log.txt", "r") as file:
    lines = file.readlines()
    organism_line = lines[2]  # Third line contains organism info
```

## Working with CSV Files

CSV files are ideal for storing tabular data like growth curves and sample metadata.

### Writing CSV Data

```python
import csv

# Sample growth data
growth_data = [
    ["Time_hours", "OD600", "Temperature_C", "pH"],
    [0, 0.05, 37.0, 7.0],
    [2, 0.12, 37.2, 6.9],
    [4, 0.35, 37.1, 6.8],
    [6, 0.78, 37.0, 6.7],
    [8, 1.24, 37.1, 6.6]
]

# Write to CSV file
with open("growth_curve.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(growth_data)

# Write dictionary data
sample_info = [
    {"Sample_ID": "S001", "Organism": "E_coli", "Antibiotic": "Ampicillin", "Zone_mm": 18},
    {"Sample_ID": "S002", "Organism": "E_coli", "Antibiotic": "Kanamycin", "Zone_mm": 22},
    {"Sample_ID": "S003", "Organism": "S_aureus", "Antibiotic": "Ampicillin", "Zone_mm": 12}
]

with open("antibiotic_results.csv", "w", newline="") as file:
    fieldnames = ["Sample_ID", "Organism", "Antibiotic", "Zone_mm"]
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(sample_info)
```

### Reading CSV Data

```python
# Read CSV with csv.reader
with open("growth_curve.csv", "r") as file:
    reader = csv.reader(file)
    header = next(reader)  # Skip header row
    
    for row in reader:
        time, od, temp, ph = row
        print(f"Time: {time}h, OD600: {od}, Temp: {temp}°C")

# Read CSV with DictReader (more convenient)
with open("antibiotic_results.csv", "r") as file:
    reader = csv.DictReader(file)
    
    for row in reader:
        sample_id = row["Sample_ID"]
        organism = row["Organism"]
        zone_size = float(row["Zone_mm"])
        
        if zone_size < 15:
            sensitivity = "Resistant"
        elif zone_size < 20:
            sensitivity = "Intermediate"
        else:
            sensitivity = "Sensitive"
        
        print(f"{sample_id} ({organism}): {sensitivity}")
```

## File Paths and Directory Operations

```python
import os
from pathlib import Path

# Working with file paths
data_dir = Path("experimental_data")
results_dir = Path("results")

# Create directories if they don't exist
data_dir.mkdir(exist_ok=True)
results_dir.mkdir(exist_ok=True)

# Construct file paths
growth_file = data_dir / "growth_curve_20231015.csv"
analysis_file = results_dir / "growth_analysis.txt"

# Check if files exist
if growth_file.exists():
    print(f"Found growth data: {growth_file}")

# List files in directory
for file_path in data_dir.glob("*.csv"):
    print(f"CSV file found: {file_path.name}")

# Get file information
if growth_file.exists():
    file_size = growth_file.stat().st_size
    print(f"File size: {file_size} bytes")
```

## Error Handling with Try/Except

```python
def safe_file_read(filename):
    """Safely read a file with error handling."""
    try:
        with open(filename, "r") as file:
            content = file.read()
            return content
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        return None
    except PermissionError:
        print(f"Error: Permission denied to read '{filename}'")
        return None
    except Exception as e:
        print(f"Unexpected error reading '{filename}': {e}")
        return None

def process_experimental_data(filename):
    """Process experimental data with comprehensive error handling."""
    try:
        with open(filename, "r") as file:
            reader = csv.DictReader(file)
            
            valid_data = []
            for row_num, row in enumerate(reader, 2):  # Start at 2 (after header)
                try:
                    # Validate and convert data
                    od_value = float(row["OD600"])
                    temperature = float(row["Temperature_C"])
                    
                    if od_value < 0:
                        print(f"Warning: Negative OD value in row {row_num}")
                        continue
                    
                    if not 20 <= temperature <= 60:
                        print(f"Warning: Unusual temperature in row {row_num}")
                    
                    valid_data.append({
                        "OD600": od_value,
                        "Temperature": temperature
                    })
                    
                except ValueError as e:
                    print(f"Error in row {row_num}: Invalid number format")
                except KeyError as e:
                    print(f"Error in row {row_num}: Missing column {e}")
            
            return valid_data
            
    except Exception as e:
        print(f"Error processing file: {e}")
        return []

# Usage
data = process_experimental_data("growth_curve.csv")
print(f"Successfully processed {len(data)} data points")
```

## Best Practices for Laboratory Data

1. **Always use context managers** (`with` statements) for file operations
2. **Include timestamps** in filenames for version control
3. **Validate data** as you read it
4. **Handle missing or corrupted data** gracefully
5. **Use descriptive headers** in CSV files
6. **Back up important data files** regularly
7. **Document file formats** and column meanings
