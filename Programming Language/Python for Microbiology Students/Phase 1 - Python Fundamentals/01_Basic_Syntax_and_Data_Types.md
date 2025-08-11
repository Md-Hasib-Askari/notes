# Basic Syntax and Data Types

## Variables and Basic Operations

```python
# Variables - containers for storing data
sample_name = "E_coli_strain_K12"  # String variable
colony_count = 42                   # Integer variable
growth_rate = 0.75                 # Float variable
is_pathogenic = False              # Boolean variable
```

## Data Types

### Numbers
```python
# Integers and floats for measurements
petri_dishes = 20
temperature = 37.5
ph_level = 7.2

# Mathematical operations
total_colonies = colony_count * petri_dishes
doubling_time = 60 / growth_rate  # minutes
```

### Strings
```python
# Text data for labels and descriptions
organism = "Escherichia coli"
strain_id = "ATCC 25922"
medium = "LB broth"

# String concatenation
sample_label = organism + " - " + strain_id
print(f"Growing {organism} in {medium} at {temperature}°C")
```

### Lists
```python
# Ordered collections - perfect for experimental data
temperatures = [25, 30, 37, 42, 50]
antibiotics = ["ampicillin", "kanamycin", "chloramphenicol"]
od_readings = [0.1, 0.3, 0.8, 1.2, 1.5]

# Accessing elements
optimal_temp = temperatures[2]  # Index 2 = 37°C
first_antibiotic = antibiotics[0]
```

### Tuples
```python
# Immutable sequences - ideal for coordinates or fixed data
coordinates = (12.5, 8.3)  # Position on microscope slide
sample_info = ("E_coli", "K12", 2023)  # Organism, strain, year
```

### Dictionaries
```python
# Key-value pairs - excellent for sample metadata
sample_data = {
    "organism": "Bacillus subtilis",
    "temperature": 30,
    "pH": 7.0,
    "growth_medium": "nutrient_agar",
    "incubation_time": 24
}

# Accessing values
temp = sample_data["temperature"]
sample_data["colony_count"] = 156  # Adding new data
```

### Sets
```python
# Unique collections - useful for removing duplicates
unique_organisms = {"E_coli", "B_subtilis", "S_aureus", "E_coli"}
print(unique_organisms)  # {'E_coli', 'B_subtilis', 'S_aureus'}

# Set operations
gram_positive = {"B_subtilis", "S_aureus", "S_pneumoniae"}
tested_organisms = {"E_coli", "B_subtilis", "P_aeruginosa"}
overlap = gram_positive.intersection(tested_organisms)
```

## Input/Output Operations

```python
# Getting user input for experiments
sample_name = input("Enter sample name: ")
dilution_factor = float(input("Enter dilution factor: "))

# Displaying results
print(f"Sample: {sample_name}")
print(f"Adjusted count: {colony_count * dilution_factor}")
```

## Best Practices for Microbiology

1. Use descriptive variable names (`colony_count` not `c`)
2. Include units in variable names when relevant (`temp_celsius`)
3. Use constants for standard values (`BODY_TEMP = 37`)
4. Comment your code to explain biological significance
