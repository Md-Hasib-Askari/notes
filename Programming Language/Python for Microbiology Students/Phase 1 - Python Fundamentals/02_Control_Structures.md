# Control Structures

## Conditional Statements

Control structures help you make decisions in your microbiology programs based on experimental conditions.

### If/Elif/Else Statements

```python
# Temperature control for bacterial growth
temperature = 37

if temperature < 4:
    growth_status = "No growth - too cold"
elif temperature < 15:
    growth_status = "Slow growth - psychrophilic conditions"
elif temperature <= 45:
    growth_status = "Optimal growth - mesophilic range"
elif temperature <= 60:
    growth_status = "Thermophilic growth"
else:
    growth_status = "No growth - too hot"

print(f"At {temperature}°C: {growth_status}")
```

### pH Testing for Culture Conditions

```python
ph_level = 6.8

if ph_level < 6.0:
    culture_condition = "Acidic - check for contamination"
    action_needed = "Adjust pH with NaOH"
elif ph_level > 8.0:
    culture_condition = "Alkaline - unusual for most bacteria"
    action_needed = "Adjust pH with HCl"
else:
    culture_condition = "Optimal pH range"
    action_needed = "Continue incubation"

print(f"pH {ph_level}: {culture_condition}")
print(f"Action: {action_needed}")
```

## Loops

### For Loops - Processing Multiple Samples

```python
# Analyzing multiple bacterial samples
samples = ["E_coli_1", "E_coli_2", "B_subtilis_1", "S_aureus_1"]
od_readings = [0.8, 1.2, 0.6, 0.9]

for i, sample in enumerate(samples):
    od_value = od_readings[i]
    if od_value > 1.0:
        density = "High density"
    elif od_value > 0.5:
        density = "Medium density"
    else:
        density = "Low density"
    
    print(f"{sample}: OD={od_value} ({density})")
```

### While Loops - Growth Monitoring

```python
# Simulating bacterial growth monitoring
time_hours = 0
cell_count = 1000
doubling_time = 1.5

print("Time (h)\tCell Count")
while time_hours < 12 and cell_count < 1000000:
    print(f"{time_hours}\t\t{cell_count:,}")
    cell_count *= 2  # Cells double each generation
    time_hours += doubling_time
```

## Control Flow Statements

### Break Statement

```python
# Finding first contaminated sample
contamination_levels = [0.1, 0.05, 2.5, 0.3, 0.8]
threshold = 2.0

for i, level in enumerate(contamination_levels):
    if level > threshold:
        print(f"Contamination detected in sample {i+1}: {level}")
        break
    print(f"Sample {i+1} is clean: {level}")
```

### Continue Statement

```python
# Processing only valid readings
od_readings = [0.8, -1, 1.2, 0.0, 0.6, 2.5]

for reading in od_readings:
    if reading < 0 or reading > 2.0:  # Invalid readings
        continue
    
    # Process valid readings only
    if reading > 1.0:
        print(f"High density culture: OD {reading}")
    else:
        print(f"Normal density culture: OD {reading}")
```

## List Comprehensions

Elegant way to create lists based on existing data:

```python
# Convert Celsius to Fahrenheit for all temperatures
celsius_temps = [25, 30, 37, 42, 50]
fahrenheit_temps = [(temp * 9/5) + 32 for temp in celsius_temps]

# Filter samples above threshold
colony_counts = [45, 120, 8, 200, 156, 12]
high_density = [count for count in colony_counts if count > 100]

# Process multiple conditions
samples = [("E_coli", 37), ("Psychrobacter", 4), ("Thermus", 70)]
mesophiles = [name for name, temp in samples if 15 <= temp <= 45]
```

## Practical Applications

Use control structures for:
- Quality control checks
- Automated data validation
- Growth condition optimization
- Sample processing workflows
- Real-time monitoring systems
