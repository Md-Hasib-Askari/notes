# NumPy Fundamentals

NumPy (Numerical Python) is essential for handling numerical data in microbiology research, especially for growth curves, population counts, and statistical analysis.

## Creating and Manipulating Arrays

### Basic Array Creation

```python
import numpy as np

# Create arrays from lists - common in microbiology
colony_counts = np.array([45, 52, 38, 61, 47, 55])
temperatures = np.array([25.0, 30.0, 37.0, 42.0, 50.0])
growth_rates = np.array([0.1, 0.3, 0.8, 0.6, 0.2])

# Create special arrays
time_points = np.arange(0, 25, 0.5)  # Every 30 minutes for 24 hours
zeros_array = np.zeros(10)  # Initialize empty measurements
ones_array = np.ones(5)     # Standard concentration references
```

### 2D Arrays for Experimental Data

```python
# Microplate data (8x12 wells)
od_readings = np.array([
    [0.12, 0.15, 0.18, 0.22, 0.25, 0.28, 0.31, 0.34, 0.37, 0.40, 0.43, 0.46],
    [0.11, 0.14, 0.17, 0.21, 0.24, 0.27, 0.30, 0.33, 0.36, 0.39, 0.42, 0.45],
    [0.13, 0.16, 0.19, 0.23, 0.26, 0.29, 0.32, 0.35, 0.38, 0.41, 0.44, 0.47],
    [0.10, 0.13, 0.16, 0.20, 0.23, 0.26, 0.29, 0.32, 0.35, 0.38, 0.41, 0.44]
])

print(f"Plate dimensions: {od_readings.shape}")  # (4, 12)
print(f"Total wells: {od_readings.size}")        # 48
```

## Mathematical Operations

### Element-wise Operations

```python
# Calculate growth over time
initial_od = np.array([0.05, 0.06, 0.04, 0.05])
final_od = np.array([1.2, 1.4, 0.9, 1.1])

# Growth fold increase
growth_fold = final_od / initial_od
print(f"Growth fold: {growth_fold}")

# Logarithmic transformation for growth rates
log_growth = np.log2(growth_fold)
print(f"Doublings: {log_growth}")

# Convert Celsius to Fahrenheit
celsius_temps = np.array([25, 30, 37, 42])
fahrenheit_temps = (celsius_temps * 9/5) + 32
```

### Mathematical Functions

```python
# Statistical analysis of colony counts
counts = np.array([156, 142, 167, 139, 148, 161, 144, 153])

# Basic statistics
mean_count = np.mean(counts)
std_count = np.std(counts)
median_count = np.median(counts)

print(f"Mean: {mean_count:.1f} ± {std_count:.1f}")
print(f"Median: {median_count}")
print(f"Range: {np.min(counts)} - {np.max(counts)}")

# Coefficient of variation (CV)
cv = (std_count / mean_count) * 100
print(f"CV: {cv:.1f}%")
```

## Array Indexing and Slicing

### Basic Indexing

```python
# Time-course growth data
time_hours = np.array([0, 2, 4, 6, 8, 10, 12, 24])
od_values = np.array([0.05, 0.12, 0.28, 0.56, 0.89, 1.24, 1.45, 1.38])

# Access specific time points
initial_od = od_values[0]
final_od = od_values[-1]
mid_exponential = od_values[3:6]  # Hours 6-10

print(f"Exponential phase ODs: {mid_exponential}")
```

### Boolean Indexing

```python
# Filter data based on conditions
high_growth = od_values[od_values > 1.0]
early_timepoints = time_hours[od_values < 0.5]

# Find contaminated wells (unusually high readings)
contamination_threshold = 2.0
contaminated_wells = od_readings > contamination_threshold
clean_data = od_readings[~contaminated_wells]

print(f"Number of contaminated wells: {np.sum(contaminated_wells)}")
```

### Advanced Indexing for Microplates

```python
# Analyze specific regions of microplate
# Select corners (quality control wells)
corners = od_readings[[0, 0, -1, -1], [0, -1, 0, -1]]

# Select border wells (edge effects)
border_wells = np.concatenate([
    od_readings[0, :],   # Top row
    od_readings[-1, :],  # Bottom row
    od_readings[:, 0],   # Left column
    od_readings[:, -1]   # Right column
])

print(f"Corner wells mean: {np.mean(corners):.3f}")
print(f"Border wells mean: {np.mean(border_wells):.3f}")
```

## Statistical Functions

### Descriptive Statistics

```python
# Analyze antibiotic inhibition zones
zones_mm = np.array([18, 22, 15, 24, 19, 21, 17, 23, 16, 20])

# Comprehensive statistics
stats = {
    'count': len(zones_mm),
    'mean': np.mean(zones_mm),
    'std': np.std(zones_mm, ddof=1),  # Sample standard deviation
    'min': np.min(zones_mm),
    'max': np.max(zones_mm),
    'median': np.median(zones_mm),
    'q25': np.percentile(zones_mm, 25),
    'q75': np.percentile(zones_mm, 75)
}

for key, value in stats.items():
    print(f"{key}: {value:.2f}")
```

### Random Number Generation

```python
# Simulate experimental variation
np.random.seed(42)  # For reproducible results

# Simulate colony counts with normal distribution
true_count = 150
measurement_error = 10
simulated_counts = np.random.normal(true_count, measurement_error, 20)

# Simulate contamination events (rare events)
contamination_probability = 0.05
contaminated = np.random.random(100) < contamination_probability
contamination_rate = np.sum(contaminated) / len(contaminated)

print(f"Simulated contamination rate: {contamination_rate:.1%}")

# Generate random sample selection
sample_ids = np.arange(1, 101)  # 100 samples
selected_samples = np.random.choice(sample_ids, size=20, replace=False)
print(f"Random samples for testing: {selected_samples[:5]}...")
```

## Array Operations for Growth Analysis

```python
# Calculate generation time from growth curve
def calculate_generation_time(time_hours, od_values):
    """Calculate bacterial generation time from exponential phase."""
    # Find exponential phase (log-linear region)
    log_od = np.log(od_values)
    
    # Linear regression on log(OD) vs time
    coeffs = np.polyfit(time_hours, log_od, 1)
    growth_rate = coeffs[0]  # slope
    
    # Generation time = ln(2) / growth_rate
    generation_time = np.log(2) / growth_rate
    return generation_time

# Example usage
time = np.array([0, 1, 2, 3, 4, 5])
od = np.array([0.05, 0.08, 0.13, 0.21, 0.34, 0.55])

gen_time = calculate_generation_time(time, od)
print(f"Generation time: {gen_time:.2f} hours")
```

## Best Practices

1. **Use NumPy arrays** instead of lists for numerical data
2. **Vectorize operations** instead of loops when possible
3. **Handle missing data** with `np.nan` and `np.nanmean()`
4. **Use appropriate data types** to save memory
5. **Document array dimensions** and what each axis represents
6. **Validate data ranges** before analysis
