# Functions and Modules

## Defining Functions

Functions are reusable blocks of code that perform specific tasks. In microbiology, they help standardize calculations and procedures.

### Basic Function Structure

```python
def calculate_dilution_factor(original_count, target_count):
    """Calculate the dilution factor needed to reach target concentration."""
    if target_count == 0:
        return None
    return original_count / target_count

# Usage
dilution = calculate_dilution_factor(1000000, 1000)
print(f"Dilution factor: 1:{dilution}")
```

### Functions with Default Parameters

```python
def growth_rate_calculation(initial_od, final_od, time_hours=24):
    """Calculate bacterial growth rate with default 24-hour period."""
    if initial_od <= 0 or time_hours <= 0:
        return 0
    
    growth_rate = (final_od - initial_od) / (initial_od * time_hours)
    return growth_rate

# Usage with default time
rate1 = growth_rate_calculation(0.1, 1.2)
# Usage with custom time
rate2 = growth_rate_calculation(0.1, 1.2, 18)
```

### Return Values

```python
def analyze_antibiotic_resistance(zone_diameters):
    """Analyze antibiotic sensitivity based on inhibition zones."""
    results = {}
    
    for antibiotic, diameter in zone_diameters.items():
        if diameter >= 20:
            sensitivity = "Sensitive"
        elif diameter >= 15:
            sensitivity = "Intermediate"
        else:
            sensitivity = "Resistant"
        
        results[antibiotic] = sensitivity
    
    return results

# Usage
zones = {"ampicillin": 18, "kanamycin": 22, "chloramphenicol": 12}
resistance_profile = analyze_antibiotic_resistance(zones)
```

## Scope and Variables

### Local vs Global Variables

```python
# Global variables - accessible throughout the program
OPTIMAL_TEMP = 37  # Standard incubation temperature
STANDARD_pH = 7.0

def check_culture_conditions(temp, ph):
    """Check if culture conditions are optimal."""
    # Local variables - only accessible within function
    temp_status = "Optimal" if abs(temp - OPTIMAL_TEMP) <= 2 else "Suboptimal"
    ph_status = "Optimal" if abs(ph - STANDARD_pH) <= 0.5 else "Suboptimal"
    
    return temp_status, ph_status

# Usage
temp_result, ph_result = check_culture_conditions(35, 7.2)
```

### Function Parameters and Arguments

```python
def prepare_growth_medium(base_medium, supplements=None, pH=7.0, volume_ml=1000):
    """Prepare bacterial growth medium with specified components."""
    if supplements is None:
        supplements = []
    
    medium_recipe = {
        "base": base_medium,
        "supplements": supplements,
        "pH": pH,
        "volume": volume_ml
    }
    
    print(f"Preparing {volume_ml}ml of {base_medium}")
    if supplements:
        print(f"Adding supplements: {', '.join(supplements)}")
    print(f"Adjusting pH to {pH}")
    
    return medium_recipe

# Different ways to call the function
recipe1 = prepare_growth_medium("LB broth")
recipe2 = prepare_growth_medium("minimal medium", ["glucose", "vitamins"])
recipe3 = prepare_growth_medium(base_medium="TSB", pH=6.8, volume_ml=500)
```

## Working with Modules

### Built-in Modules

```python
import math
import random
from datetime import datetime

def calculate_generation_time(initial_count, final_count, time_hours):
    """Calculate bacterial generation time using logarithms."""
    if initial_count <= 0 or final_count <= initial_count:
        return None
    
    generations = math.log2(final_count / initial_count)
    generation_time = time_hours / generations
    return generation_time

# Random sampling for quality control
def select_random_samples(sample_list, n=5):
    """Select random samples for quality control testing."""
    return random.sample(sample_list, min(n, len(sample_list)))

# Timestamp for experiment logging
def log_experiment_start(experiment_name):
    """Log the start time of an experiment."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Experiment '{experiment_name}' started at {timestamp}")
    return timestamp
```

### Creating Your Own Modules

Create a file named `microbio_utils.py`:

```python
# microbio_utils.py
"""Utility functions for microbiology calculations."""

def mcfarland_to_cfu(mcfarland_standard):
    """Convert McFarland standard to approximate CFU/ml."""
    conversion_factors = {
        0.5: 1.5e8,
        1.0: 3.0e8,
        2.0: 6.0e8,
        3.0: 9.0e8,
        4.0: 1.2e9
    }
    return conversion_factors.get(mcfarland_standard, None)

def calculate_viable_count(plate_count, dilution_factor, volume_plated=0.1):
    """Calculate viable count from plate count data."""
    return (plate_count * dilution_factor) / volume_plated

# Constants
GRAM_POSITIVE_STAINS = ["crystal violet", "safranin"]
GRAM_NEGATIVE_STAINS = ["safranin"]
```

### Using Your Custom Module

```python
# Import your custom module
import microbio_utils

# Or import specific functions
from microbio_utils import mcfarland_to_cfu, calculate_viable_count

# Usage
cfu_estimate = mcfarland_to_cfu(0.5)
viable_count = calculate_viable_count(156, 10000, 0.1)
```

## Best Practices

1. **Use descriptive function names** that explain what they do
2. **Write docstrings** to document function purpose and parameters
3. **Keep functions focused** on a single task
4. **Use meaningful parameter names**
5. **Handle edge cases** (zero values, empty lists, etc.)
6. **Return consistent data types**
