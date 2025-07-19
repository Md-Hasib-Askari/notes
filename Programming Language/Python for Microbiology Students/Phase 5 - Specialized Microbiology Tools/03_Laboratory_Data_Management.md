# Laboratory Data Management

## Parsing Instrument Output Files
Parse common laboratory instrument output formats.

```python
import pandas as pd
import re
from pathlib import Path

def parse_plate_reader_data(file_path):
    """Parse plate reader output files"""
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Find data section
    data_start = None
    for i, line in enumerate(lines):
        if 'Well' in line and 'A1' in line:
            data_start = i
            break
    
    if data_start is None:
        raise ValueError("Could not find data section in file")
    
    # Parse well data
    well_data = {}
    for line in lines[data_start:]:
        if re.match(r'^[A-H]\d+', line.strip()):
            parts = line.strip().split('\t')
            well = parts[0]
            values = [float(x) for x in parts[1:] if x.strip()]
            well_data[well] = values
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(well_data, orient='index')
    df.index.name = 'Well'
    
    return df

def parse_qpcr_data(file_path):
    """Parse qPCR instrument output"""
    
    # Read CSV file
    df = pd.read_csv(file_path, skiprows=7)  # Skip header rows
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Extract relevant columns
    results = df[['Well', 'Sample Name', 'Target Name', 'CT', 'Quantity']].copy()
    results['CT'] = pd.to_numeric(results['CT'], errors='coerce')
    results['Quantity'] = pd.to_numeric(results['Quantity'], errors='coerce')
    
    return results

# Example usage
plate_data = parse_plate_reader_data('plate_reader_output.txt')
qpcr_results = parse_qpcr_data('qpcr_results.csv')
```

## Growth Curve Analysis and Modeling
Analyze bacterial growth curves and fit mathematical models.

```python
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def logistic_growth(t, K, r, t0):
    """Logistic growth model"""
    return K / (1 + np.exp(-r * (t - t0)))

def gompertz_growth(t, A, mu, lam):
    """Gompertz growth model"""
    return A * np.exp(-np.exp(mu * np.e / A * (lam - t) + 1))

def analyze_growth_curve(time_points, od_values):
    """Analyze growth curve and extract parameters"""
    
    # Remove any negative or zero values
    valid_data = [(t, od) for t, od in zip(time_points, od_values) if od > 0]
    time_clean, od_clean = zip(*valid_data)
    
    time_array = np.array(time_clean)
    od_array = np.array(od_clean)
    
    # Fit logistic model
    try:
        # Initial parameter estimates
        K_init = max(od_array)  # Carrying capacity
        r_init = 0.1  # Growth rate
        t0_init = time_array[len(time_array)//2]  # Inflection point
        
        popt_logistic, _ = curve_fit(
            logistic_growth, time_array, od_array,
            p0=[K_init, r_init, t0_init],
            maxfev=2000
        )
        
        # Calculate R-squared
        y_pred = logistic_growth(time_array, *popt_logistic)
        ss_res = np.sum((od_array - y_pred) ** 2)
        ss_tot = np.sum((od_array - np.mean(od_array)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        results = {
            'model': 'logistic',
            'carrying_capacity': popt_logistic[0],
            'growth_rate': popt_logistic[1],
            'lag_time': popt_logistic[2],
            'r_squared': r_squared,
            'parameters': popt_logistic
        }
        
    except Exception as e:
        results = {'error': str(e)}
    
    return results

def plot_growth_curve(time_points, od_values, model_params=None):
    """Plot growth curve with optional model fit"""
    
    plt.figure(figsize=(10, 6))
    plt.scatter(time_points, od_values, alpha=0.7, label='Data')
    
    if model_params and 'parameters' in model_params:
        time_smooth = np.linspace(min(time_points), max(time_points), 100)
        od_smooth = logistic_growth(time_smooth, *model_params['parameters'])
        plt.plot(time_smooth, od_smooth, 'r-', 
                label=f"Logistic fit (R² = {model_params['r_squared']:.3f})")
    
    plt.xlabel('Time (hours)')
    plt.ylabel('Optical Density (600nm)')
    plt.title('Bacterial Growth Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return plt.gcf()

# Example usage
time = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
od = np.array([0.05, 0.08, 0.15, 0.35, 0.65, 0.95, 1.25, 1.45, 1.52, 1.55, 1.56])

growth_params = analyze_growth_curve(time, od)
plot_growth_curve(time, od, growth_params)
```

## Plate Reader Data Processing
Process multi-well plate reader data for high-throughput analysis.

```python
def process_plate_layout(layout_file, data_df):
    """Process plate layout and merge with data"""
    
    # Load plate layout
    layout = pd.read_csv(layout_file)
    
    # Merge layout with data
    merged_data = data_df.merge(layout, on='Well', how='left')
    
    return merged_data

def calculate_statistics(plate_data, group_by=['Strain', 'Condition']):
    """Calculate statistics for grouped data"""
    
    stats = plate_data.groupby(group_by).agg({
        'OD600': ['mean', 'std', 'count'],
        'Growth_Rate': ['mean', 'std']
    }).round(4)
    
    return stats

def quality_control_check(plate_data, blank_wells=['H12']):
    """Perform quality control checks"""
    
    qc_results = {}
    
    # Check blank wells
    blank_values = plate_data[plate_data['Well'].isin(blank_wells)]['OD600']
    qc_results['blank_mean'] = blank_values.mean()
    qc_results['blank_std'] = blank_values.std()
    
    # Check for outliers (values > 3 std from mean)
    z_scores = np.abs((plate_data['OD600'] - plate_data['OD600'].mean()) / plate_data['OD600'].std())
    qc_results['outliers'] = plate_data[z_scores > 3]['Well'].tolist()
    
    # Check coefficient of variation for replicates
    cv_data = plate_data.groupby(['Strain', 'Condition'])['OD600'].agg(['mean', 'std'])
    cv_data['CV'] = cv_data['std'] / cv_data['mean'] * 100
    qc_results['high_cv'] = cv_data[cv_data['CV'] > 15].index.tolist()
    
    return qc_results

# Example workflow
layout_file = 'plate_layout.csv'
processed_data = process_plate_layout(layout_file, plate_data)
statistics = calculate_statistics(processed_data)
qc_results = quality_control_check(processed_data)
```

These tools streamline laboratory data processing, from instrument output parsing to statistical analysis and quality control.
