# Jupyter Notebooks

Jupyter Notebooks are an essential tool for interactive data analysis, experiment documentation, and reproducible research in microbiology.

## Setting Up and Using Jupyter

### Installation and Basic Setup

```bash
# Install Jupyter via pip
pip install jupyter

# Install JupyterLab (modern interface)
pip install jupyterlab

# Start Jupyter Notebook
jupyter notebook

# Start JupyterLab
jupyter lab
```

### Essential Extensions for Scientists

```bash
# Install useful extensions
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user

# Popular extensions for scientific work:
# - Table of Contents
# - Variable Inspector
# - Code Folding
# - Collapsible Headings
```

## Notebook Structure for Laboratory Work

### Standard Laboratory Notebook Template

```markdown
# Experiment Title: Bacterial Growth Analysis
**Date:** 2023-10-15  
**Researcher:** [Your Name]  
**Objective:** Analyze growth patterns of E. coli under different temperature conditions

## Materials and Methods
- **Organism:** E. coli strain K12
- **Medium:** LB broth
- **Temperature conditions:** 25°C, 30°C, 37°C, 42°C
- **Measurement method:** OD₆₀₀ spectrophotometry
- **Time points:** Every 2 hours for 24 hours

## Data Collection
```

```python
# Import libraries at the top
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set display options
pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8-whitegrid')

# Record experiment metadata
experiment_info = {
    'date': '2023-10-15',
    'researcher': 'Lab Team A',
    'organism': 'E. coli K12',
    'protocol_version': '2.1'
}

print(f"Experiment started: {experiment_info['date']}")
print(f"Organism: {experiment_info['organism']}")
```

### Data Import and Initial Exploration

```python
# Load experimental data
growth_data = pd.read_csv('growth_curve_data.csv')

# Display basic information
print("Dataset shape:", growth_data.shape)
print("\nColumn names:", growth_data.columns.tolist())
print("\nFirst few rows:")
display(growth_data.head())

# Check for missing values
print("\nMissing values:")
print(growth_data.isnull().sum())

# Basic statistics
print("\nBasic statistics:")
display(growth_data.describe())
```

## Markdown Cells for Documentation

### Experiment Documentation Best Practices

```markdown
## Results Summary

### Key Findings:
1. **Optimal growth temperature:** 37°C showed highest growth rate (0.89 h⁻¹)
2. **Temperature sensitivity:** Growth significantly reduced below 30°C and above 42°C
3. **Generation time:** Shortest at 37°C (47 minutes)

### Statistical Analysis:
- **ANOVA results:** F-statistic = 15.67, p-value < 0.001
- **Post-hoc testing:** Tukey HSD revealed significant differences between all temperature groups

### Quality Control:
- All negative controls remained sterile
- Positive controls showed expected growth patterns
- CV between replicates: 8.3% (acceptable, <15%)
```

### Mathematical Formulas with LaTeX

```markdown
## Growth Rate Calculations

The specific growth rate (μ) is calculated using:

$$\mu = \frac{\ln(N_t) - \ln(N_0)}{t}$$

Where:
- $N_t$ = cell concentration at time t
- $N_0$ = initial cell concentration  
- $t$ = time interval

Generation time is calculated as:

$$t_g = \frac{\ln(2)}{\mu}$$

For our E. coli cultures at 37°C:
- μ = 0.89 h⁻¹
- $t_g$ = 0.78 hours (47 minutes)
```

## Interactive Data Exploration

### Dynamic Data Analysis

```python
# Interactive widgets for parameter exploration
from ipywidgets import interact, FloatSlider, Dropdown

def plot_growth_curve(temperature=37.0, organism='E_coli'):
    """Interactive plot function"""
    # Filter data based on parameters
    filtered_data = growth_data[
        (growth_data['Temperature'] == temperature) & 
        (growth_data['Organism'] == organism)
    ]
    
    plt.figure(figsize=(10, 6))
    plt.plot(filtered_data['Time_hours'], filtered_data['OD600'], 'o-')
    plt.xlabel('Time (hours)')
    plt.ylabel('OD₆₀₀')
    plt.title(f'Growth Curve: {organism} at {temperature}°C')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Display statistics
    max_od = filtered_data['OD600'].max()
    growth_rate = np.polyfit(filtered_data['Time_hours'][:5], 
                           np.log(filtered_data['OD600'][:5]), 1)[0]
    print(f"Maximum OD₆₀₀: {max_od:.3f}")
    print(f"Growth rate: {growth_rate:.3f} h⁻¹")

# Create interactive widget
interact(plot_growth_curve,
         temperature=FloatSlider(min=25, max=50, step=1, value=37),
         organism=Dropdown(options=['E_coli', 'S_aureus', 'B_subtilis']))
```

### Real-time Data Processing

```python
# Function to process new experimental data
def process_new_timepoint(time_point, od_reading, temperature):
    """Process and visualize new data point"""
    global growth_data
    
    # Add new data point
    new_row = pd.DataFrame({
        'Time_hours': [time_point],
        'OD600': [od_reading],
        'Temperature': [temperature],
        'Timestamp': [datetime.now()]
    })
    
    growth_data = pd.concat([growth_data, new_row], ignore_index=True)
    
    # Update visualization
    plt.figure(figsize=(8, 5))
    plt.plot(growth_data['Time_hours'], growth_data['OD600'], 'o-')
    plt.xlabel('Time (hours)')
    plt.ylabel('OD₆₀₀')
    plt.title('Real-time Growth Monitoring')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"Data point added: t={time_point}h, OD={od_reading}")
    
    # Auto-save updated data
    growth_data.to_csv('growth_data_updated.csv', index=False)
    print("Data automatically saved")

# Example usage during experiment
# process_new_timepoint(8, 0.756, 37)
```

## Advanced Notebook Features

### Version Control Integration

```python
# Track analysis versions
analysis_version = {
    'notebook_version': '1.3',
    'last_modified': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'changes': 'Added statistical analysis section',
    'validated_by': 'Senior Researcher'
}

print("Analysis Version Control:")
for key, value in analysis_version.items():
    print(f"{key}: {value}")
```

### Automated Report Generation

```python
# Generate automated summary report
def generate_experiment_report():
    """Generate comprehensive experiment report"""
    
    report = f"""
    # Experimental Report
    
    **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    ## Experiment Overview
    - Organism: {experiment_info['organism']}
    - Date: {experiment_info['date']}
    - Total samples analyzed: {len(growth_data)}
    
    ## Key Results
    - Temperature range tested: {growth_data['Temperature'].min()}-{growth_data['Temperature'].max()}°C
    - Maximum OD achieved: {growth_data['OD600'].max():.3f}
    - Growth rate range: {growth_data.groupby('Temperature')['OD600'].apply(lambda x: np.polyfit(range(len(x)), np.log(x.fillna(0.01)), 1)[0]).min():.3f} to {growth_data.groupby('Temperature')['OD600'].apply(lambda x: np.polyfit(range(len(x)), np.log(x.fillna(0.01)), 1)[0]).max():.3f} h⁻¹
    
    ## Data Quality
    - Missing data points: {growth_data.isnull().sum().sum()}
    - Coefficient of variation: {(growth_data['OD600'].std() / growth_data['OD600'].mean() * 100):.1f}%
    
    ## Conclusions
    - Optimal growth conditions identified
    - Statistical significance confirmed
    - Results ready for publication
    """
    
    return report

# Generate and display report
report = generate_experiment_report()
print(report)

# Save report to file
with open('experiment_report.md', 'w') as f:
    f.write(report)
```

### Integration with Laboratory Information Systems

```python
# Connect to laboratory database
import sqlite3

def save_to_lab_database(experiment_data):
    """Save experiment results to laboratory database"""
    
    conn = sqlite3.connect('lab_database.db')
    cursor = conn.cursor()
    
    # Create table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY,
            date TEXT,
            organism TEXT,
            temperature REAL,
            max_od REAL,
            growth_rate REAL,
            researcher TEXT
        )
    ''')
    
    # Insert experiment summary
    for temp in experiment_data['Temperature'].unique():
        temp_data = experiment_data[experiment_data['Temperature'] == temp]
        max_od = temp_data['OD600'].max()
        growth_rate = np.polyfit(range(len(temp_data)), 
                               np.log(temp_data['OD600'].fillna(0.01)), 1)[0]
        
        cursor.execute('''
            INSERT INTO experiments 
            (date, organism, temperature, max_od, growth_rate, researcher)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (experiment_info['date'], experiment_info['organism'], 
              temp, max_od, growth_rate, experiment_info['researcher']))
    
    conn.commit()
    conn.close()
    print("Data saved to laboratory database")

# Save current experiment
save_to_lab_database(growth_data)
```

## Best Practices for Laboratory Notebooks

1. **Document everything** - assumptions, protocols, observations
2. **Use consistent naming** for variables and files
3. **Include metadata** - dates, versions, researchers
4. **Write self-explanatory code** with comments
5. **Validate results** with statistical tests
6. **Version control** your notebooks
7. **Export important results** in multiple formats
8. **Share notebooks** with collaborators for peer review
9. **Archive completed experiments** with all supporting data
