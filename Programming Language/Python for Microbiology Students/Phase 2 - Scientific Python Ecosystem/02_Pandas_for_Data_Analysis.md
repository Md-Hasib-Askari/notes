# Pandas for Data Analysis

Pandas is the cornerstone library for data manipulation in Python, perfect for managing laboratory datasets, experimental results, and sample metadata.

## DataFrames and Series

### Creating DataFrames

```python
import pandas as pd
import numpy as np

# Create DataFrame from dictionary - ideal for experimental data
sample_data = {
    'Sample_ID': ['S001', 'S002', 'S003', 'S004', 'S005'],
    'Organism': ['E_coli', 'E_coli', 'S_aureus', 'B_subtilis', 'E_coli'],
    'Temperature': [37, 37, 37, 30, 42],
    'pH': [7.0, 7.2, 6.8, 7.0, 7.1],
    'Growth_Rate': [0.8, 0.9, 0.6, 0.4, 0.7],
    'Final_OD': [1.2, 1.4, 0.9, 0.6, 1.1]
}

df = pd.DataFrame(sample_data)
print(df.head())
print(f"\nDataFrame shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
```

### Working with Series

```python
# Extract specific columns as Series
temperatures = df['Temperature']
growth_rates = df['Growth_Rate']

# Basic Series operations
print(f"Mean temperature: {temperatures.mean():.1f}°C")
print(f"Max growth rate: {growth_rates.max():.2f}")

# Boolean indexing with Series
high_growth = df[df['Growth_Rate'] > 0.7]
print(f"\nHigh growth samples:\n{high_growth}")
```

## Reading Data from Various Formats

### CSV Files - Most Common in Lab Settings

```python
# Read experimental data
growth_data = pd.read_csv('growth_curve_data.csv')

# Common parameters for lab data
antibiotic_data = pd.read_csv(
    'antibiotic_sensitivity.csv',
    parse_dates=['Date'],          # Convert date columns
    index_col='Sample_ID',         # Set sample ID as index
    na_values=['N/A', 'contaminated']  # Handle missing values
)

# Read with custom column names
od_readings = pd.read_csv(
    'plate_reader_output.csv',
    skiprows=3,  # Skip header rows from instrument
    names=['Well', 'OD600', 'Temperature', 'Time']
)
```

### Excel Files - Common for Lab Records

```python
# Read multiple sheets from Excel file
lab_notebook = pd.read_excel('experiment_log.xlsx', sheet_name=None)

# Read specific sheet with experimental conditions
conditions = pd.read_excel(
    'experiment_log.xlsx',
    sheet_name='Conditions',
    usecols=['Sample_ID', 'Medium', 'Temperature', 'pH']
)

# Read growth data with time index
growth_curves = pd.read_excel(
    'growth_data.xlsx',
    sheet_name='TimeCourse',
    index_col='Time_hours'
)
```

### JSON Files - API Data and Metadata

```python
# Read sample metadata from JSON
metadata = pd.read_json('sample_metadata.json')

# For nested JSON structures
import json
with open('experiment_config.json') as f:
    config_data = json.load(f)
    
config_df = pd.json_normalize(config_data['samples'])
```

## Data Cleaning and Preprocessing

### Handling Missing Values

```python
# Identify missing data
print("Missing values per column:")
print(df.isnull().sum())

# Visualize missing data pattern
print("\nData completeness:")
print(df.info())

# Handle missing values
# Drop rows with any missing values
clean_df = df.dropna()

# Fill missing values with defaults
df_filled = df.fillna({
    'pH': 7.0,  # Neutral pH as default
    'Temperature': 37,  # Standard incubation temperature
    'Growth_Rate': df['Growth_Rate'].median()
})

# Forward fill for time series data
time_series_df = growth_curves.fillna(method='ffill')
```

### Data Type Conversion

```python
# Convert data types for analysis
df['Sample_ID'] = df['Sample_ID'].astype('category')
df['Temperature'] = pd.to_numeric(df['Temperature'], errors='coerce')
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

# Create categorical data for statistical analysis
df['Temperature_Category'] = pd.cut(
    df['Temperature'],
    bins=[0, 25, 35, 45, 100],
    labels=['Low', 'Medium', 'Optimal', 'High']
)
```

## Filtering, Grouping, and Aggregating Data

### Data Filtering

```python
# Filter by single condition
e_coli_samples = df[df['Organism'] == 'E_coli']

# Multiple conditions
optimal_conditions = df[
    (df['Temperature'] == 37) & 
    (df['pH'] >= 6.8) & 
    (df['pH'] <= 7.2)
]

# Filter by lists of values
target_organisms = ['E_coli', 'S_aureus']
selected_organisms = df[df['Organism'].isin(target_organisms)]

# String filtering
gram_positive = df[df['Organism'].str.contains('aureus|subtilis')]
```

### Grouping and Aggregation

```python
# Group by organism and calculate statistics
organism_stats = df.groupby('Organism').agg({
    'Growth_Rate': ['mean', 'std', 'count'],
    'Final_OD': ['mean', 'max', 'min'],
    'Temperature': 'mean'
})

print("Growth statistics by organism:")
print(organism_stats)

# Multiple grouping variables
temp_organism_stats = df.groupby(['Temperature', 'Organism'])['Growth_Rate'].describe()

# Custom aggregation functions
def coefficient_of_variation(series):
    return (series.std() / series.mean()) * 100

cv_stats = df.groupby('Organism')['Growth_Rate'].agg([
    'mean',
    'std',
    coefficient_of_variation
])
```

### Pivot Tables for Experimental Design

```python
# Create pivot table for experimental results
results_pivot = df.pivot_table(
    values='Growth_Rate',
    index='Organism',
    columns='Temperature',
    aggfunc=['mean', 'std'],
    fill_value=0
)

print("Growth rates by organism and temperature:")
print(results_pivot)

# Crosstab for frequency analysis
contamination_crosstab = pd.crosstab(
    df['Organism'],
    df['Temperature_Category'],
    margins=True
)
```

## Advanced Data Manipulation

### Merging Datasets

```python
# Merge experimental data with metadata
sample_metadata = pd.DataFrame({
    'Sample_ID': ['S001', 'S002', 'S003', 'S004', 'S005'],
    'Collection_Date': ['2023-10-01', '2023-10-01', '2023-10-02', '2023-10-02', '2023-10-03'],
    'Source': ['Patient_A', 'Patient_A', 'Patient_B', 'Environment', 'Patient_C'],
    'Antibiotic_Treatment': ['Yes', 'Yes', 'No', 'No', 'Yes']
})

# Merge datasets
combined_data = pd.merge(df, sample_metadata, on='Sample_ID', how='left')

# Join multiple datasets
resistance_data = pd.DataFrame({
    'Sample_ID': ['S001', 'S002', 'S003'],
    'Ampicillin': ['R', 'S', 'I'],
    'Kanamycin': ['S', 'S', 'R']
})

full_dataset = combined_data.merge(resistance_data, on='Sample_ID', how='left')
```

### Time Series Analysis

```python
# Create time-indexed data for growth curves
dates = pd.date_range('2023-10-01', periods=24, freq='H')
growth_time_series = pd.DataFrame({
    'Time': dates,
    'OD600': np.random.exponential(0.1, 24).cumsum() + 0.05,
    'Temperature': 37 + np.random.normal(0, 0.5, 24),
    'pH': 7.0 + np.random.normal(0, 0.1, 24)
})

growth_time_series.set_index('Time', inplace=True)

# Resample data (e.g., hourly to 4-hourly)
growth_4h = growth_time_series.resample('4H').mean()

# Rolling averages for smoothing
growth_time_series['OD600_smooth'] = growth_time_series['OD600'].rolling(window=3).mean()
```

### Data Transformation

```python
# Apply transformations
df['Log_Growth_Rate'] = np.log10(df['Growth_Rate'])
df['Normalized_OD'] = (df['Final_OD'] - df['Final_OD'].mean()) / df['Final_OD'].std()

# Apply custom functions
def growth_category(rate):
    if rate < 0.3:
        return 'Slow'
    elif rate < 0.7:
        return 'Medium'
    else:
        return 'Fast'

df['Growth_Category'] = df['Growth_Rate'].apply(growth_category)

# Group operations
df['Organism_Mean_Growth'] = df.groupby('Organism')['Growth_Rate'].transform('mean')
df['Relative_Growth'] = df['Growth_Rate'] / df['Organism_Mean_Growth']
```

## Best Practices for Laboratory Data

1. **Use descriptive column names** following consistent naming conventions
2. **Set appropriate data types** early in analysis
3. **Document data sources** and preprocessing steps
4. **Handle missing values explicitly** rather than ignoring them
5. **Validate data ranges** for biological plausibility
6. **Keep raw data separate** from processed data
7. **Use categorical data types** for efficiency with repeated values
