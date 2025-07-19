# Advanced Pandas Techniques

## Merging and Joining Datasets
Combine experimental data with metadata using merge operations.

```python
import pandas as pd

# Merge growth data with sample info
growth_data = pd.DataFrame({'sample_id': ['S1', 'S2'], 'od600': [0.5, 0.8]})
metadata = pd.DataFrame({'sample_id': ['S1', 'S2'], 'strain': ['E.coli', 'B.subtilis']})

merged_df = pd.merge(growth_data, metadata, on='sample_id')
# Use how='left'/'right'/'outer' for different join types
```

## Time Series Analysis for Growth Curves
Analyze bacterial growth data over time using pandas time series functions.

```python
import numpy as np

# Create time series data
dates = pd.date_range('2024-01-01', periods=12, freq='H')
growth_df = pd.DataFrame({
    'time': dates,
    'od600': [0.1, 0.15, 0.22, 0.35, 0.55, 0.82, 1.1, 1.25, 1.32, 1.34, 1.35, 1.36]
})

growth_df.set_index('time', inplace=True)

# Moving average to smooth noisy data
growth_df['moving_avg'] = growth_df['od600'].rolling(window=3).mean()

# Resample to different time intervals
hourly_avg = growth_df.resample('2H').mean()
```

## Pivot Tables and Advanced Grouping
Reorganize data for analysis using pivot tables and groupby operations.

```python
# Antibiotic susceptibility data
data = pd.DataFrame({
    'strain': ['E.coli', 'E.coli', 'S.aureus', 'S.aureus'],
    'antibiotic': ['Ampicillin', 'Tetracycline', 'Ampicillin', 'Tetracycline'],
    'inhibition_zone': [15, 18, 12, 20]
})

# Create pivot table
pivot = pd.pivot_table(data, values='inhibition_zone', 
                      index='strain', columns='antibiotic', aggfunc='mean')

# Advanced grouping
grouped = data.groupby(['strain', 'antibiotic'])['inhibition_zone'].agg(['mean', 'std'])
```

## Performance Optimization
Optimize pandas operations for large microbial datasets.

```python
# Use vectorized operations instead of loops
df['log_od'] = np.log(df['od600'])  # Fast vectorized operation

# Convert strings to categories for memory efficiency
df['strain'] = df['strain'].astype('category')

# Use query() for faster filtering
filtered = df.query('od600 > 0.5 and strain == "E.coli"')

# Use loc for efficient indexing
subset = df.loc[df['od600'] > 0.5, ['strain', 'od600']]
```

These techniques handle large biological datasets efficiently while maintaining code readability and performance.
