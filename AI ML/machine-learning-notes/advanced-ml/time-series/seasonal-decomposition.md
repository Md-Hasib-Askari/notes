# Seasonal Decomposition

## Overview
Seasonal decomposition is a technique used to separate a time series into its trend, seasonal, and residual (noise) components. This helps in understanding underlying patterns and improving forecasting models.

## Key Concepts
- **Trend**: Long-term movement in the data
- **Seasonality**: Regular, repeating patterns
- **Residual**: Irregular or random noise after removing trend and seasonality
- **Additive Model**: Time series = Trend + Seasonality + Residual
- **Multiplicative Model**: Time series = Trend × Seasonality × Residual

## Common Methods
- **Classical Decomposition**: Separates components using moving averages
- **STL (Seasonal-Trend Decomposition using Loess)**: Robust and flexible method for decomposition

## Example (STL in Python)
```python
import pandas as pd
from statsmodels.tsa.seasonal import STL

# Load data
data = pd.read_csv('time_series.csv', index_col='date', parse_dates=True)

# STL decomposition
stl = STL(data['value'], period=12)
result = stl.fit()

# Plot components
result.plot()
```

## Applications
- **Forecasting**: Improves model accuracy by modeling components separately
- **Anomaly Detection**: Identifies unusual patterns in residuals
- **Data Exploration**: Reveals hidden trends and cycles

## Best Practices
1. Visualize decomposed components to understand the data structure.
2. Choose the right model (additive or multiplicative) based on data characteristics.
3. Use STL for robust decomposition, especially with complex seasonality.

## Resources
- **Statsmodels Documentation**: STL and seasonal decomposition
- **Pandas**: Data manipulation and visualization
