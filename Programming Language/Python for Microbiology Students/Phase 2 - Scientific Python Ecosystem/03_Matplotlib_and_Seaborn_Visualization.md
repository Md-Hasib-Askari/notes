# Matplotlib and Seaborn for Visualization

Data visualization is crucial for understanding experimental results, presenting findings, and identifying patterns in microbiology research.

## Basic Plotting with Matplotlib

### Line Plots for Growth Curves

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Sample growth curve data
time_hours = np.array([0, 2, 4, 6, 8, 10, 12, 24])
od_values = np.array([0.05, 0.12, 0.28, 0.56, 0.89, 1.24, 1.45, 1.38])

# Basic growth curve plot
plt.figure(figsize=(10, 6))
plt.plot(time_hours, od_values, 'o-', linewidth=2, markersize=8)
plt.xlabel('Time (hours)')
plt.ylabel('OD₆₀₀')
plt.title('Bacterial Growth Curve - E. coli K12')
plt.grid(True, alpha=0.3)
plt.show()

# Multiple strains comparison
strains = {
    'E. coli K12': [0.05, 0.12, 0.28, 0.56, 0.89, 1.24, 1.45, 1.38],
    'S. aureus': [0.04, 0.08, 0.18, 0.35, 0.58, 0.82, 1.12, 1.28],
    'B. subtilis': [0.06, 0.10, 0.22, 0.41, 0.65, 0.88, 1.08, 1.15]
}

plt.figure(figsize=(12, 8))
for strain, values in strains.items():
    plt.plot(time_hours, values, 'o-', label=strain, linewidth=2, markersize=6)

plt.xlabel('Time (hours)', fontsize=12)
plt.ylabel('OD₆₀₀', fontsize=12)
plt.title('Growth Comparison of Different Bacterial Strains', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Scatter Plots for Correlation Analysis

```python
# Temperature vs Growth Rate relationship
temperatures = np.array([25, 30, 35, 37, 40, 42, 45, 50])
growth_rates = np.array([0.2, 0.4, 0.7, 0.9, 0.8, 0.6, 0.3, 0.1])

plt.figure(figsize=(10, 6))
plt.scatter(temperatures, growth_rates, s=100, alpha=0.7, color='red')
plt.xlabel('Temperature (°C)')
plt.ylabel('Growth Rate (h⁻¹)')
plt.title('Effect of Temperature on Bacterial Growth Rate')

# Add trend line
z = np.polyfit(temperatures, growth_rates, 2)
p = np.poly1d(z)
plt.plot(temperatures, p(temperatures), "--", alpha=0.8, color='blue')
plt.grid(True, alpha=0.3)
plt.show()
```

### Bar Plots for Categorical Data

```python
# Antibiotic resistance profiles
antibiotics = ['Ampicillin', 'Kanamycin', 'Chloramphenicol', 'Tetracycline', 'Streptomycin']
resistance_rates = [65, 25, 15, 40, 30]  # Percentage resistant

plt.figure(figsize=(10, 6))
bars = plt.bar(antibiotics, resistance_rates, color=['red', 'orange', 'green', 'blue', 'purple'])
plt.xlabel('Antibiotic')
plt.ylabel('Resistance Rate (%)')
plt.title('Antibiotic Resistance Profile - E. coli Clinical Isolates')
plt.xticks(rotation=45)

# Add value labels on bars
for bar, rate in zip(bars, resistance_rates):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{rate}%', ha='center', va='bottom')

plt.tight_layout()
plt.show()
```

### Histograms for Distribution Analysis

```python
# Distribution of colony sizes
np.random.seed(42)
colony_diameters = np.random.normal(2.5, 0.8, 200)  # mm

plt.figure(figsize=(10, 6))
plt.hist(colony_diameters, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
plt.xlabel('Colony Diameter (mm)')
plt.ylabel('Frequency')
plt.title('Distribution of Colony Sizes')
plt.axvline(np.mean(colony_diameters), color='red', linestyle='--', 
            label=f'Mean: {np.mean(colony_diameters):.2f} mm')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Customizing Plots

### Publication-Ready Figures

```python
# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})

# Create subplot layout for multiple analyses
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Growth curve
ax1.plot(time_hours, od_values, 'o-', color='blue', linewidth=2)
ax1.set_xlabel('Time (hours)')
ax1.set_ylabel('OD₆₀₀')
ax1.set_title('A) Growth Curve')
ax1.grid(True, alpha=0.3)

# Temperature effect
ax2.scatter(temperatures, growth_rates, s=100, color='red', alpha=0.7)
ax2.set_xlabel('Temperature (°C)')
ax2.set_ylabel('Growth Rate (h⁻¹)')
ax2.set_title('B) Temperature Effect')
ax2.grid(True, alpha=0.3)

# Resistance profile
ax3.bar(range(len(antibiotics)), resistance_rates, color='green', alpha=0.7)
ax3.set_xlabel('Antibiotic')
ax3.set_ylabel('Resistance Rate (%)')
ax3.set_title('C) Resistance Profile')
ax3.set_xticks(range(len(antibiotics)))
ax3.set_xticklabels(antibiotics, rotation=45)

# Colony size distribution
ax4.hist(colony_diameters, bins=15, color='purple', alpha=0.7, edgecolor='black')
ax4.set_xlabel('Colony Diameter (mm)')
ax4.set_ylabel('Frequency')
ax4.set_title('D) Colony Size Distribution')

plt.tight_layout()
plt.savefig('bacterial_analysis_figure.png', dpi=300, bbox_inches='tight')
plt.show()
```

## Statistical Visualizations with Seaborn

### Box Plots for Group Comparisons

```python
import seaborn as sns

# Create sample dataset
data = pd.DataFrame({
    'Organism': ['E_coli']*20 + ['S_aureus']*20 + ['B_subtilis']*20,
    'Temperature': [25, 30, 37, 42]*15,
    'Growth_Rate': np.random.normal(0.6, 0.2, 60)
})

# Adjust growth rates based on temperature and organism
data.loc[data['Temperature'] == 37, 'Growth_Rate'] *= 1.5
data.loc[data['Organism'] == 'S_aureus', 'Growth_Rate'] *= 0.8
data.loc[data['Organism'] == 'B_subtilis', 'Growth_Rate'] *= 0.7

plt.figure(figsize=(12, 6))
sns.boxplot(data=data, x='Temperature', y='Growth_Rate', hue='Organism')
plt.title('Growth Rate Distribution by Temperature and Organism')
plt.ylabel('Growth Rate (h⁻¹)')
plt.xlabel('Temperature (°C)')
plt.show()
```

### Heatmaps for Matrix Data

```python
# Microplate data visualization
plate_data = np.random.exponential(0.5, (8, 12)) + 0.1
row_labels = [f'Row_{i+1}' for i in range(8)]
col_labels = [f'Col_{i+1}' for i in range(12)]

plt.figure(figsize=(12, 6))
sns.heatmap(plate_data, 
            xticklabels=col_labels,
            yticklabels=row_labels,
            annot=True, 
            fmt='.2f', 
            cmap='viridis',
            cbar_kws={'label': 'OD₆₀₀'})
plt.title('96-Well Plate OD₆₀₀ Readings')
plt.tight_layout()
plt.show()
```

### Correlation Matrix

```python
# Correlation between experimental variables
experimental_data = pd.DataFrame({
    'Temperature': np.random.normal(37, 5, 100),
    'pH': np.random.normal(7.0, 0.5, 100),
    'Growth_Rate': np.random.normal(0.6, 0.2, 100),
    'Final_OD': np.random.normal(1.2, 0.3, 100),
    'Generation_Time': np.random.normal(45, 10, 100)
})

# Calculate correlation matrix
correlation_matrix = experimental_data.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, 
            annot=True, 
            cmap='coolwarm', 
            center=0,
            square=True,
            fmt='.2f')
plt.title('Correlation Matrix of Experimental Variables')
plt.tight_layout()
plt.show()
```

### Pair Plots for Multivariate Analysis

```python
# Comprehensive variable relationships
plt.figure(figsize=(12, 10))
g = sns.pairplot(experimental_data, 
                 diag_kind='hist',
                 plot_kws={'alpha': 0.6})
g.fig.suptitle('Pairwise Relationships in Experimental Data', y=1.02)
plt.show()
```

## Specialized Microbiology Plots

### Dose-Response Curves

```python
# Antibiotic concentration vs growth inhibition
concentrations = np.logspace(-2, 2, 20)  # 0.01 to 100 μg/ml
inhibition = 100 / (1 + (concentrations / 10)**-2)  # Sigmoid curve

plt.figure(figsize=(10, 6))
plt.semilogx(concentrations, inhibition, 'o-', linewidth=2, markersize=6)
plt.xlabel('Antibiotic Concentration (μg/ml)')
plt.ylabel('Growth Inhibition (%)')
plt.title('Dose-Response Curve - Antibiotic Sensitivity')
plt.grid(True, alpha=0.3)
plt.axhline(50, color='red', linestyle='--', alpha=0.7, label='IC₅₀')
plt.legend()
plt.show()
```

### Survival Curves

```python
# Bacterial survival under stress conditions
time_points = np.arange(0, 61, 5)  # 0 to 60 minutes
survival_rate = np.exp(-time_points / 20) * 100  # Exponential decay

plt.figure(figsize=(10, 6))
plt.semilogy(time_points, survival_rate, 'o-', linewidth=2, color='red')
plt.xlabel('Exposure Time (minutes)')
plt.ylabel('Survival Rate (%)')
plt.title('Bacterial Survival Under UV Treatment')
plt.grid(True, alpha=0.3)
plt.show()
```

## Best Practices for Scientific Visualization

1. **Use clear, descriptive titles and labels**
2. **Include units in axis labels**
3. **Choose appropriate color schemes** (colorblind-friendly)
4. **Add error bars** when showing experimental data
5. **Use consistent styling** across related figures
6. **Save figures in high resolution** for publications
7. **Include sample sizes** in figure captions
8. **Make figures self-explanatory** with proper legends
