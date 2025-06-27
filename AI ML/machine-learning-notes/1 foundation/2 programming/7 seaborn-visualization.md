# Seaborn Visualization - Brief Notes

## 1. Setup and Basics

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")

# Load sample datasets
tips = sns.load_dataset('tips')
iris = sns.load_dataset('iris')
flights = sns.load_dataset('flights')
```

## 2. Distribution Plots

### Histogram and KDE
```python
# Histogram with KDE
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
sns.histplot(tips['total_bill'], kde=True, bins=20)
plt.title('Histogram with KDE')

plt.subplot(1, 3, 2)
sns.distplot(tips['total_bill'])  # Deprecated, use histplot
plt.title('Distribution Plot')

plt.subplot(1, 3, 3)
sns.kdeplot(tips['total_bill'], shade=True)
plt.title('KDE Plot')

plt.tight_layout()
plt.show()
```

### Box and Violin Plots
```python
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
sns.boxplot(x='day', y='total_bill', data=tips)
plt.title('Box Plot')

plt.subplot(1, 3, 2)
sns.violinplot(x='day', y='total_bill', data=tips)
plt.title('Violin Plot')

plt.subplot(1, 3, 3)
sns.swarmplot(x='day', y='total_bill', data=tips, size=3)
plt.title('Swarm Plot')

plt.tight_layout()
plt.show()
```

## 3. Regression Plots

### Scatter and Regression
```python
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
sns.scatterplot(x='total_bill', y='tip', data=tips)
plt.title('Scatter Plot')

plt.subplot(1, 3, 2)
sns.regplot(x='total_bill', y='tip', data=tips)
plt.title('Regression Plot')

plt.subplot(1, 3, 3)
sns.residplot(x='total_bill', y='tip', data=tips)
plt.title('Residual Plot')

plt.tight_layout()
plt.show()
```

### Joint Plots
```python
# Joint plot with regression
sns.jointplot(x='total_bill', y='tip', data=tips, kind='reg', height=6)
plt.show()

# Joint plot with hexbin
sns.jointplot(x='total_bill', y='tip', data=tips, kind='hex', height=6)
plt.show()
```

## 4. Categorical Plots

### Bar and Count Plots
```python
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
sns.countplot(x='day', data=tips)
plt.title('Count Plot')

plt.subplot(1, 3, 2)
sns.barplot(x='day', y='total_bill', data=tips)
plt.title('Bar Plot')

plt.subplot(1, 3, 3)
sns.pointplot(x='day', y='total_bill', data=tips)
plt.title('Point Plot')

plt.tight_layout()
plt.show()
```

### Categorical Scatter
```python
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
sns.stripplot(x='day', y='total_bill', data=tips, jitter=True)
plt.title('Strip Plot')

plt.subplot(1, 3, 2)
sns.swarmplot(x='day', y='total_bill', data=tips)
plt.title('Swarm Plot')

plt.subplot(1, 3, 3)
sns.boxplot(x='day', y='total_bill', data=tips)
sns.swarmplot(x='day', y='total_bill', data=tips, color='black', alpha=0.5)
plt.title('Combined Plot')

plt.tight_layout()
plt.show()
```

## 5. Heatmaps and Correlation

### Correlation Heatmap
```python
# Correlation matrix
corr = tips.select_dtypes(include=[np.number]).corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0,
            square=True, fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()
```

### Pivot Table Heatmap
```python
# Pivot table heatmap
flights_pivot = flights.pivot(index='month', columns='year', values='passengers')

plt.figure(figsize=(10, 8))
sns.heatmap(flights_pivot, annot=True, fmt='d', cmap='YlOrRd')
plt.title('Flights Passengers Heatmap')
plt.show()
```

## 6. Pair Plots and Grids

### Pair Plot
```python
# Pair plot for all numerical features
sns.pairplot(iris, hue='species', diag_kind='kde', height=2.5)
plt.show()

# Customized pair plot
g = sns.pairplot(iris, hue='species', 
                 plot_kws={'alpha': 0.6, 's': 80, 'edgecolor': 'k'},
                 height=3)
g.map_diag(plt.hist, histtype='step', linewidth=3)
plt.show()
```

### Facet Grid
```python
# FacetGrid for categorical data
g = sns.FacetGrid(tips, col='time', row='smoker', height=4, aspect=.7)
g.map(sns.scatterplot, 'total_bill', 'tip', alpha=0.7)
g.add_legend()
plt.show()

# FacetGrid with different plot types
g = sns.FacetGrid(tips, col='day', height=4, aspect=.5)
g.map(sns.boxplot, 'sex', 'total_bill')
plt.show()
```

## 7. Styling and Themes

### Built-in Styles
```python
# Available styles
styles = ['darkgrid', 'whitegrid', 'dark', 'white', 'ticks']

for style in styles:
    sns.set_style(style)
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x='total_bill', y='tip', data=tips)
    plt.title(f'Style: {style}')
    plt.show()
```

### Color Palettes
```python
# Color palette examples
palettes = ['deep', 'muted', 'bright', 'pastel', 'dark', 'colorblind']

for palette in palettes:
    sns.set_palette(palette)
    plt.figure(figsize=(8, 4))
    sns.boxplot(x='day', y='total_bill', data=tips)
    plt.title(f'Palette: {palette}')
    plt.show()

# Custom palette
custom_palette = sns.color_palette("husl", 8)
sns.palplot(custom_palette)
plt.show()
```

## 8. Advanced Customization

### Custom Color Maps
```python
# Diverging color map
plt.figure(figsize=(10, 8))
corr = tips.select_dtypes(include=[np.number]).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

sns.heatmap(corr, mask=mask, annot=True, cmap='RdBu_r', center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title('Masked Correlation Heatmap')
plt.show()
```

### Multi-plot Figures
```python
# Complex multi-plot layout
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Distribution plot
sns.histplot(tips['total_bill'], kde=True, ax=axes[0, 0])
axes[0, 0].set_title('Total Bill Distribution')

# Box plot
sns.boxplot(x='day', y='total_bill', data=tips, ax=axes[0, 1])
axes[0, 1].set_title('Total Bill by Day')

# Scatter plot
sns.scatterplot(x='total_bill', y='tip', hue='time', data=tips, ax=axes[1, 0])
axes[1, 0].set_title('Bill vs Tip by Time')

# Count plot
sns.countplot(x='day', hue='sex', data=tips, ax=axes[1, 1])
axes[1, 1].set_title('Count by Day and Gender')

plt.tight_layout()
plt.show()
```

## 9. ML-Specific Visualizations

### Feature Distributions by Target
```python
# Assuming a classification dataset
def plot_feature_distributions(df, target_col, feature_cols, figsize=(15, 10)):
    n_features = len(feature_cols)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for i, feature in enumerate(feature_cols):
        if i < len(axes):
            sns.boxplot(x=target_col, y=feature, data=df, ax=axes[i])
            axes[i].set_title(f'{feature} by {target_col}')
    
    # Hide unused subplots
    for i in range(len(feature_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

# Example usage
# plot_feature_distributions(iris, 'species', ['sepal_length', 'sepal_width', 'petal_length'])
```

### Model Performance Visualization
```python
# Confusion matrix heatmap
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
```

## 10. Quick Reference

### Common Plot Types
```python
# Quick plotting functions
def quick_eda_plots(df, target=None):
    """Generate common EDA plots"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlations')
    plt.show()
    
    # Distribution plots
    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for i, col in enumerate(numeric_cols):
        if target:
            sns.boxplot(x=target, y=col, data=df, ax=axes[i])
        else:
            sns.histplot(df[col], kde=True, ax=axes[i])
        axes[i].set_title(f'{col} Distribution')
    
    for i in range(len(numeric_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

# Usage
# quick_eda_plots(tips, 'time')
```

### Best Practices
- Use appropriate plot types for data types
- Always add titles and labels
- Consider color-blind friendly palettes
- Use figure size appropriately
- Add context with annotations
- Combine multiple plots for comprehensive analysis

### Common Parameters
```python
# Figure aesthetics
sns.set_context("notebook")  # paper, notebook, talk, poster
sns.set_style("whitegrid")   # darkgrid, whitegrid, dark, white, ticks
sns.set_palette("husl")      # deep, muted, bright, pastel, dark, colorblind

# Plot parameters
common_params = {
    'figsize': (10, 6),
    'alpha': 0.7,
    'linewidth': 2,
    'edgecolor': 'black'
}
```