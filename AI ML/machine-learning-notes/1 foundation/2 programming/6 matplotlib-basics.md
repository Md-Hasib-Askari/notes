# Matplotlib Basics

## Overview
Matplotlib is Python's fundamental plotting library, providing a MATLAB-like interface for creating static, animated, and interactive visualizations. It's essential for data analysis, machine learning visualization, and scientific computing.

## Getting Started

### Installation and Import
```python
# Installation: pip install matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set up matplotlib for better display
plt.style.use('default')  # or 'seaborn', 'ggplot', etc.
%matplotlib inline  # For Jupyter notebooks
```

### Basic Plot Structure
```python
# Basic plot anatomy
fig, ax = plt.subplots()  # Create figure and axis
ax.plot([1, 2, 3, 4], [1, 4, 2, 3])  # Plot data
plt.show()  # Display plot

# Alternative simple syntax
plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
plt.show()
```

## Basic Plot Types

### 1. Line Plots
```python
# Simple line plot
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title('Sine Wave')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.grid(True)
plt.show()

# Multiple lines
y2 = np.cos(x)
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='sin(x)', linewidth=2)
plt.plot(x, y2, label='cos(x)', linestyle='--', linewidth=2)
plt.title('Trigonometric Functions')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Different line styles and markers
plt.figure(figsize=(12, 8))
styles = ['-', '--', '-.', ':']
markers = ['o', 's', '^', 'D']
colors = ['blue', 'red', 'green', 'orange']

for i in range(4):
    y_offset = np.sin(x + i * np.pi/4)
    plt.plot(x[::5], y_offset[::5], 
             linestyle=styles[i], 
             marker=markers[i], 
             color=colors[i],
             label=f'Line {i+1}',
             markersize=6,
             markerfacecolor='white',
             markeredgewidth=2)

plt.title('Different Line Styles and Markers')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 2. Scatter Plots
```python
# Basic scatter plot
np.random.seed(42)
n = 100
x = np.random.randn(n)
y = 2 * x + np.random.randn(n)

plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.6)
plt.title('Basic Scatter Plot')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.grid(True, alpha=0.3)
plt.show()

# Advanced scatter plot with colors and sizes
plt.figure(figsize=(12, 8))
colors = np.random.rand(n)
sizes = 1000 * np.random.rand(n)

scatter = plt.scatter(x, y, c=colors, s=sizes, alpha=0.6, cmap='viridis')
plt.colorbar(scatter, label='Color Scale')
plt.title('Advanced Scatter Plot')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.grid(True, alpha=0.3)
plt.show()

# Categorical scatter plot
categories = np.random.choice(['A', 'B', 'C'], n)
colors_dict = {'A': 'red', 'B': 'blue', 'C': 'green'}
colors_list = [colors_dict[cat] for cat in categories]

plt.figure(figsize=(10, 6))
for category in ['A', 'B', 'C']:
    mask = categories == category
    plt.scatter(x[mask], y[mask], 
               c=colors_dict[category], 
               label=f'Category {category}',
               alpha=0.7,
               s=60)

plt.title('Categorical Scatter Plot')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 3. Bar Charts
```python
# Simple bar chart
categories = ['A', 'B', 'C', 'D', 'E']
values = [23, 17, 35, 29, 12]

plt.figure(figsize=(10, 6))
bars = plt.bar(categories, values, color='skyblue', edgecolor='navy', linewidth=1.2)
plt.title('Simple Bar Chart')
plt.xlabel('Categories')
plt.ylabel('Values')

# Add value labels on bars
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             str(value), ha='center', va='bottom')

plt.show()

# Horizontal bar chart
plt.figure(figsize=(10, 6))
plt.barh(categories, values, color='lightcoral', edgecolor='darkred')
plt.title('Horizontal Bar Chart')
plt.xlabel('Values')
plt.ylabel('Categories')
plt.show()

# Grouped bar chart
data = {
    'Group 1': [20, 35, 30, 35, 27],
    'Group 2': [25, 25, 15, 30, 35],
    'Group 3': [15, 30, 35, 20, 25]
}

x = np.arange(len(categories))
width = 0.25

plt.figure(figsize=(12, 6))
for i, (group, values) in enumerate(data.items()):
    plt.bar(x + i * width, values, width, label=group, alpha=0.8)

plt.title('Grouped Bar Chart')
plt.xlabel('Categories')
plt.ylabel('Values')
plt.xticks(x + width, categories)
plt.legend()
plt.show()

# Stacked bar chart
plt.figure(figsize=(10, 6))
bottom = np.zeros(len(categories))
colors = ['red', 'green', 'blue']

for i, (group, values) in enumerate(data.items()):
    plt.bar(categories, values, bottom=bottom, label=group, color=colors[i], alpha=0.8)
    bottom += values

plt.title('Stacked Bar Chart')
plt.xlabel('Categories')
plt.ylabel('Values')
plt.legend()
plt.show()
```

### 4. Histograms
```python
# Simple histogram
np.random.seed(42)
data = np.random.normal(50, 15, 1000)

plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Simple Histogram')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.show()

# Multiple histograms
data1 = np.random.normal(50, 15, 1000)
data2 = np.random.normal(60, 10, 1000)

plt.figure(figsize=(10, 6))
plt.hist(data1, bins=30, alpha=0.5, label='Dataset 1', color='blue')
plt.hist(data2, bins=30, alpha=0.5, label='Dataset 2', color='red')
plt.title('Multiple Histograms')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Histogram with density and statistics
plt.figure(figsize=(12, 8))
n, bins, patches = plt.hist(data, bins=30, density=True, 
                           color='lightgreen', edgecolor='black', alpha=0.7)

# Add normal distribution curve
mu, sigma = np.mean(data), np.std(data)
x_curve = np.linspace(data.min(), data.max(), 100)
y_curve = ((1 / (sigma * np.sqrt(2 * np.pi))) * 
           np.exp(-0.5 * ((x_curve - mu) / sigma) ** 2))
plt.plot(x_curve, y_curve, 'r-', linewidth=2, label='Normal Distribution')

# Add statistics
plt.axvline(mu, color='red', linestyle='--', linewidth=2, label=f'Mean: {mu:.2f}')
plt.axvline(mu + sigma, color='orange', linestyle='--', linewidth=2, label=f'Mean + SD: {mu+sigma:.2f}')
plt.axvline(mu - sigma, color='orange', linestyle='--', linewidth=2, label=f'Mean - SD: {mu-sigma:.2f}')

plt.title('Histogram with Normal Distribution Overlay')
plt.xlabel('Values')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Advanced Plot Types

### 5. Box Plots
```python
# Simple box plot
data_groups = [np.random.normal(50, 15, 100),
               np.random.normal(60, 10, 100),
               np.random.normal(55, 20, 100)]

plt.figure(figsize=(10, 6))
plt.boxplot(data_groups, labels=['Group A', 'Group B', 'Group C'])
plt.title('Box Plot Comparison')
plt.ylabel('Values')
plt.grid(True, alpha=0.3)
plt.show()
```

### 6. Pie Charts
```python
# Simple pie chart
sizes = [30, 25, 20, 15, 10]
labels = ['A', 'B', 'C', 'D', 'E']
colors = ['gold', 'lightcoral', 'lightskyblue', 'lightgreen', 'plum']
explode = (0.1, 0, 0, 0, 0)  # explode first slice

plt.figure(figsize=(8, 8))
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
plt.title('Pie Chart Example')
plt.axis('equal')  # Equal aspect ratio
plt.show()
```

## Customization and Styling

### Colors and Styles
```python
# Color options
plt.figure(figsize=(12, 8))

# Named colors
x = np.linspace(0, 10, 100)
plt.plot(x, np.sin(x), color='red', label='Named color')
plt.plot(x, np.sin(x + 0.5), color='#ff6600', label='Hex color')
plt.plot(x, np.sin(x + 1), color=(0.2, 0.8, 0.3), label='RGB tuple')
plt.plot(x, np.sin(x + 1.5), color='C4', label='Default color cycle')

plt.title('Different Color Specifications')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Line styles and markers
plt.figure(figsize=(12, 8))
line_styles = ['-', '--', '-.', ':']
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

for i, (style, marker) in enumerate(zip(line_styles, markers)):
    y = np.sin(x + i * 0.5)
    plt.plot(x[::5], y[::5], linestyle=style, marker=marker, 
             markersize=8, linewidth=2, label=f'Style {i+1}')

plt.title('Line Styles and Markers')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Labels, Titles, and Annotations
```python
# Comprehensive labeling
plt.figure(figsize=(12, 8))
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y, linewidth=2, color='blue')

# Title with formatting
plt.title('Sine Wave Analysis', fontsize=16, fontweight='bold', pad=20)

# Axis labels with formatting
plt.xlabel('Time (seconds)', fontsize=12, fontweight='bold')
plt.ylabel('Amplitude', fontsize=12, fontweight='bold')

# Text annotations
plt.text(5, 0.5, 'Peak', fontsize=12, ha='center',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# Arrow annotation
plt.annotate('Zero crossing', xy=(np.pi, 0), xytext=(4, -0.5),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=12, ha='center')

# Custom text box
textstr = 'Function: y = sin(x)\nFrequency: 1/(2π)\nAmplitude: 1'
props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=props)

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Legends and Color Bars
```python
# Advanced legend customization
plt.figure(figsize=(12, 8))
x = np.linspace(0, 10, 100)

plt.plot(x, np.sin(x), label='sin(x)', linewidth=3)
plt.plot(x, np.cos(x), label='cos(x)', linewidth=3)
plt.plot(x, np.tan(x), label='tan(x)', linewidth=3)

# Custom legend
plt.legend(loc='upper right', frameon=True, shadow=True, 
          fancybox=True, framealpha=0.9, fontsize=12)

plt.title('Trigonometric Functions with Custom Legend')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.ylim(-2, 2)
plt.grid(True, alpha=0.3)
plt.show()

# Color bar example
plt.figure(figsize=(10, 8))
x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 50)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) * np.cos(Y)

contour = plt.contourf(X, Y, Z, levels=20, cmap='viridis')
cbar = plt.colorbar(contour)
cbar.set_label('Function Value', rotation=270, labelpad=20, fontsize=12)

plt.title('Contour Plot with Color Bar')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.show()
```

## Subplots and Layouts

### Basic Subplots
```python
# Simple subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Line plot
x = np.linspace(0, 10, 100)
axes[0, 0].plot(x, np.sin(x))
axes[0, 0].set_title('Sine Wave')
axes[0, 0].grid(True)

# Plot 2: Scatter plot
axes[0, 1].scatter(np.random.randn(50), np.random.randn(50))
axes[0, 1].set_title('Random Scatter')
axes[0, 1].grid(True)

# Plot 3: Bar chart
categories = ['A', 'B', 'C', 'D']
values = [23, 17, 35, 29]
axes[1, 0].bar(categories, values)
axes[1, 0].set_title('Bar Chart')

# Plot 4: Histogram
data = np.random.normal(0, 1, 1000)
axes[1, 1].hist(data, bins=20)
axes[1, 1].set_title('Histogram')

plt.tight_layout()
plt.show()
```

### Advanced Subplot Layouts
```python
# Complex subplot layout using GridSpec
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(15, 10))
gs = GridSpec(3, 3, figure=fig)

# Large plot spanning multiple cells
ax1 = fig.add_subplot(gs[0:2, 0:2])
x = np.linspace(0, 10, 100)
ax1.plot(x, np.sin(x), linewidth=2)
ax1.set_title('Main Plot (Large)', fontsize=14)
ax1.grid(True)

# Small plot top right
ax2 = fig.add_subplot(gs[0, 2])
ax2.bar(['A', 'B', 'C'], [1, 2, 3])
ax2.set_title('Small Plot 1')

# Small plot middle right
ax3 = fig.add_subplot(gs[1, 2])
ax3.scatter(np.random.randn(20), np.random.randn(20))
ax3.set_title('Small Plot 2')

# Bottom spanning plot
ax4 = fig.add_subplot(gs[2, :])
ax4.hist(np.random.normal(0, 1, 1000), bins=30)
ax4.set_title('Bottom Spanning Plot')

plt.tight_layout()
plt.show()

# Subplots with shared axes
fig, axes = plt.subplots(2, 2, figsize=(12, 8), 
                        sharex=True, sharey=True)

for i, ax in enumerate(axes.flat):
    x = np.linspace(0, 10, 100)
    y = np.sin(x + i * np.pi/4)
    ax.plot(x, y, linewidth=2)
    ax.set_title(f'Subplot {i+1}')
    ax.grid(True, alpha=0.3)

plt.suptitle('Shared Axes Example', fontsize=16)
plt.tight_layout()
plt.show()
```

## Machine Learning Visualization Examples

### Model Performance Visualization
```python
# ROC Curve
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                          n_informative=2, n_clusters_per_class=1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)
y_proba = model.predict_proba(X_test)[:, 1]

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()
```

### Learning Curves
```python
# Learning curve visualization
def plot_learning_curve(train_scores, val_scores, train_sizes):
    plt.figure(figsize=(10, 6))
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                     alpha=0.2, color='blue')
    
    plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                     alpha=0.2, color='red')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy Score')
    plt.title('Learning Curves')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.show()

# Example data (replace with actual learning curve data)
train_sizes = np.array([100, 200, 300, 400, 500])
train_scores = np.random.rand(5, 3) * 0.2 + 0.7  # Simulated scores
val_scores = np.random.rand(5, 3) * 0.3 + 0.6     # Simulated scores

plot_learning_curve(train_scores, val_scores, train_sizes)
```

### Feature Importance
```python
# Feature importance visualization
feature_names = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5']
importance_scores = [0.35, 0.25, 0.20, 0.15, 0.05]

plt.figure(figsize=(10, 6))
bars = plt.barh(feature_names, importance_scores, color='lightgreen', edgecolor='darkgreen')
plt.xlabel('Importance Score')
plt.title('Feature Importance')

# Add value labels
for bar, score in zip(bars, importance_scores):
    plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
             f'{score:.2f}', va='center')

plt.tight_layout()
plt.show()
```

## Saving and Exporting Figures

### Save Options
```python
# Create a sample plot
plt.figure(figsize=(10, 6))
x = np.linspace(0, 10, 100)
plt.plot(x, np.sin(x), linewidth=2)
plt.title('Sample Plot for Saving')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.grid(True)

# Save in different formats
plt.savefig('plot.png', dpi=300, bbox_inches='tight')  # High resolution PNG
plt.savefig('plot.pdf', bbox_inches='tight')           # PDF (vector format)
plt.savefig('plot.svg', bbox_inches='tight')           # SVG (vector format)
plt.savefig('plot.jpg', dpi=150, bbox_inches='tight')  # JPEG

plt.show()

# Save with transparency
plt.figure(figsize=(8, 6))
plt.plot(x, np.sin(x), linewidth=3, color='blue')
plt.title('Plot with Transparent Background')
plt.savefig('transparent_plot.png', transparent=True, dpi=300, bbox_inches='tight')
plt.show()
```

## Style Sheets and Themes

### Using Style Sheets
```python
# Available styles
print("Available styles:")
print(plt.style.available)

# Apply different styles
styles = ['seaborn', 'ggplot', 'dark_background', 'bmh']

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
x = np.linspace(0, 10, 100)

for i, (ax, style) in enumerate(zip(axes.flat, styles)):
    with plt.style.context(style):
        ax.plot(x, np.sin(x), linewidth=2, label='sin(x)')
        ax.plot(x, np.cos(x), linewidth=2, label='cos(x)')
        ax.set_title(f'Style: {style}')
        ax.legend()
        ax.grid(True)

plt.tight_layout()
plt.show()
```

### Custom Style
```python
# Create custom style
custom_style = {
    'axes.facecolor': '#f0f0f0',
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.2,
    'axes.grid': True,
    'grid.color': 'white',
    'grid.linewidth': 1,
    'xtick.color': 'black',
    'ytick.color': 'black',
    'font.size': 12,
    'figure.facecolor': 'white'
}

plt.rcParams.update(custom_style)

plt.figure(figsize=(10, 6))
x = np.linspace(0, 10, 100)
plt.plot(x, np.sin(x), linewidth=2, color='navy')
plt.title('Custom Style Example')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.show()

# Reset to default
plt.rcParams.update(plt.rcParamsDefault)
```

## Best Practices

### Design Principles
1. **Clarity**: Clear titles, labels, and legends
2. **Simplicity**: Avoid unnecessary elements
3. **Consistency**: Use consistent colors and styles
4. **Accessibility**: Consider color-blind friendly palettes

### Performance Tips
```python
# Efficient plotting for large datasets
def plot_large_dataset(x, y, max_points=1000):
    """Plot large datasets efficiently by sampling"""
    if len(x) > max_points:
        indices = np.random.choice(len(x), max_points, replace=False)
        x_sample = x[indices]
        y_sample = y[indices]
    else:
        x_sample, y_sample = x, y
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x_sample, y_sample, alpha=0.6)
    plt.title(f'Large Dataset Visualization (showing {len(x_sample)} points)')
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.show()

# Example usage
large_x = np.random.randn(10000)
large_y = np.random.randn(10000)
plot_large_dataset(large_x, large_y)
```

## Learning Objectives
- [ ] Create all basic plot types (line, scatter, bar, histogram)
- [ ] Customize plots with colors, styles, and annotations
- [ ] Design effective charts for ML visualization
- [ ] Use subplots and complex layouts
- [ ] Apply appropriate styling and themes
- [ ] Save plots in different formats
- [ ] Optimize plotting for large datasets
- [ ] Follow visualization best practices

## Quick Reference

**Basic Plots**: `plt.plot()`, `plt.scatter()`, `plt.bar()`, `plt.hist()`
**Customization**: `plt.title()`, `plt.xlabel()`, `plt.legend()`, `plt.grid()`
**Subplots**: `plt.subplots()`, `GridSpec`, `plt.subplot()`
**Styling**: `plt.style.use()`, `plt.rcParams`, color/linestyle options
**Saving**: `plt.savefig()` with format, dpi, and bbox options

## Common Parameters

**Colors**: 'red', '#ff0000', (1,0,0), 'C0'
**Line Styles**: '-', '--', '-.', ':'
**Markers**: 'o', 's', '^', 'D', '*'
**Figure Size**: `figsize=(width, height)` in inches
**DPI**: `dpi=300` for high resolution

## Next Steps
- Learn Seaborn for statistical plotting
- Explore interactive plotting with Plotly
- Study advanced matplotlib features (animations, widgets)
- Practice with real ML datasets
- Develop personal visualization style guide